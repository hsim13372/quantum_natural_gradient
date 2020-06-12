''' Module for running VQE iteration that stores state and parameter history '''

import pennylane as qml
from pennylane.qnodes import PassthruQNode
import numpy as np

def init_cost_fn(dev, circuit, coeffs, obs): 

    '''
    instantiate the cost function.

    args: 

	dev: configure the device using qml.device('device name', wires='# of wires')

	circuit: define the circuit (params, wires ='# of wires')

	coeffs (float): coefficients of the Hamiltonian expression

	obs: observables in the Hamiltonian expression

    ** Example **

    A Hamiltonian can be created by simply passing the list of coefficients as well as the list of observables:

    >>> coeffs = [1, 1]
    >>> obs = [qml.PauliX(0), qml.PauliZ(0)]
    >>> H = qml.Hamiltonian(coeffs, obs)
    >>> print(H)
    (1) [X0] + (1) [Z0]

    '''

    qnode = PassthruQNode(circuit, dev)

    H = qml.Hamiltonian(coeffs, obs)

    cost_fn = qml.VQECost(circuit, H, dev)

    return cost_fn

def run_vqe(cost_fn, max_iter, initial_params, opt, opt_step, dev, diag_approx=False):

    '''
    run a VQE trial.
    
    args:
    
        cost_fn (VQECost): the cost function we are trying to optimize
    
        max_iter (int): number iterations of energy improvement for the VQE run.
        
        initial_params (nump.ndarray): the initial params we're starting with 
    
        opt (string): type of optimizer we're using for the VQE run 
            (can be qml.QNGOptimizer, qml.GradientDescentOptimizer)
        
        opt_step (float): the optimization step you're using for the optimizer.
        
        if opt=QNGOptimizer,
            diag_approx (boolean): if using the quantum natural gradient descent optimization,
                this tells us if you want to use block-diagonal or diagonal approximation form
                for the fubini-study metric.
    
    '''

    if opt =='GradientDescentOptimizer':
        opt = qml.GradientDescentOptimizer(opt_step)

    elif opt =='QNGOptimizer':
        opt = qml.QNGOptimizer(opt_step, diag_approx)
        
    else:
        raise ValueError('Use either QNGOptimizer of GradientDescentOptimizer')

    conv_tol = 1e-06

    params = initial_params
    prev_energy = cost_fn(params)

    cost_history = []
    param_history = []
    states_history = []

    for n in range(max_iter):
        params = opt.step(cost_fn, params)
        param_history.append(params)

        energy = cost_fn(params)
        conv = np.abs(energy - prev_energy)

        state_step = dev._state
        states_history.append(state_step)

        if n % 20 == 0:
          print('Iteration = {:}'.format(n) ,'Energy = {:.8f} Ha,'.format(energy), 'Convergence parameter = {'
              ':.8f} Ha'.format(conv), "State", state_step)

        if conv <= conv_tol:
            break

        prev_energy = energy
        cost_history.append(energy)

        state_final = dev._state
        
        opt_steps = n

    print()
    print('Final value of the ground-state energy = {:.8f} Ha'.format(energy))
    print()
    print("Final state", state_final)
    print()
    print('Number of iterations = ', opt_steps)
        
    return cost_history, param_history, states_history, opt_steps