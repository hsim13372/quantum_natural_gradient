
"""Module for launching VQE calculations."""

import numpy as np
import pennylane as qml


def run_vqe(cost_fn, max_iter, initial_params, opt_name, step_size, 
            conv_tol=1e-6, diag_approx=False, lam=0, print_freq=20):
    """Launches a VQE calculation.
    
    Args:
    =====
    cost_fn : VQECost
        VQE cost function we are trying to optimize
    max_iter : int
        Maximum number of optimization iterations
    initial_params : numpy.ndarray
        Vector of initial parameter values
    opt_name : str
        Name of optimizer. Valid options are: QNGOptimizer or GradientDescentOptimizer.
    step_size : float
        Stepsize or learning rate of the optimizer
    conv_tol : float
        Convergence tolerance for optimizer (relative improvement in energy)
    diag_approx : bool
        If using QNGOptimizer, diag_approx is an option for using the block-diagonal 
        approximation to the Fubini-Study metric. If false, the diagonal approximation
        is used.
    lam : float
        Regularizer term for QNGOptimizer
    print_freq : int
        Optimizer progress printing frequency
       
    Returns:
    ========
    energy_history : list/numpy.ndarray
        History of energies
    n : int
        Number of steps taken to optimize (could be less than max_iter if converged)
    
    """
    energy_history = []

    if opt_name =='GradientDescentOptimizer':
        opt = qml.GradientDescentOptimizer(stepsize=step_size)

    elif opt_name =='QNGOptimizer':
        opt = qml.QNGOptimizer(stepsize=step_size, diag_approx=diag_approx, lam=lam)

    else:
        raise ValueError('Use either QNGOptimizer of GradientDescentOptimizer.')

    params = initial_params
    prev_energy = cost_fn(params)
    energy_history = [prev_energy]

    for n in range(max_iter):
        params = opt.step(cost_fn, params)
        energy = cost_fn(params)
        conv = np.abs(energy - prev_energy)

        if n % print_freq == 0:
            print('Iteration = {:},  Energy = {:.8f} Ha,  Convergence parameter = {'
                  ':.8f} Ha'.format(n, energy, conv))

        if conv <= conv_tol:
            break
        
        energy_history.append(energy)
        
        # Update energy
        prev_energy = energy
        
    print()
    print("Final value of the energy = {:.8f}".format(energy))
    print("Number of iterations = ", n)
        
    return energy_history, n

def run_single_qubit_vqe(cost_fn, dev, max_iter, initial_params, opt_name, 
                         step_size, conv_tol=1e-6, diag_approx=False):
    """Launches a VQE calculation for single-qubit systems, where we may be interested
    in plotting the optimization path on the Bloch sphere, and thus, need to save the 
    statevector and circuit parameter history.
    
    Args:
    =====
    cost_fn : VQECost
        VQE cost function we are trying to optimize
    dev : qml.Device
        Quantum simulator/device
    max_iter : int
        Maximum number of optimization iterations
    initial_params : numpy.ndarray
        Vector of initial parameter values
    opt_name : str
        Name of optimizer. Valid options are: QNGOptimizer and GradientDescentOptimizer.
    step_size : float
        Stepsize or learning rate of the optimizer
    conv_tol : float
        Convergence tolerance for optimizer (relative improvement in energy)
    diag_approx : bool
        If using QNGOptimizer, diag_approx is an option for using the block-diagonal 
        approximation to the Fubini-Study metric. If false, the diagonal approximation
        is used.
       
    Returns:
    ========
    energy_history : list/numpy.ndarray
        History of energies
    n : int
        Number of steps taken to optimize (could be less than max_iter if converged)
    state_history : list
        History of state vectors/wavefunctions
    param_history : list
        History of parameters
    """
    energy_history = []

    if opt_name =='GradientDescentOptimizer':
        opt = qml.GradientDescentOptimizer(stepsize=step_size)

    elif opt_name =='QNGOptimizer':
        opt = qml.QNGOptimizer(stepsize=step_size, diag_approx=diag_approx)

    else:
        raise ValueError('Use either QNGOptimizer of GradientDescentOptimizer.')

    params = initial_params
    prev_energy = cost_fn(params)
    energy_history = [prev_energy]
    state_history = [dev.state]
    param_history = [params]

    for n in range(max_iter):
        params = opt.step(cost_fn, params)
        energy = cost_fn(params)
        conv = np.abs(energy - prev_energy)
        
        if n % 20 == 0:
            print('Iteration = {:},  Energy = {:.8f} Ha,  Convergence parameter = {'
                  ':.8f} Ha'.format(n, energy, conv))

        if conv <= conv_tol:
            break
        
        energy_history.append(energy)
        state_history.append(dev.state)
        param_history.append(params)
        
        # Update energy
        prev_energy = energy
        
    print()
    print("Final value of the energy = {:.8f}".format(energy))
    print("Number of iterations = ", n)

    return energy_history, n, state_history, param_history
