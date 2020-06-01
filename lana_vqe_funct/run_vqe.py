import pennylane as qml
import numpy as np

class Run_VQE:
    """
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
    
    """
    
    def __init__(
        self, cost_fn, max_iter, initial_params, opt, opt_step, diag_approx=False
    ):
        self.energy_history = []
    
        if opt =='GradientDescentOptimizer':
            opt = qml.GradientDescentOptimizer(opt_step)
        
        elif opt =='QNGOptimizer':
            opt = qml.QNGOptimizer(opt_step, diag_approx)
        
        else:
            raise ValueError('Use either QNGOptimizer of GradientDescentOptimizer')
            
        energy_history = []
        conv_tol = 1e-06
        
        params = initial_params
        prev_energy = cost_fn(params)

        for n in range(max_iter):
            params = opt.step(cost_fn, params)
            energy = cost_fn(params)
            conv = np.abs(energy - prev_energy)

            if n % 20 == 0:
                print('Iteration = {:},  Ground-state energy = {:.8f} Ha,  Convergence parameter = {'
                      ':.8f} Ha'.format(n, energy, conv))

            if conv <= conv_tol:
                break

            self.energy_history.append(energy)
            prev_energy = energy