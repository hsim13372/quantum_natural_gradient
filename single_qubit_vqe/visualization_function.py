''' Module for visualizing optimization path/results '''

import scipy as sp

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np

import qutip as qt
from qutip import Bloch, basis

def plot_params(gd_param_history, qngd_param_history, plot_interval, figsize, linewidth):
    
    '''
    Plot the energy landscape of the cost function and optimization path. 

    This function requires the iterations for gradient descent and quantum natural gradient descent to have been run already. 
    Store paramters for gradient descent optimization as gd_param_history and paramters for quantum natural gradient optimization as qngd_param_history.

    Energies at different grid points have been pre-computed. 

    	args: 

    	gd_param_history (string): paramter history of vanilla gradient descent 

    	qngd_param_history (string): parameter history of quantum natural gradient descent 

    	plot_interval (int): set the interval at which points are plotted along the optimization path (ie, every 10th point)

    	figsize (np.array): two-dimensional array specifying size of figure (ie, [6,6])

    	linewidth (int): set width of line for optimization path

    '''
    # Discretize the parameter space
    theta0 = np.linspace(0.0, 2.0 * np.pi, 100)
    theta1 = np.linspace(0.0, 2.0 * np.pi, 100)

    # Load energy value at each point in parameter space
    parameter_landscape = np.load("param_landscape.npy")

    # Plot energy landscape
    fig, axes = plt.subplots(figsize=(6, 6))
    cmap = plt.cm.get_cmap("coolwarm")
    contour_plot = plt.contourf(theta0, theta1, parameter_landscape, cmap=cmap)
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')

    # Plot optimization path for gradient descent. Plot every 10th point.
    plt.plot(np.array(gd_param_history)[::plot_interval, 0],
             np.array(gd_param_history)[::plot_interval, 1],
             ".",
             color='g',
             linewidth=linewidth,
             label="Gradient descent",
            )

    plt.plot(np.array(gd_param_history)[:, 0],
             np.array(gd_param_history)[:, 1],
             "-",
             color='g',
             linewidth=linewidth,
            )

    # Plot optimization path for quantum natural gradient descent. Plot every 10th point.
    plt.plot(np.array(qngd_param_history)[::plot_interval, 0],
             np.array(qngd_param_history)[::plot_interval, 1],
             ".",
             color='k',
             linewidth=linewidth,
             label="Quantum natural gradient descent",
            )
    
    plt.plot(np.array(qngd_param_history)[:, 0],
             np.array(qngd_param_history)[:, 1],
             "-",
             color='k',
             linewidth=linewidth,
            )

    plt.legend()
    plt.show()

    return plt.show()


def prepare_plot_states(state_history):

    '''
    Convert the stored statevectors into QObjects in QuTip

    	args: 

    	state_history (list of numpy.ndarray): statevector history of optimization path
    
    '''

    # Convert statevectors for plotting into QObject
    plot_states = []
    lst = state_history

    for i in range(len(lst)):
        psi = lst[i]
        psi = psi/np.linalg.norm(psi)

        # Convert to QObject in QuTiP
        coords = [qt.Qobj(psi)]

        # Store all the statevecotrs 
        plot_states.append(coords)

    # Convert QObjects into coordinates to plot on the bloch sphere
    from qutip.expect import expect
    from qutip.operators import sigmax, sigmay, sigmaz

    coords_x = []
    coords_y = []
    coords_z = []

    for qobj in plot_states:
        st = qobj
        x = expect(sigmax(), st)
        y = expect(sigmay(), st)
        z = expect(sigmaz(), st)

        for i in range(len(x)):
            x_list = x[i]
            coords_x.append(x_list)

            y_list = y[i]
            coords_y.append(y_list)

            z_list = z[i]
            coords_z.append(z_list)
    
    return coords_x, coords_y, coords_z


def plot_bloch_sphere(qngd_coords_x, qngd_coords_y, qngd_coords_z, gd_coords_x, gd_coords_y, gd_coords_z, plot_interval, figsize, pointsize):
        
    '''
    Plot the optimization path on a bloch sphere. 

    	args: 

    	qngd_coords_x (list of floats): x-coordinates for plotting quantum natural gradient descent statevectors (same applies for qngd_coords_y and qngd_coords_z, for y and z coordinates, respectively)

    	gd_coords_x (list of floats): x-coordinates for plotting vanilla gradient descent statevectors (same applies for gd_coords_y and gd_coords_z, for y and z coordinates, respectively)

    	plot_interval (int): set the interval at which points are plotted along the optimization path (ie, every 10th point)

    	figsize (np.array): two-dimensional array specifying size of figure (ie, [6,6])

    	pointsize (int): set size of points on optimization path

    '''

    b = Bloch()
    b.sphere_alpha = 0.1
    b.figsize = figsize

    # colors #
    colors = ['g', 'g', 'k', 'k']
    b.point_color = list(colors)

    b.point_marker = 'o'
    b.point_size = [pointsize]

    # Add points for qngd
    b.add_points([qngd_coords_x[::plot_interval], qngd_coords_y[::plot_interval], qngd_coords_z[::plot_interval]])
    # Add line for qngd
    b.add_points([qngd_coords_x, qngd_coords_y, qngd_coords_z], 'l')

    # Add points for gd
    b.add_points([gd_coords_x[::plot_interval], gd_coords_y[::plot_interval], gd_coords_z[::plot_interval]])
    # Add line for gd
    b.add_points([gd_coords_x, gd_coords_y, gd_coords_z], 'l')

    b.show()