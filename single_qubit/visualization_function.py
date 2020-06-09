import pennylane as qml
import tensorflow as tf 

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

def plot_states(gd_state_history, qngd_state_history, plot_interval, figsize, pointsize):
        
    '''
    Plot the optimization path on a bloch sphere. 

    This function requires the iterations for gradient descent and quantum natural gradient descent to have been run already. 
    Store paramters for gradient descent optimization as gd_param_history and paramters for quantum natural gradient optimization as qngd_param_history.

    args: 

    gd_state_history (string): statevector history of vanilla gradient descent 

    qngd_state_history (string): statevector history of quantum natural gradient descent 

    plot_interval (int): set the interval at which points are plotted along the optimization path (ie, every 10th point)

    figsize (np.array): two-dimensional array specifying size of figure (ie, [6,6])

    pointsize (int): set size of points on optimization path

    '''
    # quantum natural gradient statevectors for plotting
    qngd_plot_states = []
    lst = qngd_state_history

    for i in range(len(lst)):
        psi = lst[i]
        psi = psi/np.linalg.norm(psi)

    # Convert to QObject in QuTiP
    coords = [qt.Qobj(psi)]

    qngd_plot_states.append(coords)

    # Vanilla gradient descent statevectors for plotting
    gd_plot_states = []
    lst = gd_state_history

    for i in range(len(lst)):
        gd_psi = lst[i]
        gd_psi = gd_psi/np.linalg.norm(gd_psi)

    # Convert to QObject in QuTiP
    coords = [qt.Qobj(gd_psi)]

    gd_plot_states.append(coords)
    from qutip.expect import expect
    from qutip.operators import sigmax, sigmay, sigmaz

    qngd_coords_x = []
    qngd_coords_y = []
    qngd_coords_z = []

    for qobj in qngd_plot_states:
        st = qobj
        x = expect(sigmax(), st)
        y = expect(sigmay(), st)
        z = expect(sigmaz(), st)

        for i in range(len(x)):
            x_list = x[i]
            qngd_coords_x.append(x_list)

            y_list = y[i]
            qngd_coords_y.append(y_list)

            z_list = z[i]
            qngd_coords_z.append(z_list)

    gd_coords_x = []
    gd_coords_y = []
    gd_coords_z = []

    for qobj in gd_plot_states:
        st = qobj
        x = expect(sigmax(), st)
        y = expect(sigmay(), st)
        z = expect(sigmaz(), st)

        for i in range(len(x)):
            x_list = x[i]
            gd_coords_x.append(x_list)

            y_list = y[i]
            gd_coords_y.append(y_list)

            z_list = z[i]
            gd_coords_z.append(z_list)
        
    ### settings for bloch sphere visualization ###

    b = Bloch()
    b.sphere_alpha = 0.1
    b.figsize = [8,8]

    # colors #
    colors = ['g', 'g', 'k', 'k']
    b.point_color = list(colors)

    b.point_marker = 'o'
    b.point_size = [pointsize]

    # Add points
    b.add_points([gd_coords_x[::plot_interval], gd_coords_y[::plot_interval], gd_coords_z[::plot_interval]])
    # Add line
    b.add_points([gd_coords_x, gd_coords_y, gd_coords_z], 'l')

    # Add points
    b.add_points([qngd_coords_x[::plot_interval], qngd_coords_y[::plot_interval], qngd_coords_z[::plot_interval]])
    # Add line
    b.add_points([qngd_coords_x, qngd_coords_y, qngd_coords_z], 'l')

    b.show()
