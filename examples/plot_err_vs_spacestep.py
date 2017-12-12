'''
Plot discrete reachable set of error with different sizes of space-step
Dung Tran: 12/2017
'''

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from engine.fem import Fem1D
from engine.verifier import ReachSetAssembler
from engine.plot import Plot
import numpy as np


if __name__ == '__main__':

    ##################################################
    # generate dPde automaton
    FEM = Fem1D()
    L = 10.0    # length of rod
    num_mesh_points = [10, 20]     # number of mesh points
    colors = ['r', 'g']
    labels = ['h = 1.0', 'h = 0.5']
    step = 0.1
    x = 8.0

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    pl1 = Plot()

    for j in xrange(0, len(num_mesh_points)):
        num_mesh_point = num_mesh_points[j]
        mesh_grid = np.arange(0, num_mesh_point + 1, step=1)
        mesh_points = np.multiply(mesh_grid, L / num_mesh_point)
        print "\nmesh_points = {}".format(mesh_points)
        toTimeStep = 100    # number of time steps
        time_grid = np.arange(0, toTimeStep + 1, step=1)
        time_list = np.multiply(time_grid, step)
        x_ind = int(x * num_mesh_point / L) - 1
        xlist = mesh_points[1: mesh_points.shape[0] - 1]
        x_dom = [2.0, 4.0]    # domain of input function

        dPde = Fem1D().get_dPde_automaton(mesh_points.tolist(), x_dom, step)

        alpha_range = (0.8, 1.1)
        beta_range = (0.9, 1.1)
        dPde.set_perturbation(alpha_range, beta_range)

        ############################################################
        # compute error dicrete reachable set
        RSA = ReachSetAssembler()
        _, e_inspace, _, _, _, _ = RSA.get_interpolationset(dPde, toTimeStep)
        e_boxes = e_inspace[toTimeStep].get_2D_boxes(alpha_range, beta_range)

        ax1 = pl1.plot_boxes(ax1, e_boxes, facecolor=colors[j], edgecolor=colors[j])

    red_patch = mpatches.Patch(color='r', label='$h = 1.0$')
    green_patch = mpatches.Patch(color='g', label='$h = 0.5$')
    ax1.set_ylim(0.0, 0.4)
    ax1.set_xlim(0, 10.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel(r'$\bar{e}(x,t=10s)$', fontsize=20)
    fig1.suptitle('Error Vs. Space-Step at $t=10s$', fontsize=25)
    fig1.savefig('err_vs_space_step.pdf')
    plt.legend(handles=[red_patch, green_patch])
    plt.show()
