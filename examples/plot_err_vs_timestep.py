'''
Plot discrete reachable set of error with different sizes of time-step
Dung Tran: 12/2017
'''

import matplotlib.pyplot as plt
from engine.fem import Fem1D
from engine.verifier import ReachSetAssembler
import matplotlib.patches as mpatches
from engine.plot import Plot
import numpy as np


if __name__ == '__main__':

    ##################################################
    # generate dPde automaton
    FEM = Fem1D()
    L = 10.0    # length of rod
    num_mesh_points = 20    # number of mesh points
    mesh_grid = np.arange(0, num_mesh_points + 1, step=1)
    mesh_points = np.multiply(mesh_grid, L / num_mesh_points)
    print "\nmesh_points = {}".format(mesh_points)
    x = 8.0
    x_ind = int(x / 0.5) - 1
    steps = [0.1, 0.05]    # time step of FEM
    colors = ['r', 'g']

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    pl3 = Plot()

    for j in xrange(0, len(steps)):
        step = steps[j]
        toTimeStep = int(10.0 / step)    # number of time steps
        time_grid = np.arange(0, toTimeStep + 1, step=1)
        time_list = np.multiply(time_grid, step)
        print "\ntime_list = {}".format(time_list)
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
        ax3 = pl3.plot_boxes(ax3, e_boxes, facecolor=colors[j], edgecolor=colors[j])

    ax3.set_ylim(0.0, 0.4)
    ax3.set_xlim(0, 10.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel(r'$\bar{e}(x,t=10s)$', fontsize=20)
    red_patch = mpatches.Patch(color='r', label='$k = 0.1$')
    green_patch = mpatches.Patch(color='g', label='$k = 0.05$')
    plt.legend(handles=[red_patch, green_patch])
    fig3.suptitle('Error Vs. Space-Step at $t=10s$', fontsize=25)
    fig3.savefig('err_vs_time_step.pdf')
    plt.show()
