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
    x = 5.0
    x_ind = int(x / 0.5) - 1
    steps = [0.01, 0.1]    # time step of FEM
    colors = ['g', 'r']

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
        RSA = ReachSetAssembler()
        _, e_dset, _ = RSA.get_dreachset(dPde, toTimeStep)    # compute discrete reachable set
        e_lines_at_x_8_list = []

        for i in xrange(0, toTimeStep + 1):
            e_lines_at_x_8, _, _, _, _ = e_dset[i].get_lines_set()
            e_lines_at_x_8_list.append(e_lines_at_x_8[x_ind])

        ax3 = pl3.plot_vlines(ax3, time_list.tolist(), e_lines_at_x_8_list, colors=colors[j], linestyles='solid')

    ax3.legend([r'$k = 0.01$', r'$k = 0.1$'])
    ax3.set_ylim(0.0, 0.3)
    ax3.set_xlim(0, 10.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$t$', fontsize=20)
    plt.ylabel(r'$\bar{e}(x=5,t)$', fontsize=20)
    fig3.suptitle('Error Vs. Time-Step at $x=5$', fontsize=25)
    fig3.savefig('err_vs_time_step.pdf')
    plt.show()
