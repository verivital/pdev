'''
Plot discrete reachable set of error with different sizes of space-step
Dung Tran: 12/2017
'''

import matplotlib.pyplot as plt
from engine.fem import Fem1D
from engine.verifier import ReachSetAssembler
from engine.plot import Plot
import numpy as np


if __name__ == '__main__':

    ##################################################
    # generate dPde automaton
    FEM = Fem1D()
    L = 10.0    # length of rod
    num_mesh_points = [10, 20, 40]     # number of mesh points
    colors = ['r', 'm', 'g']
    step = 0.1
    x = 8.0

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    pl3 = Plot()

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
        _, e_dset, _ = RSA.get_dreachset(dPde, toTimeStep)    # compute discrete reachable set
        e_lines_at_x_8_list = []
        for i in xrange(0, toTimeStep + 1):
            e_lines_at_x_8, _, _, _, _ = e_dset[i].get_lines_set()
            e_lines_at_x_8_list.append(e_lines_at_x_8[x_ind])

        ############################################################
        # Plot dreach set of e at x = 8

        ax3 = pl3.plot_vlines(ax3, time_list.tolist(), e_lines_at_x_8_list, colors=colors[j], linestyles='solid')

    ax3.legend([r'$h = {}$'.format(L / num_mesh_points[0]),
                r'$h = {}$'.format(format(L / num_mesh_points[1])),
                r'$h = {}$'.format(format(L / num_mesh_points[2]))])

    ax3.set_ylim(0, 0.9)
    ax3.set_xlim(-0.2, 10.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$t$', fontsize=20)
    plt.ylabel(r'$\tilde{}(x={},t)$'.format('e', x), fontsize=20)
    fig3.suptitle('Error Vs. Space-Step', fontsize=25)
    fig3.savefig('err_vs_spacestep.pdf')
    plt.show()
