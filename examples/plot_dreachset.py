'''
This is for ploting dreach set
Dung Tran: Dec/2017
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
    num_mesh_points = 20    # number of mesh points
    mesh_grid = np.arange(0, num_mesh_points + 1, step=1)
    mesh_points = np.multiply(mesh_grid, L / num_mesh_points)
    print "\nmesh_points = {}".format(mesh_points)
    step = 0.1    # time step of FEM
    toTimeStep = 100    # number of time steps
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
    # compute reachable set
    RSA = ReachSetAssembler()
    u_dset, e_dset, bloated_dset = RSA.get_dreachset(dPde, toTimeStep)    # compute discrete reachable set

    x = 8.0
    x_ind = int(x / 0.5) - 1
    u_lines_at_x_8_list = []
    e_lines_at_x_8_list = []
    bl_lines_at_x_8_list = []

    for i in xrange(0, toTimeStep + 1):
        u_lines_at_x_8, _, _, _, _ = u_dset[i].get_lines_set()
        e_lines_at_x_8, _, _, _, _ = e_dset[i].get_lines_set()
        bl_lines_at_x_8, _, _, _, _ = bloated_dset[i].get_lines_set()
        u_lines_at_x_8_list.append(u_lines_at_x_8[x_ind])
        e_lines_at_x_8_list.append(e_lines_at_x_8[x_ind])
        bl_lines_at_x_8_list.append(bl_lines_at_x_8[x_ind])

    u_lines_at_t_10, _, _, _, _ = u_dset[toTimeStep].get_lines_set()
    e_lines_at_t_10, _, _, _, _ = e_dset[toTimeStep].get_lines_set()
    bloated_lines_at_t_10, _, _, _, _ = bloated_dset[toTimeStep].get_lines_set()

    ############################################################
    # Plot discrete reachable set

    # Plot dreach set of u, e, at final time step

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    pl1 = Plot()
    ax1 = pl1.plot_vlines(ax1, xlist.tolist(), u_lines_at_t_10, colors='b', linestyles='solid')
    ax1 = pl1.plot_vlines(ax1, xlist.tolist(), e_lines_at_t_10, colors='r', linestyles='solid')
    ax1.legend([r'$\tilde{u}(x,t=10s)$', r'$\tilde{e}(x,t=10s)$'])
    ax1.set_ylim(0, 0.7)
    ax1.set_xlim(0, 10.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel(r'$\tilde{u}(x,t=10s)$ and $\tilde{e}(x,t=10s)$', fontsize=20)
    fig1.suptitle('Discrete Reachable Sets at $t=10s$', fontsize=25)
    fig1.savefig('dreachset_t_10.pdf')
    plt.show()

    # plot dreach set of bloated set at final time step

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    pl2 = Plot()
    ax2 = pl2.plot_vlines(ax2, xlist.tolist(), bloated_lines_at_t_10, colors='c', linestyles='solid')
    ax2.legend([r'$\bar{u}(x,t=10s)$ = $\tilde{u}(x,t=10s) + \tilde{e}(x,t=10s)$'])
    ax2.set_ylim(0, 0.9)
    ax2.set_xlim(0, 10.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel(r'$\bar{u}(x,t=10s)$', fontsize=20)
    fig2.suptitle('Bloated Discrete Reachable Set at $t=10s$', fontsize=25)
    fig2.savefig('bloated_dreachset_t_10.pdf')
    plt.show()

    # plot dreach set at x over time

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    pl3 = Plot()
    ax3 = pl3.plot_vlines(ax3, time_list.tolist(), u_lines_at_x_8_list, colors='b', linestyles='solid')
    ax3 = pl3.plot_vlines(ax3, time_list.tolist(), e_lines_at_x_8_list, colors='r', linestyles='solid')
    ax3.legend([r'$\tilde{u}(x=8.0,t)$', r'$\tilde{e}(x=8.0,t)$'])
    ax3.set_ylim(0, 0.9)
    ax3.set_xlim(-0.2, 10.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$t$', fontsize=20)
    plt.ylabel(r'$\tilde{u}(x=8.0,t)$ and $\tilde{e}(x=8.0,t)$', fontsize=20)
    fig3.suptitle('Discrete Reachable Sets at $x = 8.0$', fontsize=25)
    fig3.savefig('dreachset_x_8.pdf')
    plt.show()

    # plot bloated dreach set at x over time

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    pl4 = Plot()
    ax4 = pl4.plot_vlines(ax4, time_list.tolist(), bl_lines_at_x_8_list, colors='c', linestyles='solid')
    ax4.legend([r'$\bar{u}(x=8.0,t) = \tilde{u}(x=8.0,t) + \tilde{e}(x=8.0,t)$'])
    ax4.set_ylim(0, 0.9)
    ax4.set_xlim(-0.2, 10.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$t$', fontsize=20)
    plt.ylabel(r'$\bar{u}(x=8.0,t)$', fontsize=20)
    fig4.suptitle('Bloated Discrete Reachable Set at $x = 8.0$', fontsize=25)
    fig4.savefig('bloated_dreachset_x_8.pdf')
    plt.show()
