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
    mesh_points = np.arange(0.0, 10 + 0.5, 0.5)    # generate mesh points
    print "\nmesh_points = {}".format(mesh_points)
    step = 0.1    # time step of FEM
    toTimeStep = 100    # number of time steps
    time_list = np.arange(0, toTimeStep * step + step, step)
    xlist = mesh_points[1: mesh_points.shape[0] - 1]
    x_dom = [2.0, 4.0]    # domain of input function

    mass_mat, stiff_mat, load_vec, init_vector, dPde = Fem1D().get_dPde_automaton(mesh_points.tolist(), x_dom, step)
    print "\nmass matrix = \n{}".format(mass_mat.todense())
    print "\nstiff matrix = \n{}".format(stiff_mat.todense())
    print "\nload vector = \n{}".format(load_vec.todense())
    print "\ninit vector = \n{}".format(init_vector.todense())
    print "\ndPde matrix_a = {}".format(dPde.matrix_a.todense())

    alpha_range = (0.8, 1.1)
    beta_range = (0.9, 1.1)
    dPde.set_perturbation(alpha_range, beta_range)

    ############################################################
    # compute reachable set
    RSA = ReachSetAssembler()
    u_dset, e_dset, bloated_dset = RSA.get_dreachset(dPde, toTimeStep)    # compute discrete reachable set

    x_u_min = []
    x_u_max = []
    x_e_min = []
    x_e_max = []
    x_bl_min = []
    x_bl_max = []
    x = 8.0
    x_ind = int(x / 0.5) - 1

    for i in xrange(0, toTimeStep + 1):
        mid_min_u, _, mid_max_u, _ = u_dset[i].get_min_max()
        x_u_min.append(mid_min_u[x_ind])
        x_u_max.append(mid_max_u[x_ind])

        mid_min_e, _, mid_max_e, _ = e_dset[i].get_min_max()
        x_e_min.append(mid_min_e[x_ind])
        x_e_max.append(mid_max_e[x_ind])

        mid_min_bl, _, mid_max_bl, _ = bloated_dset[i].get_min_max()
        x_bl_min.append(mid_min_bl[x_ind])
        x_bl_max.append(mid_max_bl[x_ind])

    u_min, u_min_points, u_max, u_max_points = u_dset[toTimeStep].get_min_max()
    e_min, e_min_points, e_max, e_max_points = e_dset[toTimeStep].get_min_max()
    bloated_min, bloated_min_points, bloated_max, bloated_max_points = bloated_dset[toTimeStep].get_min_max()

    print "\nu_min = \n{}".format(u_min)
    print "\nu_max = \n{}".format(u_max)
    print "\ne_min = \n{}".format(e_min)
    print "\ne_max = \n{}".format(e_max)
    print "\nbloated_min = \n{}".format(bloated_min)
    print "\nbloated_max = \n{}".format(bloated_max)

    # u_inspace, e_inspace, bl_inspace, u_set, e_set, bl_set = RSA.get_interpolationset(dPde, toTimeStep)

    ############################################################
    # Plot discrete reachable set

    # Plot dreach set of u, e, at final time step

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    pl1 = Plot()
    ax1 = pl1.plot_vlines(ax1, xlist.tolist(), u_min.tolist(), u_max.tolist(), colors='b', linestyles='solid')
    ax1 = pl1.plot_vlines(ax1, xlist.tolist(), e_min.tolist(), e_max.tolist(), colors='r', linestyles='solid')
    ax1.legend([r'$\tilde{u}(x,t=10s)$', r'$\tilde{e}(x,t=10s)$'])
    ax1.set_ylim(0, 0.7)
    ax1.set_xlim(0, 10.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel(r'$\tilde{u}(x,t=10s)$ and $\tilde{e}(x,t=10s)$', fontsize=20)
    fig1.suptitle('Discrete Reachable Sets', fontsize=25)
    fig1.savefig('dreachset.pdf')
    plt.show()

    # plot dreach set of bloated set at final time step

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    pl2 = Plot()
    ax2 = pl2.plot_vlines(ax2, xlist.tolist(), bloated_min.tolist(), bloated_max.tolist(), colors='c', linestyles='solid')
    ax2.legend([r'$\bar{u}(x,t=10s)$ = $\tilde{u}(x,t=10s) + \tilde{e}(x,t=10s)$'])
    ax2.set_ylim(0, 0.9)
    ax2.set_xlim(0, 10.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel(r'$\bar{u}(x,t=10s)$', fontsize=20)
    fig2.suptitle('Bloated Discrete Reachable Set', fontsize=25)
    fig2.savefig('bloated_dreachset.pdf')
    plt.show()

    # plot dreach set at x over time

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    pl3 = Plot()
    ax3 = pl3.plot_vlines(ax3, time_list.tolist(), x_u_min, x_u_max, colors='b', linestyles='solid')
    ax3 = pl3.plot_vlines(ax3, time_list.tolist(), x_e_min, x_e_max, colors='r', linestyles='solid')
    ax3.legend([r'$\tilde{u}(x=8.0,t)$', r'$\tilde{e}(x=8.0,t)$'])
    ax3.set_ylim(0, 0.9)
    ax3.set_xlim(0, 10.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$t$', fontsize=20)
    plt.ylabel(r'$\tilde{u}(x=8.0,t)$ and $\tilde{e}(x=8.0,t)$', fontsize=20)
    fig3.suptitle('Discrete Reachable Sets at $x = 8.0$', fontsize=25)
    fig3.savefig('x_dreachset.pdf')
    plt.show()

    # plot bloated dreach set at x over time

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    pl4 = Plot()
    ax4 = pl4.plot_vlines(ax4, time_list.tolist(), x_bl_min, x_bl_max, colors='c', linestyles='solid')
    ax4.legend([r'$\bar{u}(x=8.0,t) = \tilde{u}(x=8.0,t) + \tilde{e}(x=8.0,t)$'])
    ax4.set_ylim(0, 0.9)
    ax4.set_xlim(0, 10.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$t$', fontsize=20)
    plt.ylabel(r'$\bar{u}(x=8.0,t)$', fontsize=20)
    fig4.suptitle('Bloated Discrete Reachable Set at $x = 8.0$', fontsize=25)
    fig4.savefig('bloated_x_dreachset.pdf')
    plt.show()
