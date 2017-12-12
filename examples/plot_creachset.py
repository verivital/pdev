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
    u_inspace, e_inspace, bl_inspace, u_set, e_set, bl_set = RSA.get_interpolationset(dPde, toTimeStep)

    ############################################################
    # Plot interpolation reachable set

    # plot interpolation in space set
    bl_sp_boxes = bl_inspace[toTimeStep].get_2D_boxes(alpha_range, beta_range)
    bl_boxes_3d = []
    for i in xrange(0, len(bl_set)):
        box3d = bl_set[i].get_3D_boxes(alpha_range, beta_range)
        bl_boxes_3d.append(box3d)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    pl1 = Plot()
    ax1 = pl1.plot_boxes(ax1, bl_sp_boxes, facecolor='c', edgecolor='c')
    ax1.legend([r'$\bar{u}(x,t=10s)$'])
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlim(0, 10.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel(r'$\bar{u}(x,t=10s)$', fontsize=20)
    fig1.suptitle('Continuous Reachable Set at $t=10s$', fontsize=25)
    fig1.savefig('con_reachset_t_10.pdf')
    plt.show()

    # plot interpolation set in both space and time
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    pl2 = Plot()
    ax2 = pl2.plot_interpolationset(ax2, bl_boxes_3d, facecolor='c', linewidth=0.5, edgecolor='r')
    ax2.set_xlim(0, 10.5)
    ax2.set_ylim(0, 10.5)
    ax2.set_zlim(0, 1.0)
    ax2.tick_params(axis='z', labelsize=20)
    ax2.tick_params(axis='x', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.set_xlabel('$x$', fontsize=20)
    ax2.set_ylabel('$t$', fontsize=20)
    ax2.set_zlabel(r'$\bar{u}(x,t)$', fontsize=20)
    fig2.suptitle('3-Dimensional Reachable Set', fontsize=25)
    fig2.savefig('reachset_3D.pdf')
    plt.show()
