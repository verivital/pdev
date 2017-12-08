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
    toTimeStep = 2    # number of time steps
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
    u_inspace, e_inspace, bl_inspace, u_set, e_set, bl_set = RSA.get_interpolationset(dPde, toTimeStep)

    ############################################################
    # Plot interpolation reachable set

    # plot interpolation in space set
    u_sp = u_inspace[toTimeStep]
    e_sp = e_inspace[toTimeStep]
    bl_sp = bl_inspace[toTimeStep]

    u_sp_boxes, _, _, _, _ = u_sp.get_2D_boxes(alpha_range, beta_range)
    e_sp_boxes, _, _, _, _ = e_sp.get_2D_boxes(alpha_range, beta_range)
    bl_sp_boxes, _, _, _, _ = bl_sp.get_2D_boxes(alpha_range, beta_range)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    pl1 = Plot()
    ax1 = pl1.plot_boxes(ax1, u_sp_boxes, facecolor='b', edgecolor='b')
    plt.show()
