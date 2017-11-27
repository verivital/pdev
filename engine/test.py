'''
This is test file
Dung Tran: Nov/2017
'''

import math
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
import numpy as np
from engine.fem import Fem1D
from engine.verifier import Verifier
from engine.set import RectangleSet
from engine.plot import Plot
from engine.interpolation import Interpolation


if __name__ == '__main__':

    ##################################################
    # test Fem1D class
    FEM = Fem1D()
    mesh_points = [0.0, 0.5, 1.0, 1.5, 2.0]    # generate mesh points
    step = 0.1    # time step of FEM
    x_dom = [0.5, 1.0]    # domain of input function

    mass_mat, stiff_mat, load_vec, init_vector, dPde = Fem1D().get_dPde_automaton(mesh_points, x_dom, step)
    print "\nmass matrix = \n{}".format(mass_mat.todense())
    print "\nstiff matrix = \n{}".format(stiff_mat.todense())
    print "\nload vector = \n{}".format(load_vec.todense())
    print "\ninit vector = \n{}".format(init_vector.todense())
    print "\ndPde matrix_a = {}".format(dPde.matrix_a.todense())
    print "\ndPde vector_b = {}".format(dPde.vector_b.todense())

    # get trace with initial vector u0
    u = FEM.get_trace(dPde.matrix_a, dPde.vector_b, init_vector, step=0.1, num_steps=4)

    alpha_beta_range = np.array([[0, 1], [1, 1]])    # set perturbation range
    perturbation = dPde.set_perturbation(alpha_beta_range)

    ########################################################
    # test verifier class

    verifier = Verifier()
    toTimeStep = 10
    reachable_set = verifier.compute_reach_set(
        dPde, toTimeStep)    # compute reachable set

    direction_matrix = lil_matrix((1, dPde.matrix_a.shape[0]), dtype=float)
    direction_matrix[0, math.ceil(dPde.matrix_a.shape[0] / 2)] = 1
    unsafe_vector = lil_matrix((1, 1), dtype=float)
    unsafe_vector[0, 0] = -1

    dPde.set_unsafe_set(
        direction_matrix.tocsc(),
        unsafe_vector.tocsc())    # set unsafe set
    print"\nSafety of discreted Pde:"
    verifier.on_fly_check(dPde, 10)

    ############################################################
    # test interpolation class
    intpl = Interpolation()
    intpl_inspace_list = intpl.interpolate_in_space_for_all_timesteps(mesh_points, reachable_set)
    interpolation_set = intpl.interpolate_in_time_and_space(step, intpl_inspace_list)

    #########################################################
    # test plot class

    rectangle_set_list = []
    for i in xrange(0, 3):
        xmin = float(i)
        ymin = float(i)
        xmax = float(i + 2)
        ymax = float(i + 2)
        rect = RectangleSet()
        rect.set_bounds(xmin, xmax, ymin, ymax)
        rectangle_set_list.append(rect)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    pl = Plot()
    ax = pl.plot_boxes(ax, rectangle_set_list, facecolor='b', edgecolor='None')
    plt.show()    # plot boxes

    tlist = [1, 2, 3]
    xmin = [0.2, 0.4, 0.5]
    xmax = [0.6, 1.0, 1.2]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    pl1 = Plot()
    ax1 = pl.plot_vlines(
        ax1,
        tlist,
        xmin,
        xmax,
        colors='b',
        linestyles='solid')
    plt.show()    # plot vlines
