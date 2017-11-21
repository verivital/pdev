'''
This is test file
Dung Tran: Nov/2017
'''

import math
from scipy.sparse import lil_matrix
import numpy as np
from engine.fem import Fem1D
from engine.pde_automaton import DPdeAutomaton
from engine.verifier import DVerifier
from engine.set import Plot, RectangleSet
import matplotlib.pyplot as plt

if __name__ == '__main__':

    ##################################################
    # test Fem1D class
    FEM = Fem1D()
    mesh_points = [0.0, 0.5, 1.0, 1.5, 2.0]    # generate mesh points

    mass_matrix = FEM.mass_assembler(mesh_points)    # compute mass matrix M
    stiff_matrix = FEM.stiff_assembler(mesh_points)    # compute stiff matrix S

    fc = [1.0, 0.0, 2.0]    # define input function f
    fdom = [0.5, 1.0]    # domain of input function
    load_vector = FEM.load_assembler(
        mesh_points, fc, fdom)    # compute load vector

    print "\nmass matrix = \n{}".format(mass_matrix.todense())
    print "\nstiff matrix = \n{}".format(stiff_matrix.todense())
    print "\nload vector = \n{}".format(load_vector.todense())

    step = 0.1    # time step of FEM
    A, b = FEM.get_ode(mass_matrix, stiff_matrix, load_vector,
                       step)    # get the discreted ODE model

    print "\nA = {} \nb = {}".format(A.todense(), b.todense())
    print "\ntype of A is {}, type of b is {}".format(type(A), type(b))

    y = []
    c = [1, 2]    # parameters for initial function u0(x)
    u0_func = FEM.u0x_func(y, c)    # define initial function u0(x)
    # compute initial conditions
    u0 = FEM.get_init_cond(mesh_points, u0_func)
    print"\nu0 = {}".format(u0.todense())    # initial condition vector u0

    # get trace with initial vector u0
    u = FEM.get_trace(A, b, u0, step=0.1, num_steps=4)

    #########################################################
    # test DPdeAutomation object class
    dPde = DPdeAutomaton()
    dPde.set_dynamics(A, b, u0, step)    # set dynamic of discreted PDE
    alpha_beta_range = np.array([[0, 1], [1, 1]])    # set perturbation range
    perturbation = dPde.set_perturbation(alpha_beta_range)

    ########################################################
    # test Dverifier class

    verifier = DVerifier()
    toTimeStep = 10
    reachable_set = verifier.compute_reach_set(
        dPde, toTimeStep)    # compute reachable set

    direction_matrix = lil_matrix((1, A.shape[0]), dtype=float)
    direction_matrix[0, math.ceil(A.shape[0] / 2)] = 1
    unsafe_vector = lil_matrix((1, 1), dtype=float)
    unsafe_vector[0, 0] = -1

    dPde.set_unsafe_set(
        direction_matrix.tocsc(),
        unsafe_vector.tocsc())    # set unsafe set
    print"\nSafety of discreted Pde:"
    verifier.on_fly_check(dPde, 10)

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
    plt.show()

    tlist = [1, 2, 3]
    xmin = [0.2, 0.4, 0.5]
    xmax = [0.6, 1, 1.2]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    pl1 = Plot()
    ax1 = pl.plot_vlines(ax1, tlist, xmin, xmax, colors='b', linestyles='solid')
    plt.show()
