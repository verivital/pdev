'''
This is test file
Dung Tran: Nov/2017
'''

import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from engine.fem import Fem1D
from engine.verifier import Verifier, ReachSetAssembler
from engine.plot import Plot


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

    alpha_range = (0.99, 1.01)
    beta_range = (0.99, 1.01)
    dPde.set_perturbation(alpha_range, beta_range)

    ########################################################
    # test verifier class

    verifier = Verifier()
    toTimeStep = 2
    dis_reachable_set = verifier.get_dreach_set(dPde, toTimeStep)    # compute discrete reachable set
    dis_min_vec, _, dis_max_vec, _ = dis_reachable_set[toTimeStep - 1].get_min_max()

    unsafe_mat = lil_matrix((1, dPde.matrix_a.shape[0]), dtype=float)
    unsafe_mat[0, dPde.matrix_a.shape[0] - 1] = 1
    unsafe_vector = lil_matrix((1, 1), dtype=float)
    unsafe_vector[0, 0] = -1

    dPde.set_unsafe_set(unsafe_mat.tocsc(), unsafe_vector.tocsc())
    verifier.on_fly_check_dPde(dPde, toTimeStep)

    #residual = verifier.compute_residul_r_u(dPde)
    #print"\nresidual r(u) = \n{}".format(residual)
    #dis_err_set = verifier.get_error_dreach_set(dPde)
    #print"\nerror discrete reach set = \n{}".format(dis_err_set)
    #min_err_vec, _, max_err_vec, _ = dis_err_set[toTimeStep - 1].get_min_max()

    ############################################################
    # test ReachSetAssembler
    RSA = ReachSetAssembler()
    u_set, e_set, bloated_set = RSA.get_dreachset(dPde, toTimeStep)
    print "\nu_set = {}".format(u_set)
    print "\ne_set = {}".format(e_set)
    print "\nbloated_set = {}".format(bloated_set)

    ############################################################
    # test interpolation class
    intpl_set_inspace, intpl_set = verifier.get_interpolation_set(dPde, toTimeStep)
    print "\nlen of intpl_set_inspace = {}".format(len(intpl_set_inspace))
    print "\nlen of intpl_set = {}".format(len(intpl_set))

    intpl_inspace_boxes_list = verifier.get_intpl_inspace_boxes(dPde, toTimeStep)
    intpl_boxes_list = verifier.get_intpl_boxes(dPde, toTimeStep)
    #########################################################
    # test plot class

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    pl2 = Plot()
    ax2 = pl2.plot_3d_boxes(ax2, intpl_boxes_list[1], facecolor='cyan', linewidth=1, edgecolor='r')
    plt.show()

    rectangle_set_list = intpl_inspace_boxes_list[5]    # plot at 5-step
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
