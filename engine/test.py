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
    u_dset, e_dset, bloated_dset = RSA.get_dreachset(dPde, toTimeStep)
    print "\nu_dset = {}".format(u_dset)
    print "\ne_dset = {}".format(e_dset)
    print "\nbloated_dset = {}".format(bloated_dset)
    u_min, u_min_points, u_max, u_max_points = u_dset[toTimeStep].get_min_max()
    e_min, e_min_points, e_max, e_max_points = e_dset[toTimeStep].get_min_max()
    bloated_min, bloated_min_points, bloated_max, bloated_max_points = bloated_dset[toTimeStep].get_min_max()
    print "\nu_min = \n{}, u min_points = \n{}".format(u_min, u_min_points)
    print "\nu_max = \n{}, u max_points = \n{}".format(u_max, u_max_points)

    print "\ne_min = \n{}, e min_points = \n{}".format(e_min, e_min_points)
    print "\ne_max = \n{}, e max_points = \n{}".format(e_max, e_max_points)

    print "\nbloated_min = \n{}, bloated min_points = \n{}".format(bloated_min, bloated_min_points)
    print "\nbloated_max = \n{}, bloated max_points = \n{}".format(bloated_max, bloated_max_points)

    u_inspace, e_inspace, bl_inspace, u_set, e_set, bl_set = RSA.get_interpolationset(dPde, toTimeStep)

    print "\nu_inspace = {}".format(u_inspace)
    print "\nu_set = {}".format(u_set)
    print "\ne_inspace = {}".format(e_inspace)
    print "\ne_set = {}".format(e_set)
    print "\nbl_inspace = {}".format(bl_inspace)
    print "\nbl_set = {}".format(bl_set)

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
