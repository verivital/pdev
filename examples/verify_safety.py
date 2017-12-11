'''
This is for verifying safety of heat equation
Dung Tran: Dec/2017
'''


from engine.fem import Fem1D
from engine.verifier import Verifier
from engine.specification import SafetySpecification
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
    xlist = mesh_points[1: mesh_points.shape[0] - 1]
    x_dom = [2.0, 4.0]    # domain of input function

    ############################################################
    # set safety specification
    SP = SafetySpecification()
    u1 = 0.0
    u2 = 0.2
    x_range = [7.2, 8.3]
    t_range = [0.05, 0.3]

    print "\nset safety specification"
    SP.set_constraints(u1, u2, x_range, t_range)
    print "\nobtaining discrete Pde automaton"
    mass_mat, stiff_mat, load_vec, init_vector, dPde = Fem1D().get_dPde_automaton(mesh_points.tolist(), x_dom, step)

    alpha_range = (0.8, 1.1)
    beta_range = (0.9, 1.1)
    dPde.set_perturbation(alpha_range, beta_range)

    print "\nverifying safety"
    status, unsafe_trace, unsafe_point, unsafe_time_point = Verifier().check_safety(dPde, SP)

    print "\nstatus = {}".format(status)
    print "\nunsafe_trace = {}".format(unsafe_trace)
    print "\nunsafe_point = {}".format(unsafe_point)
    print "\nviolate safety specification at time t = {}".format(unsafe_time_point)

    # plot unsafe trace
