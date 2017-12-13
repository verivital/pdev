'''
This script measure the reachable set computation time with different sizes
of time step and space step
Dung Tran: Dec/2017
'''

from engine.fem import Fem1D
from engine.verifier import ReachSetAssembler
import numpy as np
import time


def store_data(x_data, y_data, x_y_name, file_name):
    'store data in .dat file to plot figure with gnuplot'

    assert isinstance(x_data, list)
    assert isinstance(y_data, list)
    assert isinstance(x_y_name, list)

    if len(x_y_name) != 2:
        raise ValueError('too many names')
    if len(x_data) != len(y_data):
        raise ValueError('data length mismatch')

    if x_y_name == []:
        name_x = 'x'
        name_y = 'y'
    else:
        name_x = x_y_name[0]
        name_y = x_y_name[1]

    data_file = open(file_name, 'w')
    data_file.write("#    {}        {}\n".format(name_x, name_y))

    for i in xrange(0, len(x_data)):
        data_file.write("    {: < 28} {: < 28}\n".format(x_data[i], y_data[i]))


def computationtime_vs_nummeshpoint():
    'measure reachability analysis computation time for time range [0, 10s] with different number of mesh points'

    ##################################################
    # generate dPde automaton
    L = 10.0    # length of rod
    num_mesh_points = [10, 20]     # number of mesh points
    step = 0.1
    x_dom = [2.0, 4.0]    # domain of input function
    alpha_range = (0.8, 1.1)
    beta_range = (0.9, 1.1)
    computation_time = []

    for j in xrange(0, len(num_mesh_points)):
        start = time.time()
        num_mesh_point = num_mesh_points[j]
        mesh_grid = np.arange(0, num_mesh_point + 1, step=1)
        mesh_points = np.multiply(mesh_grid, L / num_mesh_point)
        print "\nmesh_points = {}".format(mesh_points)
        toTimeStep = 2    # number of time steps
        time_grid = np.arange(0, toTimeStep + 1, step=1)
        time_list = np.multiply(time_grid, step)
        xlist = mesh_points[1: mesh_points.shape[0] - 1]
        dPde = Fem1D().get_dPde_automaton(mesh_points.tolist(), x_dom, step)

        dPde.set_perturbation(alpha_range, beta_range)

        ############################################################
        # compute error dicrete reachable set
        RSA = ReachSetAssembler()
        RSA.get_interpolationset(dPde, toTimeStep)

        end = time.time()
        computation_time.append(end - start)

        print "\ncomputation time for number of mesh points = {} is {}".format(num_mesh_point, computation_time[j])

    store_data(num_mesh_points, computation_time, ['number of mesh points', 'computation time'], 'computationtime_vs_nummeshpoints.dat')


def computationtime_vs_numtimesteps():
    'measure reachability analysis computation time for time range [0, 10s] with different number of time steps'

    ##################################################
    # generate dPde automaton
    L = 10.0    # length of rod
    num_mesh_points = 20    # number of mesh points
    mesh_grid = np.arange(0, num_mesh_points + 1, step=1)
    mesh_points = np.multiply(mesh_grid, L / num_mesh_points)
    print "\nmesh_points = {}".format(mesh_points)
    steps = [0.1, 0.05, 0.01]    # time step of FEM
    x_dom = [2.0, 4.0]    # domain of input function
    alpha_range = (0.8, 1.1)
    beta_range = (0.9, 1.1)
    computation_time = []
    number_time_steps = []
    print "\nmeasuring reachability analysis time for time range [0, 10s] using different number of time steps"
    for j in xrange(0, len(steps)):
        start = time.time()
        step = steps[j]
        toTimeStep = int(10.0 / step)    # number of time steps
        number_time_steps.append(toTimeStep)
        time_grid = np.arange(0, toTimeStep + 1, step=1)
        time_list = np.multiply(time_grid, step)
        print "\ntime_list = {}".format(time_list)
        dPde = Fem1D().get_dPde_automaton(mesh_points.tolist(), x_dom, step)
        dPde.set_perturbation(alpha_range, beta_range)

        ############################################################
        RSA = ReachSetAssembler()
        RSA.get_interpolationset(dPde, toTimeStep)    # compute discrete reachable set
        end = time.time()
        computation_time.append(end - start)

    store_data(number_time_steps, computation_time, ['number of time steps', 'computation time'], 'compuationtime_vs_numtimestpes.dat')

if __name__ == '__main__':

    computationtime_vs_nummeshpoint()
    computationtime_vs_numtimesteps()
