'''
This module generates all SpaceEx and Flow* models for ARCH2018 paper
Dung Tran: 4/30/2018 Update:

'''

import math
import numpy as np
from pdes.pdes import HeatOneDimension, HeatTwoDimension1, HeatTwoDimension2, HeatThreeDimension, FirstOrderWaveEqOneDimension, FirstOrderWaveEqTwoDimension
from engine.printer import Printer


def heat1d_generator():
    'generate heat 1d benchmarks'

    len_x = 200
    diffusity_const = 1.16  # cm2/sec
    thermal_cond = 0.93  # cal/cm-sec-degree
    heat_exchange_coeff = 1
    he = HeatOneDimension(
        diffusity_const,
        thermal_cond,
        heat_exchange_coeff,
        len_x)
    num_x_list = [4, 9, 19]  # number of meshpoint between 0 and len_x

    for num_x in num_x_list:
        matrix_a, matrix_b = he.get_odes(num_x)
        matrix_c = None

        init_xmin_vec = np.zeros((matrix_a.shape[0],)).tolist()
        init_xmax_vec = np.zeros((matrix_a.shape[0],)).tolist()
        input_umin = [20]    # the input function     20 <= g2(t) <= 20.5
        input_umax = [20.5]

        stoptime = 10.0
        step = 0.1
        file_name_spaceEx = 'heat1d_{}'.format(num_x)
        file_name_Flowstar = 'heat1d_{}.flow*'.format(num_x)

        Printer().print_spaceex_non_homo_odes(
            matrix_a.todense(),
            matrix_b.todense(),
            matrix_c,
            init_xmin_vec,
            init_xmax_vec,
            input_umin,
            input_umax,
            stoptime,
            step,
            file_name_spaceEx)
        Printer().print_flow_non_homo_odes(
            matrix_a.todense(),
            matrix_b.todense(),
            init_xmin_vec,
            init_xmax_vec,
            input_umin,
            input_umax,
            stoptime,
            step,
            file_name_Flowstar)

        init_xmin_vec = np.zeros((matrix_a.shape[0],)).tolist()
        init_xmin_vec[0] = 0.1
        init_xmax_vec = np.zeros((matrix_a.shape[0],)).tolist()
        init_xmax_vec[0] = 0.2
        file_name_spaceEx = 'heat1d_autonomous_{}'.format(num_x)
        file_name_Flowstar = 'heat1d_autonomous_{}.flow*'.format(num_x)
        Printer().print_spaceex_homo_odes(
            matrix_a.todense(),
            matrix_c,
            init_xmin_vec,
            init_xmax_vec,
            stoptime,
            step,
            file_name_spaceEx)
        Printer().print_flow_homo_odes(
            matrix_a.todense(),
            init_xmin_vec,
            init_xmax_vec,
            stoptime,
            step,
            file_name_Flowstar)


def heat2d_generator():
    'generate heat 2d benchmarks'

    diffusity_const = 1.16  # cm^2/sec
    heat_exchange_coeff = 1
    thermal_cond = 0.93  # cal/cm-sec-degree
    len_x = 100.0  # cm
    len_y = 100.0  # cm
    he = HeatTwoDimension1(
        diffusity_const,
        heat_exchange_coeff,
        thermal_cond,
        len_x,
        len_y)
    # get linear ode model of 2-d heat equation
    # number of discretized step points in each direction
    num_list = [4, 9, 19]
    for num in num_list:
        matrix_a, matrix_b = he.get_odes(num, num)
        init_vec = []
        for j in xrange(0, num):
            # IC: u(x, y, 0) = sin(pi * x /100)
            init_vec.append(math.sin(math.pi * (j + 1) / (num + 1)))

        IC_vector = []    # initial condition for the system
        for j in xrange(0, num):
            IC_vector = IC_vector + init_vec

        init_xmin_vec = IC_vector
        init_xmax_vec = IC_vector

        # input conditions: f1(t) = 1, g1(t) = 1, 10.0 <= g2(t) <= 10.5
        input_umin = [1.0, 1.0, 10.0]
        input_umax = [1.0, 1.0, 10.5]
        stoptime = 10.0
        step = 0.1

        file_name_spaceEx = 'heat2d_{}_x_{}'.format(num, num)
        file_name_Flowstar = 'heat2d_{}_x_{}.flow*'.format(num, num)

        Printer().print_spaceex_non_homo_odes(
            matrix_a.todense(),
            matrix_b.todense(),
            None,
            init_xmin_vec,
            init_xmax_vec,
            input_umin,
            input_umax,
            stoptime,
            step,
            file_name_spaceEx)
        Printer().print_flow_non_homo_odes(
            matrix_a.todense(),
            matrix_b.todense(),
            init_xmin_vec,
            init_xmax_vec,
            input_umin,
            input_umax,
            stoptime,
            step,
            file_name_Flowstar)


def heat3d_generator():
    'generate heat 3d benchmarks'

    pass


def wave1d_generator():
    'generate wave 1d benchmarks'

    len_x = 30
    speed_const = 1

    wave = FirstOrderWaveEqOneDimension(speed_const, len_x)
    num_x_list = [5, 10, 20]
    for num_x in num_x_list:
        matrix_a = wave.get_odes(num_x)
        # IC: u(x,0) = a if 14 <= x <= 18 else u(x,0) = 0, where 0.1 <= a <=
        # 0.2

        x_step = 30.0 / num_x
        init_xmin_vec = []
        init_xmax_vec = []
        for j in xrange(0, num_x):
            if 14 <= j * x_step <= 18:
                init_xmin_vec.append(0.1)
                init_xmax_vec.append(0.2)
            else:
                init_xmin_vec.append(0)
                init_xmax_vec.append(0)

        stoptime = 10.0
        step = 0.1
        file_name_spaceEx = 'wave1d_{}'.format(num_x)
        file_name_Flowstar = 'wave1d_{}.flow*'.format(num_x)

        Printer().print_spaceex_homo_odes(
            matrix_a.todense(),
            None,
            init_xmin_vec,
            init_xmax_vec,
            stoptime,
            step,
            file_name_spaceEx)
        Printer().print_flow_homo_odes(
            matrix_a.todense(),
            init_xmin_vec,
            init_xmax_vec,
            stoptime,
            step,
            file_name_Flowstar)


def wave2d_generator():
    'generate wave 2d benchmarks'

    # parameters
    len_x = 6
    len_y = 6
    a_speed_const = -1
    b_speed_const = -1

    wave = FirstOrderWaveEqTwoDimension(
        a_speed_const, b_speed_const, len_x, len_y)
    num_list = [5, 10, 20]
    for num in num_list:
        matrix_a = wave.get_odes(num, num)
        # IC: u(x,y,0) = a where x,y \in [0,1] and u(x, y, 0) = 0 otherwise
        # in which 0.1 <= a <= 0.2

        x_step = len_x / num
        y_step = len_y / num
        init_xmin_vec = []
        init_xmax_vec = []
        for j in xrange(0, num * num):
            x_pos = j % num
            y_pos = (j - x_pos) / num
            if ((x_pos + 1) * x_step <= 1.0) and ((y_pos + 1) * y_step <= 1.0):
                init_xmin_vec.append(0.1)
                init_xmax_vec.append(0.2)

            else:
                init_xmin_vec.append(0)
                init_xmax_vec.append(0)

        stoptime = 10.0
        step = 0.1

        file_name_spaceEx = 'wave2d_{}_x_{}'.format(num, num)
        file_name_Flowstar = 'wave2d_{}_x_{}.flow*'.format(num, num)

        Printer().print_spaceex_homo_odes(
            matrix_a.todense(),
            None,
            init_xmin_vec,
            init_xmax_vec,
            stoptime,
            step,
            file_name_spaceEx)
        Printer().print_flow_homo_odes(
            matrix_a.todense(),
            init_xmin_vec,
            init_xmax_vec,
            stoptime,
            step,
            file_name_Flowstar)


if __name__ == '__main__':

    heat1d_generator()
    heat2d_generator()
    heat3d_generator()
    wave1d_generator()
    wave2d_generator()
