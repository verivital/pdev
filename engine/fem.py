'''
This module implements Finite Element Method
Dung Tran Nov/2017

Main references:
    1) An introduction to the Finite Element Method for Differential Equations, M.Asadzaded, 2010
    2) The Finite Element Method: Theory, Implementation and Applications, Mats G. Larson, Fredirik Bengzon
'''

from scipy.sparse import lil_matrix, csc_matrix, linalg
import numpy as np
from engine.functions import Functions


class Fem1D(object):
    'contains functions of finite element method for 1D PDEs'

    @staticmethod
    def mass_assembler(x):
        'compute mass matrix for 1D problem'

        # x is list of discretized mesh points for example x = [0 , 0.1, 0.2,
        # .., 0.9, 1]
        assert isinstance(x, list)
        assert len(x) > 3, 'len(x) should >= 3'

        for i in xrange(0, len(x) - 2):
            assert isinstance(
                x[i], float), 'x[{}] should be float type'.format(i)
            assert isinstance(
                x[i + 1], float), 'x[{}] should be float type'.format(i + 1)
            assert x[i +
                     1] > x[i], 'x[i + 1] = {} should be > x[i] = {}'.format(x[i +
                                                                               1], x[i])

        n = len(x)    # number of discretized variables
        mass_matrix = lil_matrix((n, n), dtype=float)

        # filling mass_matrix

        for i in xrange(0, n):
            if i == 0:
                hi = 0
            else:
                hi = x[i] - x[i - 1]
            if i + 1 > n - 1:
                hi_plus_1 = 0
            else:
                hi_plus_1 = x[i + 1] - x[i]

            mass_matrix[i, i] = hi / 3 + hi_plus_1 / 3
            if i + 1 <= n - 1:
                mass_matrix[i, i + 1] = hi_plus_1 / 6
            if i - 1 >= 0:
                mass_matrix[i, i - 1] = hi / 6

        return mass_matrix.tocsc()

    @staticmethod
    def stiff_assembler(x):
        'compute stiff matrix for 1D problem'

        # x is list of discretized mesh points for example x = [0 , 0.1, 0.2,
        # .., 0.9, 1]
        assert isinstance(x, list)
        assert len(x) > 3, 'len(x) should >= 3'

        for i in xrange(0, len(x) - 2):
            assert isinstance(
                x[i], float), 'x[{}] should be float type'.format(i)
            assert isinstance(
                x[i + 1], float), 'x[{}] should be float type'.format(i + 1)
            assert x[i +
                     1] > x[i], 'x[i + 1] = {} should be > x[i] = {}'.format(x[i +
                                                                               1], x[i])

        n = len(x)    # number of discretized variables
        stiff_matrix = lil_matrix((n, n), dtype=float)

        # filling stiff_matrix

        for i in xrange(0, n):

            if i > 0:
                hi = x[i] - x[i - 1]
            else:
                hi = 0
            if i + 1 <= n - 1:
                hi_plus_1 = x[i + 1] - x[i]
            else:
                hi_plus_1 = 0

            if i == 0:
                stiff_matrix[i, i] = 1 / hi_plus_1
            elif i == n - 1:
                stiff_matrix[i, i] = 1 / hi
            elif 0 < i < n - 1:
                stiff_matrix[i, i] = 1 / hi + 1 / hi_plus_1

            if i + 1 <= n - 1:
                stiff_matrix[i, i + 1] = -1 / hi_plus_1
            if i - 1 >= 0:
                stiff_matrix[i, i - 1] = -1 / hi

        return stiff_matrix.tocsc()

    @staticmethod
    def load_assembler(x, x_dom, time_step):
        'compute load vector for 1D problem'

        # x is list of discretized mesh points for example x = [0 , 0.1, 0.2, .., 0.9, 1]
        # y is a variable that used to construct the f * phi function
        # the input function is defined in engine.functions.Functions class
        # x_dom = [x1, x2] defines the domain where the input function effect,
        # t_dom = (0 <= t<= time_step))
        # return [b_i] = integral (f * phi_i dx dt), ((x1 <= x <= x2), (0 <=
        # t<= time_step))

        assert isinstance(x, list)
        assert isinstance(x_dom, list)

        assert len(x) > 3, 'len(x) should >= 3'
        assert len(x_dom) == 2, 'len(f_domain) should be 2'
        assert (x[0] <= x_dom[0]) and (x_dom[0] <= x_dom[1]) and (
            x_dom[1] <= x[len(x) - 1]), 'inconsistent domain'

        for i in xrange(0, len(x) - 2):
            assert isinstance(
                x[i], float), 'x[{}] should be float type'.format(i)
            assert isinstance(
                x[i + 1], float), 'x[{}] should be float type'.format(i + 1)
            assert x[i +
                     1] > x[i], 'x[i + 1] = {} should be > x[i] = {}'.format(x[i + 1], x[i])

        assert time_step > 0, 'invalid time_step'

        n = len(x)    # number of discretized variables

        b = lil_matrix((n, 1), dtype=float)

        for i in xrange(0, n):
            if i == 0:
                seg_x = [x[0], x[0], x[1]]
            elif i == n - 1:
                seg_x = [x[i - 1], x[i], x[i]]
            elif 0 < i < n - 1:
                seg_x = [x[i - 1], x[i], x[i + 1]]

            b[i, 0] = Functions.integrate_input_func_mul_phi(seg_x, x_dom, [0.0, time_step])

        return b.tocsc()

    @staticmethod
    def get_ode(mass_mat, stiff_mat, load_vec, time_step):
        'obtain discreted ODE model'

        # the discreted ODE model has the form of: U_n = A * U_(n-1) + b

        assert isinstance(mass_mat, csc_matrix)
        assert isinstance(stiff_mat, csc_matrix)
        assert isinstance(load_vec, csc_matrix)
        assert isinstance(time_step, float)
        assert (time_step > 0), 'time step k = {} should be >= 0'.format(time_step)

        matrix_a = linalg.inv((mass_mat + stiff_mat.multiply(time_step / 2))) * \
            (mass_mat - stiff_mat.multiply(time_step / 2))
        vector_b = linalg.inv(mass_mat + stiff_mat.multiply(time_step / 2)) * \
            load_vec

        return matrix_a, vector_b

    @staticmethod
    def u0x_func(x, c):
        'return u(x, 0) = u_0(x) initial function at t = 0'

        # assumpe u_0(x) is a polynomial function defined by u_0(x) = c0 + c1 *
        # x1 + ... + cn * x^n

        assert isinstance(c, list)
        assert len(c) >= 1, 'len(c) = {} should be >= 1'.format(len(c))

        def func(x):
            'function'

            return sum(a * x**i for i, a in enumerate(c))

        return func

    @staticmethod
    def get_init_cond(x, u0x_func):
        'get initial condition from initial condition function'

        # x is list of discreted mesh points, for example x = [0 , 0.1, 0.2, .., 0.9, 1]
        # u0x_func is defined above
        assert isinstance(x, list)
        assert len(x) >= 3, 'len(x) = {} should be >= 3'.format(len(x))

        n = len(x)
        u0 = lil_matrix((n, 1), dtype=float)

        for i in xrange(0, n):
            v = x[i]
            u0[i, 0] = u0x_func(v)

        return u0.tocsc()

    @staticmethod
    def get_trace(matrix_a, vector_b, vector_u0, step, num_steps):
        'produce a trace of the discreted ODE model'

        u_list = []
        times = np.linspace(0, step * num_steps, num_steps + 1)
        print "\n times = {}".format(times)

        n = len(times)

        for i in xrange(0, n):
            print "\ni={}".format(i)
            if i == 0:
                u_list.append(vector_u0)
            else:
                u_n_minus_1 = u_list[i - 1]
                u_n = matrix_a * u_n_minus_1 + vector_b
                u_list.append(u_n)

            print "\n t = {} -> \n U =: \n{}".format(i * step, u_list[i].todense())

        return u_list
