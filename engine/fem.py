'''
This module implements Finite Element Method
Dung Tran Nov/2017

Main references:
    1) An introduction to the Finite Element Method for Differential Equations, M.Asadzaded, 2010
    2) The Finite Element Method: Theory, Implementation and Applications, Mats G. Larson, Fredirik Bengzon
'''

from scipy.sparse import lil_matrix, csc_matrix, linalg
from scipy.integrate import quad
import numpy as np


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
    def f_mul_phi(y, fc, pc):
        'return f*phi function for computing load vector'

        # assume f is polynomial function f = c0 + c1*y + c2*y^2 + ... + cm * y^m
        # phi is hat function defined on pc = [pc[0], pc[1], pc[2]]
        # at left boundary pc[0] = pc[1] = 0
        # at right boudary pc[1] = pc[2] = ?

        # todo: add more general function f which depends on t and x. using np.dblquad to calculate
        # double integration

        assert isinstance(fc, list)
        assert isinstance(pc, list)
        assert len(pc) == 3, 'len(pc) = {} != 3'.format(len(pc))
        for i in xrange(0, len(pc) - 1):
            assert pc[i] >= 0 and pc[i] <= pc[i + 1], 'pc[{}] = {} should be >= 0 and \
                <= pc[{}] = {}'.format(i, pc[i], i + 1, pc[i + 1])

        def func(y):
            'piecewise function'

            if pc[0] == pc[1]:    # left boundary phi_0
                return (1 / (pc[2] - pc[1])) * (-sum((a * y**(i + 1) for i, a in enumerate(
                    fc))) + pc[2] * sum((a * y**i for i, a in enumerate(fc))))
            elif pc[1] == pc[2]:    # right boundary phi_m
                return (1 / (pc[1] - pc[0])) * (sum((a * y**(i + 1) for i, a in enumerate(
                    fc))) - pc[0] * sum((a * y**i for i, a in enumerate(fc))))
            elif pc[0] < pc[1] < pc[2]:
                if y < pc[0]:
                    return 0.0
                elif y >= pc[0] and y < pc[1]:
                    return (1 / (pc[1] - pc[0])) * (sum((a * y**(i + 1) for i, a in enumerate(
                        fc))) - pc[0] * sum((a * y**i for i, a in enumerate(fc))))

                elif y >= pc[1] and y < pc[2]:
                    return (1 / (pc[2] - pc[1])) * (-sum((a * y**(i + 1) for i, a in enumerate(
                        fc))) + pc[2] * sum((a * y**i for i, a in enumerate(fc))))

                elif y >= pc[2]:
                    return 0.0

        return np.vectorize(func)

    @staticmethod
    def load_assembler(x, fc, f_dom):
        'compute load vector for 1D problem'

        # x is list of discretized mesh points for example x = [0 , 0.1, 0.2, .., 0.9, 1]
        # y is a variable that used to construct the f * phi function
        # fc is the input function, here we consider polynomial function in space
        # i.e. f = c0 + c1 * x + c2 * x^2 + .. + cm * x^m
        # input fc = [c0, c1, c2, ... cm]
        # f_dom defines the segment where the input function effect,
        # f_domain = [x1, x2] ,  0 <= x1 <= x2 <= x_max
        # return [b_i] = integral (f * phi_i dx)

        assert isinstance(x, list)
        assert isinstance(fc, list)
        assert isinstance(f_dom, list)

        assert len(x) > 3, 'len(x) should >= 3'
        assert len(fc) > 1, 'len(f) should >= 1'
        assert len(f_dom) == 2, 'len(f_domain) should be 2'
        assert (x[0] <= f_dom[0]) and (f_dom[0] <= f_dom[1]) and (
            f_dom[1] <= x[len(x) - 1]), 'inconsistent domain'

        for i in xrange(0, len(x) - 2):
            assert isinstance(
                x[i], float), 'x[{}] should be float type'.format(i)
            assert isinstance(
                x[i + 1], float), 'x[{}] should be float type'.format(i + 1)
            assert x[i +
                     1] > x[i], 'x[i + 1] = {} should be > x[i] = {}'.format(x[i +
                                                                               1], x[i])

        n = len(x)    # number of discretized variables

        b = lil_matrix((n, 1), dtype=float)

        y = []

        for i in xrange(0, n):
            if i == 0:
                pc = [x[0], x[0], x[1]]
            elif i == n - 1:
                pc = [x[i - 1], x[i], x[i]]
            elif 0 < i < n - 1:
                pc = [x[i - 1], x[i], x[i + 1]]
            fphi = Fem1D.f_mul_phi(y, fc, pc)
            I = quad(fphi, f_dom[0], f_dom[1])
            b[i, 0] = I[0]

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
            load_vec.multiply(time_step)

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

    @staticmethod
    def plot_trace(trace, step):
        'plot trace of the discreted ODE model'

        assert isinstance(trace, list)
        n = len(trace)
        assert n >= 2, 'trace should have at least two points, currently it has {} points'.format(
            n)
