'''
This module implements some basic functions used for reachable set computation and interpolation
Dung Tran: Nov/2017
'''

from sympy import Piecewise, And, Function, lambdify
from sympy.abc import x, t, alpha, beta
from scipy.optimize import minimize


class Functions(object):
    'implements related functions for interplation'

    @staticmethod
    def phi(seg_x):
        'define hat function phi_i(x) used in interpolation'

        # phi_i = (x - x[j - 1]) / h[j]             if  x[j - 1] <= x < x[j]
        # phi_i = (x[j + 1] - x) / h[j + 1]         if x[j] <= x < x[j + 1]
        # phi_i = 0                                 if x < x[j - 1] or x > x[j
        # + 1]

        # at left boundary, x[j - 1] = x[j] = 0 --> phi_i = (x[j] - x) / h[j]
        # at right boundary, x[j] = x[j + 1] = len(x) --> phi_i = (x[j + 1] - x) /
        # h[j]

        assert isinstance(seg_x, list)
        assert len(seg_x) == 3
        for i in xrange(0, len(seg_x) - 1):
            assert seg_x[i] >= 0 and seg_x[i] <= seg_x[i +
                                                       1], 'invalid segment'

        if seg_x[0] == seg_x[1]:
            hj = seg_x[1]
            func = Piecewise((0, x <= 0), (0, x > seg_x[1]), ((
                seg_x[1] - x) / hj, And(0 < x, x <= seg_x[1])))    # don't care seg_x[2] in this case
        elif seg_x[1] == seg_x[2]:
            hj = seg_x[1] - seg_x[0]
            func = Piecewise((0, x <= seg_x[0]), (0, x > seg_x[1]), ((
                x - seg_x[0]) / hj, And(seg_x[0] < x, x <= seg_x[1])))

        elif seg_x[0] < seg_x[1] < seg_x[2]:
            hj = seg_x[1] - seg_x[0]
            hj_plus_1 = seg_x[2] - seg_x[1]
            func = Piecewise((0,
                              x <= seg_x[0]),
                             ((x - seg_x[0]) / hj,
                              And(seg_x[0] < x,
                                  x <= seg_x[1])),
                             ((seg_x[2] - x) / hj_plus_1,
                              And(seg_x[1] < x,
                                  x <= seg_x[2])),
                             (0, x > seg_x[2]))

        return lambdify(x, func)

    @staticmethod
    def input_func():
        'define the input function f(x,t)'

        # can be general function with x and t variables
        func = Function('func')
        func = 2 * x + 3 * t    # you can change f(x,t) function here
        return lambdify((x, t), func)

    @staticmethod
    def init_func():
        'define initial condition function u0(x)'

        # can be general function with variable x
        func = Function('func')
        func = 2 * x ** 2 + 1    # you can change u0(x) function here
        return lambdify(x, func)

    @staticmethod
    def Si_n_func(step, t_n_minus_1):
        'Si[n](t) function at time step t = tn,  used in interpolation in time'

        assert step > 0, 'invalid time_step'
        assert isinstance(step, float)
        assert isinstance(t_n_minus_1, float)
        assert t_n_minus_1 >= 0

        func = Function('func')
        func = (t - t_n_minus_1) / step

        return lambdify(t, func)

    @staticmethod
    def Si_n_minus_1_func(step, t_n):
        'Si[n - 1] (t) function at time step tn = n, used in interpolation in time'

        assert step > 0, 'invalid time_step'
        assert isinstance(step, float)
        assert isinstance(t_n, float)
        assert t_n >= 0

        func = Function('func')
        func = (t_n - t) / step

        return lambdify(t, func)

    @staticmethod
    def intpl_inspace_func(a, b, c, d):
        'Un(x) function after doing interpolation in space'

        # U_n(x) = (a*alpha + b*beta)*x + c*alpha + d*beta

        assert isinstance(a, float)
        assert isinstance(b, float)
        assert isinstance(c, float)
        assert isinstance(d, float)
        func = Function('func')
        func = (a * alpha + b * beta) * x + c * alpha + d * beta
        func_eval = lambdify((x, alpha, beta), func)

        return func_eval(*tuple(x))

    @staticmethod
    def intpl_in_time_and_space_func(
            step, delta_a, delta_b, delta_gamma_a, delta_gamma_b, delta_gamma_c, delta_gamma_d):
        'U(x,t) function after doing interpolation in both space and time'

        # U(x) = (1/step) * (delta_a * alpha + delta_b * beta) * t * x +
        #         (delta_gamma_a * alpha + delta_gamma_b * beta) * x +
        #         (delta_gamma_c * alpha + delta_gamma_d * beta)

        assert isinstance(step, float)
        assert step > 0, 'invalid time step'
        assert isinstance(delta_a, float)
        assert isinstance(delta_b, float)
        assert isinstance(delta_gamma_a, float)
        assert isinstance(delta_gamma_b, float)
        assert isinstance(delta_gamma_c, float)
        assert isinstance(delta_gamma_d, float)

        func = Function('func')
        func = (1 / step) * (delta_a * alpha + delta_b * beta) * t * x + \
            (delta_gamma_a * alpha + delta_gamma_b * beta) * x + \
            delta_gamma_c * alpha + delta_gamma_d * beta

        func_eval = lambdify((t, x, alpha, beta), func)

        return func_eval(*(tuple(x)))


if __name__ == '__main__':

    a1 = 0.1
    b1 = 0.2
    c1 = 0.3
    d1 = 0.0
    myfun = Functions.intpl_inspace_func(a1, b1, c1, d1)
    print "\nf(2, 1, 1) = {}".format(myfun([2, 1, 1]))

    x0 = [1, 1, 1]
    bnds = ((-1, 1), (1, 2), (0, 1))
    res = minimize(myfun, x0, method='TNC', bounds=bnds, tol=1e-10)
    print "\nresult_MIN = {}".format(res)

    myfun2 = Functions.intpl_inspace_func(-a1, -b1, -c1, -d1)

    res2 = minimize(myfun2, x0, method='TNC', bounds=bnds, tol=1e-10)
    print "\nresult_MAX = {}".format(res2)
