'''
This module implements Finite Element Method
Dung Tran Nov/2017

Main references:
    1) An introduction to the Finite Element Method for Differential Equations, M.Asadzaded, 2010
    2) The Finite Element Method: Theory, Implementation and Applications, Mats G. Larson, Fredirik Bengzon
'''

from scipy.sparse import lil_matrix
from scipy.integrate import quad
import numpy as np


def MassAssembler1D(x):
    'compute mass matrix for 1D problem'

    # x is list of discretized mesh points for example x = [0 , 0.1, 0.2, .., 0.9, 1]
    assert isinstance(x, list)
    assert len(x) > 3, 'len(x) should >= 3'

    for i in xrange(0, len(x) - 2):
        assert isinstance(x[i], float), 'x[{}] should be float type'.format(i)
        assert isinstance(x[i + 1], float), 'x[{}] should be float type'.format(i + 1)
        assert x[i + 1] > x[i], 'x[i + 1] = {} should be > x[i] = {}'.format(x[i + 1], x[i])

    n = len(x) - 2    # number of discretized variables
    mass_matrix = lil_matrix((n, n), dtype=float)

    # filling mass_matrix

    for i in xrange(0, n):
        hi = x[i + 1] - x[i]
        hi_plus_1 = x[i + 2] - x[i + 1]

        mass_matrix[i, i] = hi / 3 + hi_plus_1 / 3
        if i + 1 <= n - 1:
            mass_matrix[i, i + 1] = hi_plus_1 / 6
        if i - 1 >= 0:
            mass_matrix[i, i - 1] = hi / 6

    return mass_matrix.tocsr()


def StiffAssembler1D(x):
    'compute stiff matrix for 1D problem'

    # x is list of discretized mesh points for example x = [0 , 0.1, 0.2, .., 0.9, 1]
    assert isinstance(x, list)
    assert len(x) > 3, 'len(x) should >= 3'

    for i in xrange(0, len(x) - 2):
        assert isinstance(x[i], float), 'x[{}] should be float type'.format(i)
        assert isinstance(x[i + 1], float), 'x[{}] should be float type'.format(i + 1)
        assert x[i + 1] > x[i], 'x[i + 1] = {} should be > x[i] = {}'.format(x[i + 1], x[i])

    n = len(x) - 2    # number of discretized variables
    stiff_matrix = lil_matrix((n, n), dtype=float)

    # filling stiff_matrix

    for i in xrange(0, n):
        hi = x[i + 1] - x[i]
        hi_plus_1 = x[i + 2] - x[i + 1]

        stiff_matrix[i, i] = 1 / hi + 1 / hi_plus_1
        if i + 1 <= n - 1:
            stiff_matrix[i, i + 1] = -1 / hi_plus_1
        if i - 1 >= 0:
            stiff_matrix[i, i - 1] = -1 / hi

    return stiff_matrix.tocsr()


def input_func_poly(x, c, f_dom):
    'define input function f'

    assert isinstance(c, list)
    assert isinstance(f_dom, list)
    assert len(c) >= 1, 'len(c) should be >= 1'
    assert len(f_dom) == 2, 'len(f_dom) = {} != 2'.format(f_dom)
    assert f_dom[0] < f_dom[1], 'f_dom[0] = {} >= f_dom[1]'.format(f_dom[0], f_dom[1])

    def N(x):
        if x < f_dom[0]:
            return 0.0
        elif x >= f_dom[0] and x <= f_dom[1]:
            return np.poly1d(c)
        elif x > f_dom[1]:
            return 0.0

    return np.vectorize(N)


def hat_func(x, c):
    'define hat function phi(x)'

    assert isinstance(c, list)
    assert len(c) == 3, 'len(c) should be == 3'
    for i in xrange(0, len(c) - 1):
        assert c[i] < c[i + 1], 'c[{}] = {} should be smaller than c[{}] = {}'.format(i, c[i], i + 1, c[i + 1])

    def N(x):
            if x < c[0]:
                return 0.0
            elif x >= c[0] and x < c[1]:
                para = [1 / (c[1] - c[0]), -c[0] / (c[1] - c[0])]
                return np.poly1d(para)
            elif x >= c[1] and x < c[2]:
                para = [-1 / (c[2] - c[1]), c[2] / (c[2] - c[1])]
                return np.poly1d(para)
            elif x >= c[2]:
                return 0.0

    return np.vectorize(N)


def LoadAssembler1D(x, f, f_dom):
    'compute load vector for 1D problem'

    # x is list of discretized mesh points for example x = [0 , 0.1, 0.2, .., 0.9, 1]
    # f is input function, here we consider polynomial function in space
    # i.e. f = c0 + c1 * x + c2 * x^2 + .. + cm * x^m
    # input f = [c0, c1, c2, ... cm]
    # f_dom defines the segment where the input function effect,
    # f_domain = [x1, x2] ,  0 <= x1 <= x2 <= x_max
    # return [b_i] = integral (f * phi_i dx)

    assert isinstance(x, list)
    assert isinstance(f, list)
    assert isinstance(f_dom, list)

    assert len(x) > 3, 'len(x) should >= 3'
    assert len(f) > 1, 'len(f) should >= 1'
    assert len(f_dom) == 2, 'len(f_domain) should be 2'
    assert (x[0] <= f_dom[0]) and (f_dom[0] <= f_dom[1]) and (f_dom[1] <= x[len(x) - 1]), 'inconsistent domain'

    for i in xrange(0, len(x) - 2):
        assert isinstance(x[i], float), 'x[{}] should be float type'.format(i)
        assert isinstance(x[i + 1], float), 'x[{}] should be float type'.format(i + 1)
        assert x[i + 1] > x[i], 'x[i + 1] = {} should be > x[i] = {}'.format(x[i + 1], x[i])

    n = len(x) - 2    # number of discretized variables


if __name__ == '__main__':

    x = [0.0, 0.5, 1.0, 1.5, 2.0]
    mass_matrix = MassAssembler1D(x)
    stiff_matrix = StiffAssembler1D(x)
    print "\nmass matrix = \n{}".format(mass_matrix.todense())
    print "\nstiff matrix = \n{}".format(stiff_matrix.todense())

    cf = [1, 0, 2]
    f_dom = [0.5, 1.0]
    f = input_func_poly(x, cf, f_dom)
    print"\nf = {}".format(f)
    print"\nf(0.8) = {}".format(f(0.8))

    c = [0.5, 1.0, 1.5]
    phi = hat_func(x, c)

    print "\nphi = {}, phi(0.8) = {} ".format(phi, phi(0.8))

    fphi = f * phi
    #I2 = quad(phi, 0.5, 1.5)
   # print "\nI2={}".format(I2)
