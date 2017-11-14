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


def f_mul_phi_func(y, fc, pc):
    'return f*phi function for computing load vector'

    # assume f is polynomial function f = c0 + c1*y + c2*y^2 + ... + cm * y^m
    # phi is hat function defined on pc = [pc[0], pc[1], pc[2]]

    assert isinstance(fc, list)
    assert isinstance(pc, list)
    assert len(pc) == 3, 'len(pc) = {} != 3'.format(len(pc))
    for i in xrange(0, len(pc) - 1):
        assert pc[i] >= 0 and pc[i] < pc[i + 1], 'pc[{}] = {} should be larger than 0 and \
            smaller than pc[{}] = {}'.format(i, pc[i], i + 1, pc[i + 1])

    def N(y):
        if y < pc[0]:
            return 0.0
        elif y >= pc[0] and y < pc[1]:
            return (1 / (pc[1] - pc[0])) * (sum((a * y**(i + 1) for i, a in enumerate(fc))) - pc[0] * sum((a * y**i for i, a in enumerate(fc))))

        elif y >= pc[1] and y < pc[2]:
            return (1 / (pc[2] - pc[1])) * (-sum((a * y**(i + 1) for i, a in enumerate(fc))) + pc[2] * sum((a * y**i for i, a in enumerate(fc))))

        elif y >= pc[2]:
            return 0.0

    return np.vectorize(N)


def LoadAssembler1D(x, fc, f_dom):
    'compute load vector for 1D problem'

    # x is list of discretized mesh points for example x = [0 , 0.1, 0.2, .., 0.9, 1]
    # y is a variable that used to construct the f * phi function
    # f is input function, here we consider polynomial function in space
    # i.e. f = c0 + c1 * x + c2 * x^2 + .. + cm * x^m
    # input f = [c0, c1, c2, ... cm]
    # f_dom defines the segment where the input function effect,
    # f_domain = [x1, x2] ,  0 <= x1 <= x2 <= x_max
    # return [b_i] = integral (f * phi_i dx)

    assert isinstance(x, list)
    assert isinstance(fc, list)
    assert isinstance(f_dom, list)

    assert len(x) > 3, 'len(x) should >= 3'
    assert len(fc) > 1, 'len(f) should >= 1'
    assert len(f_dom) == 2, 'len(f_domain) should be 2'
    assert (x[0] <= f_dom[0]) and (f_dom[0] <= f_dom[1]) and (f_dom[1] <= x[len(x) - 1]), 'inconsistent domain'

    for i in xrange(0, len(x) - 2):
        assert isinstance(x[i], float), 'x[{}] should be float type'.format(i)
        assert isinstance(x[i + 1], float), 'x[{}] should be float type'.format(i + 1)
        assert x[i + 1] > x[i], 'x[i + 1] = {} should be > x[i] = {}'.format(x[i + 1], x[i])

    n = len(x) - 2    # number of discretized variables

    b = lil_matrix((n, 1), dtype=float)

    y = []

    for i in xrange(0, n):
        pc = [x[i], x[i + 1], x[i + 2]]
        fphi = f_mul_phi_func(y, fc, pc)
        I = quad(fphi, f_dom[0], f_dom[1])
        b[i, 0] = I[0]

    return b.tocsr()


if __name__ == '__main__':

    x = [0.0, 0.5, 1.0, 1.5, 2.0]
    mass_matrix = MassAssembler1D(x)
    stiff_matrix = StiffAssembler1D(x)
    fc = [1.0, 0.0, 2.0]
    fdom = [0.5, 1.0]
    load_vector = LoadAssembler1D(x, fc, fdom)

    print "\nmass matrix = \n{}".format(mass_matrix.todense())
    print "\nstiff matrix = \n{}".format(stiff_matrix.todense())
    print "\nload vector = \n{}".format(load_vector.todense())
