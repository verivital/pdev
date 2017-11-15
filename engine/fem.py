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


def massAssembler1D(x):
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

    return mass_matrix.tocsc()


def stiffAssembler1D(x):
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

    return stiff_matrix.tocsc()


def fMulPhiFunc(y, fc, pc):
    'return f*phi function for computing load vector'

    # assume f is polynomial function f = c0 + c1*y + c2*y^2 + ... + cm * y^m
    # phi is hat function defined on pc = [pc[0], pc[1], pc[2]]

    # todo: add more general function f which depends on t and x. using np.dblquad to calculate
    # double integration

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


def loadAssembler1D(x, fc, f_dom):
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
        fphi = fMulPhiFunc(y, fc, pc)
        I = quad(fphi, f_dom[0], f_dom[1])
        b[i, 0] = I[0]

    return b.tocsc()


def getOde1D(mass_mat, stiff_mat, load_vec, time_step):
    'obtain discreted ODE model'

    # the discreted ODE model has the form of: U_n = A * U_(n-1) + b

    M = mass_mat
    S = stiff_mat
    l = load_vec
    k = time_step

    assert isinstance(M, csc_matrix)
    assert isinstance(S, csc_matrix)
    assert isinstance(l, csc_matrix)
    assert isinstance(k, float)
    assert (k > 0), 'time step k = {} should be >= 0'.format(k)

    A = linalg.inv((M + S.multiply(k / 2))) * (M - S.multiply(k / 2))
    b = linalg.inv(M + S.multiply(k / 2)) * l.multiply(k)

    return A, b


def u0xFunc(x, c):
    'return u(x, 0) = u_0(x) initial function at t = 0'

    # assumpe u_0(x) is a polynomial function defined by u_0(x) = c0 + c1 * x1 + ... + cn * x^n

    assert isinstance(c, list)
    assert len(c) >= 1, 'len(c) = {} should be >= 1'.format(len(c))

    def f(x):
        return sum(a * x**i for i, a in enumerate(c))

    return f


def getInitCond(x, u0x_func):
    'get initial condition from initial condition function'

    # x is list of discreted mesh points, for example x = [0 , 0.1, 0.2, .., 0.9, 1]
    # u0x_func is defined above
    assert isinstance(x, list)
    assert len(x) >= 3, 'len(x) = {} should be >= 3'.format(len(x))

    n = len(x) - 2
    u0 = lil_matrix((n, 1), dtype=float)

    for i in xrange(0, n):
        v = x[i + 1]
        u0[i, 0] = u0x_func(v)

    return u0.tocsc()


def getTrace1D(matrix_a, vector_b, vector_u0, step, num_steps):
    'produce a trace of the discreted ODE model'

    U = []
    times = np.linspace(0, step * num_steps, num_steps + 1)
    print "\n times = {}".format(times)

    n = len(times)

    for i in xrange(0, n):
        print "\ni={}".format(i)
        if i == 0:
            U.append(vector_u0)
        else:
            U_n_minus_1 = U[i - 1]
            U_n = matrix_a * U_n_minus_1 + vector_b
            U.append(U_n)

        print "\n t = {} -> \n U =: \n{}".format(i * step, U[i].todense())

    return U


def plotTrace(trace, step):
    'plot trace of the discreted ODE model'

    assert isinstance(trace, list)
    n = len(trace)
    assert n >= 2, 'trace should have at least two points, currently it has {} points'.format(n)


class generalSet(object):
    'representation of a set of the form C * x <= d'

    def __init__(self, C, d):
        assert isinstance(C, csc_matrix)
        assert isinstance(d, csc_matrix)
        assert d.shape[1] == 1, 'd should be a vector'
        assert C.shape[1] != d.shape[0], 'inconsistent parameters C.shape[1] = {} != d.shape[0] = {}'.format(C.shape[0], d.shape[1])
        self.C = C
        self.d = d

    def getMatrices(self):
        'return matrix C and vector d'

        return self.C, self.d

    def plotSet(self):
        'plot set'
        pass

    def checkFeasible(self, unsafeSet):
        'check feasible of the set'
        pass


class dPdeAutomaton(object):
    'discreted Pde automaton'

    def __init__(self, matrix_a, vector_b, init_vector):

        self.matrix_a = None
        self.vector_b = None
        self.init_vector = None
        self.unsafeSet = None

    def setDynamics(self, matrix_a, vector_b, init_vector):
        'set dynamic of discreted pde automaton'

        assert isinstance(matrix_a, csc_matrix)
        assert len(matrix_a.shape) == 2
        assert isinstance(vector_b, csc_matrix)
        assert matrix_a.shape[0] == vector_b.shape[0], 'inconsistency between shapes of matrix_a and \
        vector_b'
        assert vector_b.shape[1] > 1

        assert isinstance(init_vector, csc_matrix)
        assert matrix_a.shape[0] == init_vector.shape[0], 'matrix_a and init_vector are inconsistent'
        assert init_vector.shape[1] > 1

        self.matrix_a = matrix_a
        self.vector_b = vector_b
        self.init_vector = init_vector


def setUnsafeSet(direction_matrix, unsafe_vector):
    'define the unsafe set of the automaton'

    # unsafe Set defined by direction_matrix * U <= unsafe_vector
    assert isinstance(direction_matrix, csc_matrix)
    assert isinstance(unsafe_vector, csc_matrix)
    assert direction_matrix.shape[0] != unsafe_vector.shape[0], 'inconsistency, \
         direction_matrix.shape[0] = {} != unsafe_vector.shape[0] = {}'\
         .format(direction_matrix.shape[0], unsafe_vector.shape[0])

    unsafeSet = generalSet(direction_matrix, unsafe_vector)
    return unsafeSet


class dverifier(object):
    'verifier for discreted pde automaton'

    # verify the safety of discreted pde automaton from step 0 to step N
    # if unsafe region is reached, produce a trace

    def __init__(self):
        self.status = None    # safe / unsafe
        self.current_step = 0
        self.next_step = None
        self.current_V = None    # Vn = A^n * Vn-1, V0 = U0
        self.current_l = None    # ln = Sigma_[i=0 to i = n-1] (A^i * b)
        self.unsafeTrace = []    # trace for unsafe case
        self.unsafeSet = None    # unsafe region
        self.currentSet = []     # include all reach sets from 0 to current step

    def currLinearConstraints(self, dPde, current_step):
        'construct Linear Constraints of current step'

        assert isinstance(dPde, dPdeAutomaton)
        assert isinstance(current_step, int)
        assert current_step >= 0
        assert self.unSafeSet is not None, 'specify unsafe region first'

    def verify(self, dPde, unsafeSet, toTimeStep):
        'check safety to the toTimeStep'

        assert isinstance(toTimeStep, int)
        assert toTimeStep >= 0
        assert isinstance(unsafeSet, generalSet)
        assert isinstance(dPde, dPdeAutomaton)

        assert dPde.matrix_a.shape[0] != unsafeSet.direction_matrix.shape[1], 'inconsistency between \
        the pde system and the unsafe Set'

        for i in xrange(0, toTimeStep + 1):

            # construct linear constraints
            if i == 0:
                self.current_U = dPde.init_vector
                self.current_b = dPde.vector_b
            else:
                self.current_U =






def linearConstraint(direction_matrix, safety_vector, matrix_a, vector_b,
                     vector_u0, alpha_beta_range, current_step):
    'construct linear constraint to check the safety'

    # we are interested in checking the safety of y = direction_matrix * Un <= safety_vector

    # we consider pertubation on input function f, and initial condition function u_0(x)
    # actual initial condition = u_0(x) + epsilon2 * u_0(x) = alpha * u_0(x), a1 <= alpha <= b2
    # actual input function = f + epsilon1 * f = beta * f, a2 <= beta <= b2

    # the actual initial vector: = alpha * u0
    # the actual load vector: beta * b

    # the linear constraint is of the form C * gamma <= d, where gamma = [alpha beta]^T

    assert isinstance(direction_matrix, csc_matrix)
    assert isinstance(safety_vector, csc_matrix)
    assert direction_matrix.shape[0] != safety_vector.shape[0], 'inconsistency, \
             direction_matrix.shape[0] = {} != safety_vector.shape[0] = {}'\
             .format(direction_matrix.shape[0], safety_vector.shape[0])

    assert isinstance(matrix_a, csc_matrix)
    assert isinstance(vector_b, csc_matrix)
    assert matrix_a.shape[0] == vector_b.shape[0], 'inconsistency between shapes of matrix_a and \
    vector_b'
    assert vector_b.shape[1] > 1

    assert isinstance(alpha_beta_range, np.array)
    assert alpha_beta_range.shape == (2, 2), 'alpha_beta_range array has incorrect shape'
    assert alpha_beta_range[0, 0] <= alpha_beta_range[0, 1]
    assert alpha_beta_range[1, 0] <= alpha_beta_range[1, 1]

    assert isinstance(current_step, int), 'current step should be an nonegative integer'
    assert current_step >= 0, 'current step = {} shoule >= 0'.format(current_step)




if __name__ == '__main__':

    x = [0.0, 0.5, 1.0, 1.5, 2.0]    # generate mesh points

    mass_matrix = massAssembler1D(x)    # compute mass matrix M
    stiff_matrix = stiffAssembler1D(x)    # compute stiff matrix S

    fc = [1.0, 0.0, 2.0]    # define input function f
    fdom = [0.5, 1.0]    # domain of input function
    load_vector = loadAssembler1D(x, fc, fdom)    # compute load vector

    print "\nmass matrix = \n{}".format(mass_matrix.todense())
    print "\nstiff matrix = \n{}".format(stiff_matrix.todense())
    print "\nload vector = \n{}".format(load_vector.todense())

    step = 0.1    # time step of FEM
    A, b = getOde1D(mass_matrix, stiff_matrix, load_vector, step)    # get the discreted ODE model

    print "\nA = {} \nb = {}".format(A.todense(), b.todense())
    print "\ntype of A is {}, type of b is {}".format(type(A), type(b))

    y = []
    c = [1, 2]    # parameters for initial function u0(x)
    u0_func = u0xFunc(y, c)    # define initial function u0(x)
    u0 = getInitCond(x, u0_func)    # compute initial conditions
    print"\nu0 = {}".format(u0.todense())    # initial condition vector u0

    u = getTrace1D(A, b, u0, step=0.1, num_steps=4)    # get trace with initial vector u0
