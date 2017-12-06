'''
This module implements continuous/ discreted pde automaton classes and its methods
Dung Tran: Nov/2017
'''

from scipy.sparse import csc_matrix
import numpy as np
from engine.set import GeneralSet


class DPdeAutomaton(object):
    'discreted Pde automaton: U[n] = A * U[n-1] + b[n]'

    def __init__(self):

        self.matrix_a = None
        self.vector_b = []    # a list of vector bn corresponding to different time steps
        self.inv_b_matrix = None    # use to compute vector b at each time step
        self.f_xdom = None    # range of space that input function is affected.

        self.time_step = None
        self.xlist = None
        self.init_vector = None
        self.alpha_range = None
        self.beta_range = None
        self.unsafe_set = None

    def set_matrix_a(self, matrix_a):
        'set matrix _a for DPde automaton'
        assert isinstance(matrix_a, csc_matrix)
        assert len(matrix_a.shape) == 2
        self.matrix_a = matrix_a

    def set_vector_b(self, vector_b):
        'add new vector b into vector list, in general, different steps have different vector b'

        assert isinstance(vector_b, csc_matrix)
        if self.matrix_a is not None:
            assert self.matrix_a.shape[0] == vector_b.shape[0], 'inconsistency between shapes of matrix_a and vector_b'
        self.vector_b.append(vector_b)

    def set_inv_b_matrix(self, inv_b_matrix):
        'store inv_b_matrix to compute vector b iteratively'

        assert isinstance(inv_b_matrix, csc_matrix)
        if self.matrix_a is not None:
            assert inv_b_matrix.shape == self.matrix_a.shape, 'inconsistent inv_b_matrix'

        self.inv_b_matrix = inv_b_matrix

    def set_init_condition(self, init_vector):
        'set initial condition'

        assert isinstance(init_vector, csc_matrix)
        if self.matrix_a is not None:
            assert self.matrix_a.shape[0] == init_vector.shape[0], 'inconsistent initial condition'
        self.init_vector = init_vector

    def set_xlist_time_step(self, xlist, time_step):
        'set list of meshpoints and time step'
        assert isinstance(xlist, list)
        assert len(xlist) >= 3, 'xlist should have at least three poibnts'
        for i in xrange(0, len(xlist) - 1):
            assert 0 <= xlist[i] < xlist[i + 1], 'invalid xlist'

        if self.matrix_a is not None:
            assert len(xlist) - 2 == self.matrix_a.shape[0], 'inconsistent xlist'

        assert (time_step > 0), 'time step k = {} should be >= 0'.format(time_step)
        self.xlist = xlist
        self.time_step = time_step

    def set_fxdom(self, xdom):
        'set range of space that input function f(x,t) is affected'

        assert isinstance(xdom, list) and len(xdom) == 2, 'invalid domain of f(x,t)'
        if self.xlist is not None:
            assert xdom[0] >= self.xlist[0] and xdom[1] <= self.xlist[len(self.xlist) - 1]

        self.f_xdom = xdom

    def set_perturbation(self, alpha_range, beta_range):
        'set pertupation on input function f and initial function u0(x)'

        # we consider pertubation on input function f, and initial condition function u_0(x)
        # actual initial condition = u_0(x) + epsilon2 * u_0(x) = alpha * u_0(x), a1 <= alpha <= b2
        # actual input function = f + epsilon1 * f = beta * f, a2 <= beta <= b2

        # the actual initial vector: = alpha * u0
        # the actual load vector: beta * b

        # pertupation defined by a general set C * [alpha beta]^T <= d
        assert isinstance(alpha_range, tuple) and len(alpha_range) == 2 and alpha_range[0] <= alpha_range[1], 'invalid alpha_range'
        assert isinstance(beta_range, tuple) and len(beta_range) == 2 and beta_range[0] <= beta_range[1], 'invalid beta_range'
        self.alpha_range = alpha_range
        self.beta_range = beta_range

    def set_unsafe_set(self, direction_matrix, unsafe_vector):
        'define the unsafe set of the automaton'

        # unsafe Set defined by direction_matrix * U <= unsafe_vector
        assert isinstance(direction_matrix, csc_matrix)
        assert isinstance(unsafe_vector, csc_matrix)
        assert direction_matrix.shape[0] == unsafe_vector.shape[0], 'inconsistency, \
             direction_matrix.shape[0] = {} != unsafe_vector.shape[0] = {}'\
             .format(direction_matrix.shape[0], unsafe_vector.shape[0])

        self.unsafe_set = GeneralSet()
        self.unsafe_set.set_constraints(direction_matrix, unsafe_vector)

        return self.unsafe_set

    def get_trace(self, alpha_value, beta_value, num_steps):
        'produce a trace of the discreted ODE model corresponding to specific values of alpha and beta'

        assert self.matrix_a is not None, 'empty dPde'
        assert isinstance(alpha_value, float)
        assert isinstance(beta_value, float)

        u_list = []
        times = np.linspace(0, self.time_step * num_steps, num_steps + 1)

        n = len(times)

        for i in xrange(0, n):

            if i == 0:
                current_V = self.init_vector
                current_l = csc_matrix((self.init_vector.shape[0], 1), dtype=float)
            else:
                current_V = self.matrix_a * current_V
                current_l = self.vector_b[i] + self.matrix_a * current_l

            u_list.append(current_V.multiply(alpha_value) + current_l.multiply(beta_value))

            print "\n t = {} -> \n U =: \n{}".format(i * self.time_step, u_list[i].todense())

        return u_list
