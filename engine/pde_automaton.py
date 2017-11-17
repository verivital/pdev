'''
This module implements continuous/ discreted pde automaton classes and its methods
Dung Tran: Nov/2017
'''

from scipy.sparse import csc_matrix
import numpy as np
from engine.set import GeneralSet

class DPdeAutomaton(object):
    'discreted Pde automaton'

    def __init__(self):

        self.matrix_a = None
        self.vector_b = None
        self.init_vector = None
        self.perturbation = None
        self.unsafe_set = None

    def set_dynamics(self, matrix_a, vector_b, init_vector):
        'set dynamic of discreted pde automaton'

        assert isinstance(matrix_a, csc_matrix)
        assert len(matrix_a.shape) == 2
        assert isinstance(vector_b, csc_matrix)
        assert matrix_a.shape[0] == vector_b.shape[0], 'inconsistency between shapes of matrix_a and \
        vector_b'
        assert vector_b.shape[1] == 1, 'vector_b shape = {} # 1'.format(vector_b.shape[1])

        assert isinstance(init_vector, csc_matrix)
        assert matrix_a.shape[0] == init_vector.shape[0], 'matrix_a and init_vector are inconsistent'
        assert init_vector.shape[1] == 1

        self.matrix_a = matrix_a
        self.vector_b = vector_b
        self.init_vector = init_vector

    def set_perturbation(self, alpha_beta_range):
        'set pertupation on input function f and initial function u0(x)'

        # we consider pertubation on input function f, and initial condition function u_0(x)
        # actual initial condition = u_0(x) + epsilon2 * u_0(x) = alpha * u_0(x), a1 <= alpha <= b2
        # actual input function = f + epsilon1 * f = beta * f, a2 <= beta <= b2

        # the actual initial vector: = alpha * u0
        # the actual load vector: beta * b

        # pertupation defined by a general set C * [alpha beta]^T <= d

        assert isinstance(alpha_beta_range, np.ndarray)
        assert alpha_beta_range.shape == (2, 2), 'alpha_beta_range shape is incorrect'
        assert alpha_beta_range[0, 0] <= alpha_beta_range[1, 0], 'incorrect alpha range'
        assert alpha_beta_range[0, 1] <= alpha_beta_range[1, 1], 'incorrect beta range'

        alpha_beta_matrix = lil_matrix((4, 2), dtype=float)
        alpha_beta_matrix[0, 0] = -1
        alpha_beta_matrix[1, 0] = 1
        alpha_beta_matrix[2, 1] = -1
        alpha_beta_matrix[3, 1] = 1

        alpha_beta_vector = lil_matrix((4, 1), dtype=float)
        alpha_beta_vector[0, 0] = alpha_beta_range[0, 0]
        alpha_beta_vector[1, 0] = alpha_beta_range[0, 1]
        alpha_beta_vector[2, 0] = alpha_beta_range[1, 0]
        alpha_beta_vector[3, 0] = alpha_beta_range[1, 1]

        per_set = GeneralSet()
        per_set.set_constraints(alpha_beta_matrix.tocsc(), alpha_beta_vector.tocsc())

        self.perturbation = per_set

        return self.perturbation

    def set_unsafe_set(self, direction_matrix, unsafe_vector):
        'define the unsafe set of the automaton'

        # unsafe Set defined by direction_matrix * U <= unsafe_vector
        assert isinstance(direction_matrix, csc_matrix)
        assert isinstance(unsafe_vector, csc_matrix)
        assert direction_matrix.shape[0] != unsafe_vector.shape[0], 'inconsistency, \
             direction_matrix.shape[0] = {} != unsafe_vector.shape[0] = {}'\
             .format(direction_matrix.shape[0], unsafe_vector.shape[0])

        self.unsafe_set = GeneralSet()
        self.unsafe_set.set_constraints(direction_matrix, unsafe_vector)

        return self.unsafe_set
