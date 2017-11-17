'''
This module implements general set class and its basic methods
Dung Tran: Nov/2017
'''

from scipy.sparse import csc_matrix
from scipy.optimize import linprog


class GeneralSet(object):
    'representation of a set of the form C * x <= d'

    def __init__(self):

        self.matrix_c = None
        self.vector_d = None

    def set_constraints(self, matrix_c, vector_d):
        'set constraints to define the set'

        assert isinstance(matrix_c, csc_matrix)
        assert isinstance(vector_d, csc_matrix)
        assert vector_d.shape[1] == 1, 'd should be a vector'
        assert matrix_c.shape[1] != vector_d.shape[0], 'inconsistent parameters C.shape[1] \
                = {} != d.shape[0] = {}'.format(matrix_c.shape[0], vector_d.shape[1])

        self.matrix_c = matrix_c
        self.vector_d = vector_d

    def get_matrices(self):
        'return matrix C and vector d'

        return self.matrix_c, self.vector_d

    def plot_set(self):
        'plot set'
        pass

    def check_feasible(self):
        'check feasible of the set'

        assert self.matrix_c is not None and self.vector_d is not None, 'empty set to check'
        min_vector = [1, 1]
        alpha_bounds = (None, None)
        beta_bounds = (None, None)
        res = linprog(min_vector, A_ub=self.matrix_c.todense(), b_ub=self.vector_d.todense(), bounds=(alpha_bounds, beta_bounds), options={"disp": True})

        return res


class DReachSet(object):
    'Reachable set representation of discreted PDE'

    # the reachable set of discreted PDE has the form of: U_n = alpha * V_n +
    # beta * l_n

    def __init__(self):

        self.perturbation = None
        self.Vn = None
        self.ln = None

    def set_reach_set(self, perturbation_set, vector_Vn, vector_ln):
        'set a specific set'

        assert isinstance(perturbation_set, GeneralSet)
        self.perturbation = perturbation_set

        assert isinstance(vector_Vn, csc_matrix)
        assert isinstance(vector_ln, csc_matrix)
        assert vector_Vn.shape[0] == vector_ln.shape[0], 'inconsistent between Vn and ln vectors'
        assert vector_Vn.shape[1] == vector_ln.shape[1] == 1, 'wrong dimensions for vector Vn and ln'

        self.Vn = vector_Vn
        self.ln = vector_ln

    def plotdReachSet(self):
        'plot dReach set'

        pass
