'''
This module implements general set class and DReachSet class and their methods
Dung Tran: Nov/2017
'''

from scipy.sparse import lil_matrix, csc_matrix, eye, vstack, hstack
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

    def check_feasible(self):
        'check feasible of the set'

        # todo: implements check feasible using glpk package for large sparse constraints
        # another option can be using commercial Gurobi solver

        assert self.matrix_c is not None and self.vector_d is not None, 'empty set to check'
        min_vector = [1, 1]
        alpha_bounds = (None, None)
        beta_bounds = (None, None)
        res = linprog(
            min_vector,
            A_ub=self.matrix_c.todense(),
            b_ub=self.vector_d.todense(),
            bounds=(
                alpha_bounds,
                beta_bounds),
            options={
                "disp": False})

        return res


class LineSet(object):
    'Line Set'

    def __init__(self):
        self.xmin = None
        self.xmax = None

    def set_bounds(self, xmin, xmax):
        'specify a segment'

        assert isinstance(xmin, float)
        assert isinstance(xmax, float)

        assert xmin < xmax, 'invalid set, xmin = {} is not < than xmax = {}'.format(xmin, xmax)
        self.xmin = xmin
        self.xmax = xmax


class RectangleSet2D(object):
    'Rectangle Set'

    def __init__(self):
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None

    def set_bounds(self, xmin, xmax, ymin, ymax):
        'specify a rectangle'

        assert isinstance(xmin, float)
        assert isinstance(xmax, float)
        assert isinstance(ymin, float)
        assert isinstance(ymax, float)

        assert xmin < xmax, 'invalid set, xmin = {} is not < than xmax = {}'.format(
            xmin, xmax)
        assert ymin < ymax, 'invalid set, ymin = {} is not < than ymax = {}'.format(
            ymin, ymax)

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax


class RectangleSet3D(object):
    'Hyper Rectangle Set'

    def __init__(self):
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.zmin = None
        self.zmax = None

    def set_bounds(self, xmin, xmax, ymin, ymax, zmin, zmax):
        'specify a rectangle'

        assert isinstance(xmin, float)
        assert isinstance(xmax, float)
        assert isinstance(ymin, float)
        assert isinstance(ymax, float)
        assert isinstance(zmin, float)
        assert isinstance(zmax, float)

        assert xmin < xmax, 'invalid set, xmin = {} is not < than xmax = {}'.format(xmin, xmax)
        assert ymin < ymax, 'invalid set, ymin = {} is not < than ymax = {}'.format(ymin, ymax)
        assert zmin < zmax, 'invalid set, zmin = {} is not < than zmax = {}'.format(zmin, zmax)

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax


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

    def range_reach_set(self, direction_matrix):
        'compute range of reach set in a specific direction, i.e.,  x_min[i] <= x[i] <= x_max[i]'

        if self.Vn is None and self.ln is None:
            raise ValueError('empty set')
        elif self.perturbation is None:
            raise ValueError('specify perturbation to plot Reachable Set')

        assert isinstance(direction_matrix, csc_matrix)
        assert direction_matrix.shape[1] == self.Vn.shape[0] == self.ln.shape[0], 'inconsistency between \
            direction matrix and vector Vn and vector ln'

        inDirection_vector_Vn = direction_matrix * self.Vn
        inDirection_vector_ln = direction_matrix * self.ln

        num_var = direction_matrix.shape[0]
        indentity_mat = eye(num_var, dtype=float)
        Aeq_mat = hstack(
            [inDirection_vector_Vn, inDirection_vector_ln, -indentity_mat])
        Beq_mat = csc_matrix((num_var + 2, 1), dtype=float)

        matrix_c = self.perturbation.matrix_c
        vector_d = self.perturbation.vector_d
        zero_mat = csc_matrix((num_var, num_var), dtype=float)
        zero_vec = csc_matrix((num_var, 1), dtype=float)
        Aub_mat = hstack([matrix_c, zero_mat])
        Bub_mat = vstack([vector_d, zero_vec])

        min_range = []    # = [xmin[1], xmin[2], ...xmin[num_var]]
        max_range = []    # = [xmax[1], xmax[2], ...xmax[num_var]]

        # compute min range
        for i in xrange(0, num_var):
            vector_c = lil_matrix((1, num_var + 2), dtype=float)
            vector_c[0, i + 2] = 1
            res = linprog(
                vector_c.todense(),
                A_ub=Aub_mat.todense(),
                b_ub=Bub_mat.todense(),
                A_eq=Aeq_mat.todense(),
                b_eq=Beq_mat.todense())

            if res.status == 0:
                min_range.append(res.fun)
            else:
                print"\nInfeasible or Error when computing min range"

        # compute max range
        for i in xrange(0, num_var):
            vector_c = lil_matrix((1, num_var + 2), dtype=float)
            vector_c[0, i + 2] = -1
            res = linprog(
                vector_c.todense(),
                A_ub=Aub_mat.todense(),
                b_ub=Bub_mat.todense(),
                A_eq=Aeq_mat.todense(),
                b_eq=Beq_mat.todense())

            if res.status == 0:
                max_range.append(-res.fun)
            else:
                print"\nInfeasible or Error when computing max range"

        return min_range, max_range
