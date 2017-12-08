'''
This module implements general set class and DReachSet class and their methods
Dung Tran: Nov/2017
'''

from scipy.sparse import csc_matrix, vstack
from scipy.optimize import linprog, minimize
from engine.functions import Functions
import numpy as np


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

    def check_feasible(self, alpha_range, beta_range):
        'check feasible of the set'

        # todo: implements check feasible using glpk package for large sparse constraints
        # another option can be using commercial Gurobi solver

        assert self.matrix_c is not None and self.vector_d is not None, 'empty set to check'
        min_vector = [1, 1]
        assert isinstance(alpha_range, tuple) and len(alpha_range) == 2 and alpha_range[0] <= alpha_range[1]
        assert isinstance(beta_range, tuple) and len(beta_range) == 2 and beta_range[0] <= beta_range[1]
        res = linprog(
            min_vector,
            A_ub=self.matrix_c.todense(),
            b_ub=self.vector_d.todense(),
            bounds=(alpha_range, beta_range),
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

        assert xmin <= xmax, 'invalid set, xmin = {} is not <= than xmax = {}'.format(xmin, xmax)
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

        self.alpha_range = None
        self.beta_range = None
        self.Vn = None
        self.ln = None

    def set_reach_set(self, alpha_range, beta_range, vector_Vn, vector_ln):
        'set a specific set'

        assert isinstance(alpha_range, tuple) and len(alpha_range) == 2 and alpha_range[0] <= alpha_range[1], 'invalid alpha_range'
        assert isinstance(beta_range, tuple) and len(beta_range) == 2 and beta_range[0] <= beta_range[1], 'invalid beta range'

        assert isinstance(vector_Vn, csc_matrix)
        assert isinstance(vector_ln, csc_matrix)
        assert vector_Vn.shape[0] == vector_ln.shape[0], 'inconsistent between Vn and ln vectors'
        assert vector_Vn.shape[1] == vector_ln.shape[1] == 1, 'wrong dimensions for vector Vn and ln'

        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.Vn = vector_Vn
        self.ln = vector_ln

    def get_lines_set(self):
        'compute range of discrete reach set, i.e.,  x_min[i] <= x[i] <= x_max[i]'

        assert self.alpha_range is not None and self.beta_range is not None, 'set perturbation parameters'
        assert self.Vn is not None and self.ln is not None, 'empty set to get min max'

        n = self.Vn.shape[0]
        min_vec = np.zeros((n,), dtype=float)
        max_vec = np.zeros((n,), dtype=float)
        min_points = []
        max_points = []
        line_set_list = []

        for i in xrange(0, n):
            min_func = Functions.U_n_i_func(self.Vn[i, 0], self.ln[i, 0])
            max_func = Functions.U_n_i_func(-self.Vn[i, 0], -self.ln[i, 0])

            x0 = [self.alpha_range[0], self.beta_range[0]]
            bnds = (self.alpha_range, self.beta_range)
            min_res = minimize(
                min_func,
                x0,
                method='L-BFGS-B',
                bounds=bnds,
                tol=1e-10, options={'disp': False})    # add options={'disp': True} to display optimization result
            max_res = minimize(
                max_func,
                x0,
                method='L-BFGS-B',
                bounds=bnds,
                tol=1e-10, options={'disp': False})    # add  options={'disp': True} to display optimization result

            if min_res.status == 0:
                min_vec[i] = min_res.fun
                min_points.append(min_res.x)
            else:
                print "\nmin_res.status = {}".format(min_res.status)
                print "\nminimization message: {}".format(min_res.message)
                raise ValueError(
                    'minimization fail!')

            if max_res.status == 0:
                max_vec[i] = -max_res.fun
                max_points.append(max_res.x)
            else:
                print "\nmax_res.status = {}".format(max_res.status)
                print "\nmaximization message: {}".format(max_res.message)
                raise ValueError(
                    'maximization fail!')

            line = LineSet()
            line.set_bounds(min_vec[i], max_vec[i])
            line_set_list.append(line)

        return line_set_list, min_vec, min_points, max_vec, max_points
