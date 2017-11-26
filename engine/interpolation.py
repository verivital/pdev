'''
This module implements interpolation class and its methods
Dung Tran: Nov/2017
'''

import math
import numpy as np
from engine.set import DReachSet
from engine.functions import Functions
from scipy.optimize import minimize


class InterpolSetInSpace(object):
    'represent the set after doing interpolation in space'

    ##########################################################################
    # U_n(x) = (a_n,i * alpha + b_n,i * beta) * x + c_n,i * alpha + d_n,i * beta
    # x[i] <= x < x[i + 1], timestep = n
    ##########################################################################

    def __init__(self):
        self.a_vec = None
        self.b_vec = None
        self.c_vec = None
        self.d_vec = None
        self.xlist = None

    def set_values(self, xlist, a_current_step_vec, b_current_step_vec,
                   c_current_step_vec, d_current_step_vec):
        'set values for parameter of interpolation set'

        assert isinstance(a_current_step_vec, np.ndarray)
        assert isinstance(b_current_step_vec, np.ndarray)
        assert isinstance(c_current_step_vec, np.ndarray)
        assert isinstance(d_current_step_vec, np.ndarray)

        assert a_current_step_vec.shape == b_current_step_vec.shape == \
            c_current_step_vec.shape == d_current_step_vec.shape, 'inconsistent data'
        assert a_current_step_vec.shape[1] == 1, 'invalid shape'

        assert isinstance(xlist, list)
        for i in xrange(0, len(xlist) - 1):
            assert xlist[i] < xlist[i + 1], 'invalid list'

        assert len(xlist) == a_current_step_vec.shape[0] + \
            1, 'inconsistency between xlist and vectors shapes'

        self.a_vec = a_current_step_vec
        self.b_vec = b_current_step_vec
        self.c_vec = c_current_step_vec
        self.d_vec = d_current_step_vec
        self.xlist = xlist

    def get_min_max(self, alpha_range, beta_range):
        'find minimum value of U_n(x)'

        assert self.a_vec is not None and self.b_vec is not None and self.c_vec is not None and self.d_vec is not None
        assert isinstance(alpha_range, list)
        assert isinstance(beta_range, list)
        assert len(alpha_range) == len(beta_range) == 2, 'invalid parameters'
        assert alpha_range[0] <= alpha_range[1]
        assert beta_range[0] <= beta_range[1]

        n = self.a_vec.shape[0]

        a_vec = self.a_vec
        b_vec = self.b_vec
        c_vec = self.c_vec
        d_vec = self.d_vec

        alpha_bounds = (alpha_range[0], alpha_range[1])
        beta_bounds = (beta_range[0], beta_range[1])

        # minimum value of Un(x) at each segment
        min_vec = np.zeros((n, 1), dtype=float)
        # minimum points [x_min, alpha_min, beta_min]
        min_points = np.zeros((n, 3), dtype=float)
        # maximum value of Un(x) at each segment
        max_vec = np.zeros((n, 1), dtype=float)
        # maximum points [x_max, alpha_max, beta_max]
        max_points = np.zeros((n, 3), dtype=float)

        for i in xrange(0, n):
            min_func = Functions.intpl_inspace_func(
                a_vec[i, 0], b_vec[i, 0], c_vec[i, 0], d_vec[i, 0])
            max_func = Functions.intpl_inspace_func(
                -a_vec[i, 0], -b_vec[i, 0], -c_vec[i, 0], -d_vec[i, 0])
            xbounds = (self.xlist[i], self.xlist[i + 1])
            x0 = [self.xlist[i], alpha_range[0], beta_range[0]]
            bnds = (xbounds, alpha_bounds, beta_bounds)

            min_res = minimize(
                min_func,
                x0,
                method='TNC',
                bounds=bnds,
                tol=1e-10)
            max_res = minimize(
                max_func,
                x0,
                method='TNC',
                bounds=bnds,
                tol=1e-10)

            if min_res.status == 0:
                min_vec[i, 0] = min_res.fun
                min_points[i, :] = min_res.x
            else:
                raise ValueError('min-optimization fail')

            if max_res.status == 0:
                max_vec[i, 0] = -max_res.fun
                max_points[i, :] = max_res.x
            else:
                raise ValueError('max-optimization fail')

        return min_vec, min_points, max_vec, max_points


class InterpolationSet(object):
    'represent the set after doing interpolation in both space and time'

    ###################################################################
    #    u(x,t) = (1 / k) *(delt_a_n,i * alpha + delt_bn,i * beta) * x * t +
    #             (delt_gamma_a_n,i * alpha + delt_gamma_b_n,i * beta) * x +
    #             (delta_gamma_c_n,i * alpha + delta_gamma_d_n,i * beta)
    #
    #    where x[i]< x <= x[i+1], t[n-1] < t <= t[n]
    ###################################################################

    def __init__(self):
        self.step = None
        self.xlist = None
        # [delta_a_i,j], i = 1, .., m (space step) and j = 1, ..., n (time step)
        self.delta_a_matrix = None
        self.delta_gamma_a_matrix = None
        # [delta_b_i,j], i = 1, .., m (space step) and j = 1, ..., n (time step)
        self.delta_b_matrix = None
        self.delta_gamma_b_matrix = None
        # [delta_gamma_c_i,j], i = 1, .., m (space step) and j = 1, ..., n (time step)
        self.delta_gamma_c_matrix = None
        # [delta_gamma_d_i,j], i = 1, .., m (space step) and j = 1, ..., n (time step)
        self.delta_gamma_d_matrix = None

    def set_values(self, step, xlist, delta_a_matrix, delta_b_matrix,
                   delta_gamma_a_matrix, delta_gamma_b_matrix,
                   delta_gamma_c_matrix, delta_gamma_d_matrix):
        'set values for the set'

        assert isinstance(step, float)
        assert step > 0, 'invalid time step'
        assert isinstance(delta_a_matrix, np.ndarray)
        assert isinstance(delta_b_matrix, np.ndarray)
        assert isinstance(delta_gamma_a_matrix, np.ndarray)
        assert isinstance(delta_gamma_b_matrix, np.ndarray)
        assert isinstance(delta_gamma_c_matrix, np.ndarray)
        assert isinstance(delta_gamma_d_matrix, np.ndarray)

        assert delta_a_matrix.shape == delta_b_matrix.shape == \
            delta_gamma_a_matrix == delta_gamma_b_matrix == delta_gamma_c_matrix.shape == \
            delta_gamma_d_matrix.shape, 'invalid data set'

        assert isinstance(xlist, list)
        for i in xrange(0, len(xlist) - 1):
            assert xlist[i] < xlist[i + 1], 'invalid list'

        assert len(xlist) == delta_a_matrix.shape[0] + \
            1, 'inconsistency between xlist and matrices shapes'

        self.step = step
        self.xlist = xlist
        self.delta_a_matrix = delta_a_matrix
        self.delta_b_matrix = delta_b_matrix
        self.delta_gamma_a_matrix = delta_gamma_a_matrix
        self.delta_gamma_b_matrix = delta_gamma_b_matrix
        self.delta_gamma_c_matrix = delta_gamma_c_matrix
        self.delta_gamma_d_matrix = delta_gamma_d_matrix

    def get_min_max(self, time_range, x_range, alpha_range, beta_range):
        'find minimum and maximum values of interpolation set U(x, t)'

        # We are interested in U(x,t) where:  x_range[0] <= x <= x_range[1],
        # and t_range[0] <= t <= t_range[1]

        assert self.delta_a_matrix is not None, 'empty set'

        assert isinstance(time_range, list)
        assert isinstance(x_range, list)
        assert len(x_range) == len(
            time_range) == 2, 'invalid time range or invalid x_range'
        assert isinstance(time_range[0], float) and isinstance(time_range[1], float)

        assert isinstance(x_range[0], float) and isinstance(x_range[1], float)
        assert x_range[0] >= self.xlist[0] and x_range[1] <= self.xlist[len(
            self.xlist) - 1], 'invalid x range'

        # map time_range
        time_start_point = int(math.floor(float(time_range[0]) / self.step))
        time_stop_point = int(math.ceil(float(time_range[1]) / self.step))
        assert time_start_point >= 0 and time_stop_point <= self.delta_a_matrix.shape[
            1], 'invalid time range (> time range of computed reach set)'

        # map x_range
        for i in xrange(len(self.xlist) - 1, 0, -1):
            if x_range[0] <= self.xlist[i]:
                x_start_point = i

        for i in xrange(0, len(self.xlist)):
            if x_range[1] >= self.xlist[i]:
                x_stop_point = i

        space_indx = x_stop_point - x_start_point + 1
        time_indx = time_stop_point - time_start_point + 1

        min_vec = np.zeros((space_indx, time_indx), dtype=float)
        max_vec = np.zeros((space_indx, time_indx), dtype=float)

        for j in xrange(time_start_point, time_stop_point):
            for i in xrange(x_start_point, x_stop_point):
                delta_a = self.delta_a_matrix[i, j]
                delta_b = self.delta_b_matrix[i, j]
                delta_gamma_a = self.delta_gamma_a_matrix[i, j]
                delta_gamma_b = self.delta_gamma_b_matrix[i, j]
                delta_gamma_c = self.delta_gamma_c_matrix[i, j]
                delta_gamma_d = self.delta_gamma_d_matrix[i, j]






class Interpolation(object):
    'implements linear interpolation method'

    @staticmethod
    def interpolate_in_space(xlist, vector_Vn, vector_ln):
        'interpolation of u(x) in space at time t = tn'

        # at t = n*time_step, using FEM, we have U_n,i = alpha * vector_Vn,i + beta * vector_ln,i
        # i = 1, 2, ... m is discreted meshpoints

        #######################################################################
        # we want to find U_n(x) between two meshpoints i and i+1 at time t = (n*time_step)
        # using the linear interpolation in space, we have:
        # U_n(x) = (1/h[i+1]) * [(U_n,i+1 - Un,i) * x + (U_n,i * x[i + 1] - U_n,i+1 * x[i])]
        # where: x[i] <= x < x[i+1] and h[i + 1] = x[i + 1] - x[i]
        # using above U_n,i we obtain the interpolation set:
        #######################################################################
        # U_n(x) = (a_n,i * alpha + b_n,i * beta) * x + c_n,i * alpha + d_n,i * beta
        #######################################################################

        assert isinstance(xlist, list)
        for i in xrange(0, len(xlist) - 1):
            assert xlist[i] < xlist[i + 1], 'invalid list'

        assert isinstance(vector_Vn, np.ndarray)
        assert isinstance(vector_ln, np.ndarray)
        assert len(
            xlist) == vector_Vn.shape[0] == vector_ln.shape[0], 'inconsistent data'
        assert vector_Vn.shape[1] == vector_ln.shape[1] == 1, 'invalid vectors'

        n = len(xlist)
        Vn = vector_Vn
        ln = vector_ln
        a_n_vector = np.zeros((n - 1, 1), dtype=float)
        b_n_vector = np.zeros((n - 1, 1), dtype=float)
        c_n_vector = np.zeros((n - 1, 1), dtype=float)
        d_n_vector = np.zeros((n - 1, 1), dtype=float)

        for i in xrange(0, n - 1):
            hi_plus_1 = xlist[i + 1] - xlist[i]
            delta_Vn_i = Vn[i + 1, 0] - Vn[i, 0]
            delta_ln_i = ln[i + 1, 0] - ln[i, 0]

            a_n_vector[i, 0] = delta_Vn_i / hi_plus_1
            b_n_vector[i, 0] = delta_ln_i / hi_plus_1
            c_n_vector[i, 0] = (Vn[i, 0] * xlist[i + 1] -
                                Vn[i + 1, 0] * xlist[i]) / hi_plus_1
            d_n_vector[i, 0] = (ln[i, 0] * xlist[i + 1] -
                                ln[i + 1, 0] * xlist[i]) / hi_plus_1

        interpol_inspace_set = InterpolSetInSpace()
        interpol_inspace_set.set_values(
            xlist,
            a_n_vector,
            b_n_vector,
            c_n_vector,
            d_n_vector)

        return interpol_inspace_set

    @staticmethod
    def interpolate_in_space_for_all_timesteps(xlist, dreachset_list):
        'do interpolation in space for all time steps'

        # we need to use to reachable set at all time steps to do interpolation

        assert isinstance(xlist, list)
        for i in xrange(0, len(xlist) - 1):
            assert xlist[i] < xlist[i + 1], 'invalid list'

        assert isinstance(dreachset_list, list)
        for dreachset in dreachset_list:
            assert isinstance(
                dreachset, DReachSet), 'invalid discreted reach set'

        interpol_inspace_set_list = []
        for dreachset in dreachset_list:
            interpol_inspace_set_list.append(
                Interpolation.interpolate_in_space(
                    xlist, dreachset.Vn, dreachset.ln))

        return interpol_inspace_set_list

    @staticmethod
    def interpolate_in_time_and_space(step, interpol_inspace_set_list):
        'linear interpolation in time and space'

        assert isinstance(step, float)
        assert step > 0, 'invalid time step'
        assert isinstance(interpol_inspace_set_list, list)
        for interpol_set in interpol_inspace_set_list:
            assert isinstance(interpol_set, InterpolSetInSpace)

        n = len(interpol_inspace_set_list)    # number of time intervals
        assert n > 0, 'empty interpolation set'
        intpl_set_0 = interpol_inspace_set_list[0]
        m = intpl_set_0.a_vec.shape[0]    # number of space intervals

        delta_a_matrix = np.zeros((m, n), dtype=float)
        delta_b_matrix = np.zeros((m, n), dtype=float)
        delta_gamma_a_matrix = np.zeros((m, n), dtype=float)
        delta_gamma_b_matrix = np.zeros((m, n), dtype=float)
        delta_gamma_c_matrix = np.zeros((m, n), dtype=float)
        delta_gamma_d_matrix = np.zeros((m, n), dtype=float)

        for j in xrange(0, n - 1):
            intpl_set_j = interpol_inspace_set_list[j]
            intpl_set_j_plus_1 = interpol_inspace_set_list[j + 1]

            delta_a_vec_j = intpl_set_j_plus_1.a_vec - intpl_set_j.a_vec
            delta_b_vec_j = intpl_set_j_plus_1.b_vec - intpl_set_j.b_vec
            gamma_a_vec_j = np.multiply(intpl_set_j.a_vec, float(j))
            gamma_b_vec_j = np.multiply(intpl_set_j.b_vec, float(j))
            gamma_a_vec_j_plus_1 = np.multiply(
                intpl_set_j_plus_1.a_vec, float(j + 1))
            gamma_b_vec_j_plus_1 = np.multiply(
                intpl_set_j_plus_1.b_vec, float(j + 1))

            delta_gamma_a_vec_j = gamma_a_vec_j_plus_1 - gamma_a_vec_j
            delta_gamma_b_vec_j = gamma_b_vec_j_plus_1 - gamma_b_vec_j

            gamma_c_vec_j = np.multiply(intpl_set_j.c_vec, float(j))
            gamma_c_vec_j_plus_1 = np.multiply(
                intpl_set_j_plus_1.c_vec, float(j + 1))
            delta_gamma_c_vec_j = gamma_c_vec_j_plus_1 - gamma_c_vec_j
            gamma_d_vec_j = np.multiply(intpl_set_j.d_vec, float(j))
            gamma_d_vec_j_plus_1 = np.multiply(
                intpl_set_j_plus_1.d_vec, float(j + 1))
            delta_gamma_d_vec_j = gamma_d_vec_j_plus_1 - gamma_d_vec_j

            delta_a_matrix[:, j] = delta_a_vec_j
            delta_b_matrix[:, j] = delta_b_vec_j
            delta_gamma_a_matrix[:, j] = delta_gamma_a_vec_j
            delta_gamma_b_matrix[:, j] = delta_gamma_b_vec_j
            delta_gamma_c_matrix[:, j] = delta_gamma_c_vec_j
            delta_gamma_d_matrix[:, j] = delta_gamma_d_vec_j

        interpolation_set = InterpolationSet()
        interpolation_set.set_values(
            step,
            intpl_set_0.xlist,
            delta_a_matrix,
            delta_b_matrix,
            delta_gamma_a_matrix,
            delta_gamma_b_matrix,
            delta_gamma_c_matrix,
            delta_gamma_d_matrix)

        return interpolation_set
