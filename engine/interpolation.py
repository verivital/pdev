'''
This module implements interpolation class and its methods
Dung Tran: Nov/2017
'''

from scipy.sparse import csc_matrix, lil_matrix
from engine.set import DReachSet


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

    def set_values(self, a_current_step_vec, b_current_step_vec,
                   c_current_step_vec, d_current_step_vec):
        'set values for parameter of interpolation set'

        assert isinstance(a_current_step_vec, csc_matrix)
        assert isinstance(b_current_step_vec, csc_matrix)
        assert isinstance(c_current_step_vec, csc_matrix)
        assert isinstance(d_current_step_vec, csc_matrix)

        assert a_current_step_vec.shape == b_current_step_vec.shape == \
            c_current_step_vec.shape == d_current_step_vec.shape, 'inconsistent data'
        assert a_current_step_vec.shape[1] == 1, 'invalid shape'

        self.a_vec = a_current_step_vec
        self.b_vec = b_current_step_vec
        self.c_vec = c_current_step_vec
        self.d_vec = d_current_step_vec


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
        # [delta_a_i,j], i = 1, .., m (space step) and j = 1, ..., n (time step)
        self.delta_a_matrix = None
        # [delta_b_i,j], i = 1, .., m (space step) and j = 1, ..., n (time step)
        self.delta_b_matrix = None
        # [delta_gamma_c_i,j], i = 1, .., m (space step) and j = 1, ..., n (time step)
        self.delta_gamma_c_matrix = None
        # [delta_gamma_d_i,j], i = 1, .., m (space step) and j = 1, ..., n (time step)
        self.delta_gamma_d_matrix = None

    def set_values(self, step, delta_a_matrix, delta_b_matrix,
                   delta_gamma_c_matrix, delta_gamma_d_matrix):

        assert isinstance(step, float)
        assert step > 0, 'invalid time step'
        assert isinstance(delta_a_matrix, csc_matrix)
        assert isinstance(delta_b_matrix, csc_matrix)
        assert isinstance(delta_gamma_c_matrix, csc_matrix)
        assert isinstance(delta_gamma_d_matrix, csc_matrix)

        assert delta_a_matrix.shape == delta_b_matrix.shape == delta_gamma_c_matrix.shape == \
            delta_gamma_d_matrix.shape, 'invalid data set'

        self.step = step
        self.delta_a_matrix = delta_a_matrix
        self.delta_b_matrix = delta_b_matrix
        self.delta_gamma_c_matrix = delta_gamma_c_matrix
        self.delta_gamma_d_matrix = delta_gamma_d_matrix


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

        assert isinstance(vector_Vn, csc_matrix)
        assert isinstance(vector_ln, csc_matrix)
        assert len(
            xlist) == vector_Vn.shape[0] == vector_ln.shape[0], 'inconsistent data'
        assert vector_Vn.shape[1] == vector_ln.shape[1] == 1, 'invalid vectors'

        n = len(xlist)
        Vn = vector_Vn.tolil()
        ln = vector_ln.tolil()
        a_n_vector = lil_matrix((n - 1, 1), dtype=float)
        b_n_vector = lil_matrix((n - 1, 1), dtype=float)
        c_n_vector = lil_matrix((n - 1, 1), dtype=float)
        d_n_vector = lil_matrix((n - 1, 1), dtype=float)

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
            a_n_vector.tocsc(),
            b_n_vector.tocsc(),
            c_n_vector.tocsc(),
            d_n_vector.tocsc())

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

        delta_a_matrix = lil_matrix((m, n), dtype=float)
        delta_b_matrix = lil_matrix((m, n), dtype=float)
        delta_gamma_c_matrix = lil_matrix((m, n), dtype=float)
        delta_gamma_d_matrix = lil_matrix((m, n), dtype=float)

        for j in xrange(0, n - 1):
            intpl_set_j = interpol_inspace_set_list[j]
            intpl_set_j_plus_1 = interpol_inspace_set_list[j + 1]

            delta_a_vec_j = intpl_set_j_plus_1.a_vec - intpl_set_j.a_vec
            delta_b_vec_j = intpl_set_j_plus_1.b_vec - intpl_set_j.b_vec
            gamma_c_vec_j = intpl_set_j.c_vec.tolil().multiply(float(j))
            gamma_c_vec_j_plus_1 = intpl_set_j_plus_1.c_vec.tolil().multiply(float(j + 1))
            delta_gamma_c_vec_j = gamma_c_vec_j_plus_1 - gamma_c_vec_j
            gamma_d_vec_j = intpl_set_j.d_vec.tolil().multiply(float(j))
            gamma_d_vec_j_plus_1 = intpl_set_j_plus_1.d_vec.tolil().multiply(float(j + 1))
            delta_gamma_d_vec_j = gamma_d_vec_j_plus_1 - gamma_d_vec_j

            delta_a_matrix[:, j] = delta_a_vec_j
            delta_b_matrix[:, j] = delta_b_vec_j
            delta_gamma_c_matrix[:, j] = delta_gamma_c_vec_j
            delta_gamma_d_matrix[:, j] = delta_gamma_d_vec_j

        interpolation_set = InterpolationSet()
        interpolation_set.set_values(
            step,
            delta_a_matrix.tocsc(),
            delta_b_matrix.tocsc(),
            delta_gamma_c_matrix.tocsc(),
            delta_gamma_d_matrix.tocsc())

        return interpolation_set
