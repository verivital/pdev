'''
This module implements interpolation class and its methods
Dung Tran: Nov/2017
'''

from scipy.sparse import csc_matrix


class InterpolSetInSpace(object):
    'represent the set after doing interpolation in space'

    ##########################################################################
    # U_n(x) = (a_n,i * alpha + b_n,i * beta) * x + c_n,i * alpha + d_n,i * beta
    # x[i] <= x < x[i + 1], timestep = n
    ##########################################################################

    def __init__(self):
        self.a_current_step_list = None
        self.b_current_step_list = None
        self.c_current_step_list = None
        self.d_current_step_list = None

    def set_values(self, a_current_step_list, b_current_step_list,
                   c_current_step_list, d_current_step_list):
        'set values for parameter of interpolation set'

        assert isinstance(a_current_step_list, list)
        assert isinstance(b_current_step_list, list)
        assert isinstance(c_current_step_list, list)
        assert isinstance(d_current_step_list, list)

        assert len(a_current_step_list) == len(b_current_step_list) == len(
            c_current_step_list) == len(d_current_step_list), 'inconsistent data'

        self.a_current_step_list = a_current_step_list
        self.b_current_step_list = b_current_step_list
        self.c_current_step_list = c_current_step_list
        self.d_current_step_list = d_current_step_list


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
        assert isinstance(vector_Vn, csc_matrix)
        assert isinstance(vector_ln, csc_matrix)
        assert len(
            xlist) == vector_Vn.shape[0] == vector_ln.shape[0], 'inconsistent data'
        assert vector_Vn.shape[1] == vector_ln.shape[1] == 1, 'invalid vectors'

        n = len(xlist)
        Vn = vector_Vn.tolil()
        ln = vector_ln.tolil()
        a_n_list = []
        b_n_list = []
        c_n_list = []
        d_n_list = []

        for i in xrange(0, n - 1):
            hi_plus_1 = xlist[i + 1] - xlist[i]
            delta_Vn_i = Vn[i + 1, 0] - Vn[i, 0]
            delta_ln_i = ln[i + 1, 0] - ln[i, 0]

            a_n_i = delta_Vn_i / hi_plus_1
            b_n_i = delta_ln_i / hi_plus_1
            c_n_i = (Vn[i, 0] * xlist[i + 1] -
                     Vn[i + 1, 0] * xlist[i]) / hi_plus_1
            d_n_i = (ln[i, 0] * xlist[i + 1] -
                     ln[i + 1, 0] * xlist[i]) / hi_plus_1

            a_n_list.append(a_n_i)
            b_n_list.append(b_n_i)
            c_n_list.append(c_n_i)
            d_n_list.append(d_n_i)

        interpol_set = InterpolSetInSpace()
        interpol_set.set_values(a_n_list, b_n_list, c_n_list, d_n_list)

        return interpol_set

    @staticmethod
    def interpolate_in_time(xlist, reach_set):
        'linear interpolation in time'

        pass
