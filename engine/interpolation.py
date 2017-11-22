'''
This module implements interpolation class and its methods
Dung Tran: Nov/2017
'''

from scipy.sparse import csc_matrix


class InterpolSetInSpace(object):
    'represent the set after doing interpolation in space'

    ##########################################################################
    # U_n(x) = (a1_n,i * alpha + b1_n,i * beta) * x + a2_n,i * alpha + b2_n,i * beta
    # x[i] <= x < x[i + 1], timestep = n
    ##########################################################################

    def __init__(self):
        self.a1_current_step_list = None
        self.b1_current_step_list = None
        self.a2_current_step_list = None
        self.b2_current_step_list = None

    def set_values(self, a1_current_step_list, b1_current_step_list,
                   a2_current_step_list, b2_current_step_list):
        'set values for parameter of interpolation set'

        assert isinstance(a1_current_step_list, list)
        assert isinstance(b1_current_step_list, list)
        assert isinstance(a2_current_step_list, list)
        assert isinstance(b2_current_step_list, list)

        assert len(a1_current_step_list) == len(b1_current_step_list) == len(
            a2_current_step_list) == len(b2_current_step_list), 'inconsistent data'

        self.a1_current_step_list = a1_current_step_list
        self.b1_current_step_list = b1_current_step_list
        self.a2_current_step_list = a2_current_step_list
        self.b2_current_step_list = b2_current_step_list


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
        # U_n(x) = (a1_n,i * alpha + b1_n,i * beta) * x + a2_n,i * alpha + b2_n,i * beta
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
        a1_n_list = []
        b1_n_list = []
        a2_n_list = []
        b2_n_list = []

        for i in xrange(0, n - 1):
            hi_plus_1 = xlist[i + 1] - xlist[i]
            delta_Vn_i = Vn[i + 1, 0] - Vn[i, 0]
            delta_ln_i = ln[i + 1, 0] - ln[i, 0]

            a1_n_i = delta_Vn_i / hi_plus_1
            b1_n_i = delta_ln_i / hi_plus_1
            a2_n_i = (Vn[i, 0] * xlist[i + 1] -
                      Vn[i + 1, 0] * xlist[i]) / hi_plus_1
            b2_n_i = (ln[i, 0] * xlist[i + 1] -
                      ln[i + 1, 0] * xlist[i]) / hi_plus_1

            a1_n_list.append(a1_n_i)
            b1_n_list.append(b1_n_i)
            a2_n_list.append(a2_n_i)
            b2_n_list.append(b2_n_i)

        interpol_set = InterpolSetInSpace()
        interpol_set.set_values(a1_n_list, b1_n_list, a2_n_list, b2_n_list)

        return interpol_set

    @staticmethod
    def interpolate_in_time(xlist, reach_set):
        'linear interpolation in time'
