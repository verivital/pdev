'''
This module implements interpolation class and its methods
Dung Tran: Nov/2017
'''

from scipy.sparse import csc_matrix


class Interpolation(object):
    'implements linear interpolation method'

    @staticmethod
    def interpolate_in_space(xlist, vector_Vn, vector_ln):
        'interpolation of u(x) in space at time t = tn'

        # at t = n*time_step, using FEM, we have U_n,i = alpha * vector_Vn,i + beta * vector_ln,i
        # i = 1, 2, ... m is discreted meshpoints

        ###################################################################################
        # we want to find U_n(x) between two meshpoints i and i+1 at time t = (n*time_step)
        # using the linear interpolation in space, we have:
        # U_n(x) = (1/h[i+1]) * [(U_n,i+1 - Un,i) * x + (U_n,i * x[i + 1] - U_n,i+1 * x[i])
        # where: x[i] <= x <= x[i+1] and h[i + 1] = x[i + 1] - x[i]
        ###################################################################################

        assert isinstance(xlist, list)
        assert isinstance(vector_Vn, csc_matrix)
        assert isinstance(vector_ln, csc_matrix)
        assert len(xlist) == vector_Vn.shape[0] == vector_ln.shape[0], 'inconsistent data'
        assert vector_Vn.shape[1] == vector_ln.shape[1] == 1, 'invalid vectors'

        n = len(xlist)
        hi_plus_1_list = []
        delta_Vn_list = []
        delta_ln_list = []
        Vn = vector_Vn.tolil()
        ln = vector_ln.tolil()

        for i in xrange(0, n - 1):
            hi_plus_1 = xlist[i + 1] - xlist[i]
            delta_Vn = Vn[i + 1, 0] - Vn[i, 0]
            delta_ln = ln[i + 1, 0] - ln[i, 0]

            hi_plus_1_list.append(hi_plus_1)
            delta_Vn_list.append(delta_Vn)
            delta_ln_list.append(delta_ln)
