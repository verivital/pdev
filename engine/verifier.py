'''
This module implements continuous/discreted verifier for PDE automaton
Dung Tran: Nov/2017
'''

from scipy.sparse import csc_matrix, lil_matrix
from scipy.optimize import minimize
from engine.pde_automaton import DPdeAutomaton
from engine.set import DReachSet
from engine.interpolation import Interpolation
from engine.fem import Fem1D
from engine.functions import Functions
from engine.specification import SafetySpecification
import math
import numpy as np


class ReachSetAssembler(object):
    'compute all necessary reachable sets'

    def __init__(self):
        self.u_dreachset = []
        self.err_dreachset = []
        self.bloated_dreachset = []

    @staticmethod
    def get_cur_u_dreachset(matrix_a, prev_u, cur_b_vec):
        'compute the current approximate discrete reachable set of u'

        # has the form of u[n] = Au[n-1] + b[n] = alpha * Vn + beta * ln
        assert isinstance(matrix_a, csc_matrix)
        assert isinstance(prev_u, DReachSet)
        assert isinstance(cur_b_vec, csc_matrix) and cur_b_vec.shape[1] == 1, 'invalid current vector b'
        assert matrix_a.shape[0] == cur_b_vec.shape[0] == prev_u.Vn.shape[0] == prev_u.ln.shape[0]

        cur_u_dreachset = DReachSet()
        cur_u_dreachset.Vn = matrix_a * prev_u.Vn
        cur_u_dreachset.ln = matrix_a * prev_u.ln + cur_b_vec
        cur_u_dreachset.alpha_range = prev_u.alpha_range
        cur_u_dreachset.beta_range = prev_u.beta_range

        return cur_u_dreachset

    @staticmethod
    def get_cur_err_dreachset(matrix_a, prev_e, cur_b_set):
        'compute the current approximate discreate reachable set of error e'

        # cur_b_set = alpha * Ve + beta * le
        # prev_e = alpha * Vn + beta * ln
        # e[n] = A * e[n-1] + b[n]

        assert isinstance(matrix_a, csc_matrix)
        assert isinstance(prev_e, DReachSet)
        assert isinstance(cur_b_set, DReachSet)
        assert matrix_a.shape[0] == cur_b_set.Vn.shape[0] == prev_e.Vn.shape[0] \
          == cur_b_set.ln.shape[0] == prev_e.ln.shape[0], 'inconsistent'

        cur_err_dreachset = DReachSet()
        cur_err_dreachset.Vn = matrix_a * prev_e.Vn + cur_b_set.Vn
        cur_err_dreachset.ln = matrix_a * prev_e.ln + cur_b_set.ln
        cur_err_dreachset.alpha_range = prev_e.alpha_range
        cur_err_dreachset.beta_range = prev_e.beta_range

        return cur_err_dreachset

    @staticmethod
    def get_cur_be(pre_u, cur_u, dPde, cur_time):
        'compute b[n], e[n] = A * e[n-1] + be[n]'

        assert isinstance(dPde, DPdeAutomaton)

        def get_V1_l1(V, l, xlist):
            '\int (u_n(x) p_i(x))dx = alpha * V1 + beta * l1'

            assert isinstance(V, csc_matrix)
            assert isinstance(l, csc_matrix)
            assert V.shape == l.shape
            n = V.shape[0]
            assert n == len(xlist) - 2

            V1 = lil_matrix((n, 1), dtype=float)
            l1 = lil_matrix((n, 1), dtype=float)
            for i in xrange(0, n):
                hi = xlist[i + 1] - xlist[i]
                hi_plus_1 = xlist[i + 2] - xlist[i + 1]
                if i == 0:
                    V1[i, 0] = V[i, 0] * (hi / 3 + hi_plus_1 / 3) + V[i + 1, 0] * hi_plus_1 / 6
                    l1[i, 0] = l[i, 0] * (hi / 3 + hi_plus_1 / 3) + l[i + 1, 0] * hi_plus_1 / 6
                elif i == n - 1:
                    V1[i, 0] = V[i, 0] * (hi / 3 + hi_plus_1 / 3) + V[i - 1, 0] * hi / 6
                    l1[i, 0] = l[i, 0] * (hi / 3 + hi_plus_1 / 3) + l[i - 1, 0] * hi / 6
                elif 0 < i < n - 1:
                    V1[i, 0] = V[i, 0] * (hi / 3 + hi_plus_1 / 3) + V[i - 1, 0] * hi / 6 + V[i + 1, 0] * hi_plus_1 / 6
                    l1[i, 0] = l[i, 0] * (hi / 3 + hi_plus_1 / 3) + l[i - 1, 0] * hi / 6 + l[i + 1, 0] * hi_plus_1 / 6

            return V1, l1

        cur_be = DReachSet()
        pre_V1, pre_l1 = get_V1_l1(pre_u.Vn, pre_u.ln, dPde.xlist)
        cur_V1, cur_l1 = get_V1_l1(cur_u.Vn, cur_u.ln, dPde.xlist)

        cur_b_vec = Fem1D.load_assembler(dPde.xlist, dPde.f_xdom, dPde.time_step, cur_time)
        cur_be.Vn = dPde.inv_b_matrix * (pre_V1 - cur_V1)
        cur_be.ln = dPde.inv_b_matrix * (cur_b_vec + pre_l1 - cur_l1)
        cur_be.alpha_range = dPde.alpha_range
        cur_be.beta_range = dPde.beta_range

        return cur_be

    @staticmethod
    def get_dreachset(dPde, toTimeStep):
        'compute approximate discrete reachable set of u and e and the bloated u + e'

        assert isinstance(dPde, DPdeAutomaton)
        assert isinstance(toTimeStep, int) and toTimeStep >= 0

        u_dreachset_list = []
        err_dreachset_list = []
        bloated_dreachset_list = []

        for cur_time in xrange(0, toTimeStep + 1):
            err_dreachset = DReachSet()
            u_dreachset = DReachSet()
            if cur_time == 0:
                u_Vn = dPde.init_vector
                u_ln = csc_matrix((dPde.init_vector.shape[0], 1), dtype=float)
                err_Vn = csc_matrix((dPde.init_vector.shape[0], 1), dtype=float)
                err_ln = csc_matrix((dPde.init_vector.shape[0], 1), dtype=float)
                u_dreachset.set_reach_set(dPde.alpha_range, dPde.beta_range, u_Vn, u_ln)
                err_dreachset.set_reach_set(dPde.alpha_range, dPde.beta_range, err_Vn, err_ln)

            else:
                cur_b_vec = Fem1D.load_assembler(dPde.xlist, dPde.f_xdom, dPde.time_step, cur_time)
                u_dreachset = ReachSetAssembler.get_cur_u_dreachset(dPde.matrix_a, \
                                                                        u_dreachset_list[cur_time - 1], cur_b_vec)

                cur_be = ReachSetAssembler.get_cur_be(u_dreachset_list[cur_time - 1], u_dreachset, dPde, cur_time)
                err_dreachset = ReachSetAssembler.get_cur_err_dreachset(dPde.matrix_a, err_dreachset_list[cur_time - 1], cur_be)

            u_dreachset_list.append(u_dreachset)
            err_dreachset_list.append(err_dreachset)

            bloated_dreachset = DReachSet()
            bloated_dreachset.set_reach_set(dPde.alpha_range, dPde.beta_range, u_dreachset.Vn + err_dreachset.Vn, u_dreachset.ln + err_dreachset.ln)
            bloated_dreachset_list.append(bloated_dreachset)

        return u_dreachset_list, err_dreachset_list, bloated_dreachset_list

    @staticmethod
    def get_interpolationset(dPde, toTimeStep):
        'compute the interpolation set in both space and time'

        assert isinstance(dPde, DPdeAutomaton)
        assert isinstance(toTimeStep, int) and toTimeStep >= 0

        u_dset, e_dset, bl_dset = ReachSetAssembler.get_dreachset(dPde, toTimeStep)
        n = len(u_dset)
        u_setinspace_list = []    # interpolation set of u set in space
        e_setinspace_list = []    # interpolation set of error set in space
        bl_setinspace_list = []    # interpolation set of bloated set in space
        u_set_list = []    # interpolation set of u in both time and space
        e_set_list = []    # interpolation set of error set in both time and space
        bl_set_list = []    # interpolation set of bloated set in both time and space

        # interpolation set in space
        for i in xrange(0, n):
            u_setinspace_list.append(Interpolation.interpolate_in_space(dPde.xlist, u_dset[i].Vn.todense(), u_dset[i].ln.todense()))
            e_setinspace_list.append(Interpolation.interpolate_in_space(dPde.xlist, e_dset[i].Vn.todense(), e_dset[i].ln.todense()))
            bl_setinspace_list.append(Interpolation.interpolate_in_space(dPde.xlist, bl_dset[i].Vn.todense(), bl_dset[i].ln.todense()))

        # interpolation set in both space and time
        for i in xrange(1, n):
            u_set_list.append(Interpolation.increm_interpolation(dPde.time_step, i, u_setinspace_list[i - 1], u_setinspace_list[i]))
            e_set_list.append(Interpolation.increm_interpolation(dPde.time_step, i, e_setinspace_list[i - 1], e_setinspace_list[i]))
            bl_set_list.append(Interpolation.increm_interpolation(dPde.time_step, i, bl_setinspace_list[i - 1], bl_setinspace_list[i]))

        return u_setinspace_list, e_setinspace_list, bl_setinspace_list, u_set_list, e_set_list, bl_set_list


class VerificationResult(object):
    'Result object for verification'

    def __init__(self):
        self.status = None                # status Safe/Unsafe
        self.unsafe_time_point = None     # time that system reach unsafe state
        self.unsafe_x_point = None        # position that system reach unsafe state
        self.unsafe_u_point = None        # value of u(x,t) at unsafe state
        self.unsafe_trace_funcs = []    # u(x,t) at unsafe_x_point is a list of functions of t
        self.unsafe_numerical_trace = None    # numerical trace for plotting
        self.step = None    # time step

    def generate_numerical_trace(self):
        'generate a numerical trace for unsafe case'

        assert self.step is not None and self.step > 0, 'Verification Result is empty object'
        assert isinstance(self.unsafe_trace_funcs, list) and self.unsafe_trace_funcs != []

        n = len(self.unsafe_trace_funcs)
        time_list = []
        u_list = []
        for j in xrange(0, n):
            func = self.unsafe_trace_funcs[j]
            time_list.append(j * self.step)
            u_list.append(func([time_list[j]]))

        return (time_list, u_list)


class Verifier(object):
    'verifier for the pde automaton'

    def __init__(self):

        self.result = VerificationResult()

    def check_safety(self, dPde, safety_specification):
        'verify safety of Pde automaton'

        assert isinstance(dPde, DPdeAutomaton)
        assert isinstance(safety_specification, SafetySpecification)

        # check consistency
        xlist = dPde.xlist
        step = dPde.time_step
        self.result.step = step
        assert xlist is not None, 'empty dPde'
        x_range = safety_specification.x_range

        if x_range[0] < xlist[0] or x_range[1] > xlist[len(xlist) - 1]:
            raise ValueError('x_range is out of range of dPde.xlist')

        u1 = safety_specification.u1
        u2 = safety_specification.u2
        assert u1 is not None or u2 is not None, 'u1 and u2 are both None'
        x1 = safety_specification.x_range[0]
        x2 = safety_specification.x_range[1]
        T1 = safety_specification.t_range[0]
        T2 = safety_specification.t_range[1]

        end_time_step = int(math.ceil(T2 / step))
        start_time_step = int(math.floor(T1 / step))

        m = len(xlist)
        for i in xrange(1, m):
            if xlist[i - 1] <= x1 < xlist[i]:
                start_point = i - 1
                break
            elif x1 == xlist[i]:
                start_point = i
                break

        for i in xrange(0, m):
            if xlist[m - 2 - i] < x2 <= xlist[m - 1 - i]:
                end_point = m - 1 - i
                break
            elif x2 == xlist[m - 2 - i]:
                end_point = m - 2 - i
                break

        # compute continuous reachable set
        _, _, _, _, _, bloated_set = ReachSetAssembler.get_interpolationset(dPde, end_time_step)

        # decompose x1 x2 into list of x_range
        x_range_list = []

        for i in xrange(start_point, end_point):
            if i == start_point:
                x_range_list.append((x1, xlist[start_point + 1]))
            elif i == end_point - 1:
                x_range_list.append((xlist[end_point - 2], x2))
            elif start_point < i < end_point - 1:
                x_range_list.append((xlist[i], xlist[i + 1]))

        # decompose T1, T2 into list of t_range
        t_range_list = []
        for j in xrange(start_time_step, end_time_step):
            if j == start_time_step:
                t_range_list.append((T1, (start_time_step + 1) * step))
            elif j == end_time_step - 1:
                t_range_list.append(((end_time_step - 2) * step, T2))
            elif start_time_step < j < end_time_step - 1:
                t_range_list.append((j * step, (j + 1) * step))

        # check safety
        print "\nstart_time_step = {}".format(start_time_step)
        print "\nend_time_step = {}".format(end_time_step)
        for j in xrange(start_time_step, end_time_step):
            print "\n j = {}".format(j)
            bl_set = bloated_set[j]
            time_range = t_range_list[j - start_time_step]
            print "\ntime_range = {}".format(time_range)

            for i in xrange(start_point, end_point):
                print "\ni = {}".format(i)
                x_range = x_range_list[i - start_point]
                print "\nx_range = {}".format(x_range)

                min_func = Functions.intpl_in_time_and_space_func(step, bl_set.delta_a_vec[i],
                                                                  bl_set.delta_b_vec[i],
                                                                  bl_set.delta_gamma_a_vec[i],
                                                                  bl_set.delta_gamma_b_vec[i],
                                                                  bl_set.delta_c_vec[i],
                                                                  bl_set.delta_d_vec[i],
                                                                  bl_set.delta_gamma_c_vec[i],
                                                                  bl_set.delta_gamma_d_vec[i])

                max_func = Functions.intpl_in_time_and_space_func(step, - bl_set.delta_a_vec[i],
                                                                  - bl_set.delta_b_vec[i],
                                                                  - bl_set.delta_gamma_a_vec[i],
                                                                  - bl_set.delta_gamma_b_vec[i],
                                                                  - bl_set.delta_c_vec[i],
                                                                  - bl_set.delta_d_vec[i],
                                                                  - bl_set.delta_gamma_c_vec[i],
                                                                  - bl_set.delta_gamma_d_vec[i])

                x0 = [time_range[0], x_range[0], dPde.alpha_range[0], dPde.beta_range[0]]

                bnds = (time_range, x_range, dPde.alpha_range, dPde.beta_range)

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

                min_points = []
                max_points = []
                if min_res.status == 0:
                    min_value = min_res.fun
                    min_points.append(min_res.x)
                else:
                    print "\nmin_res.status = {}".format(min_res.status)
                    print "\nminimization message: {}".format(min_res.message)
                    raise ValueError(
                        'minimization for interpolation function fail!')

                if max_res.status == 0:
                    max_value = -max_res.fun
                    max_points.append(max_res.x)
                else:
                    print "\nmax_res.status = {}".format(max_res.status)
                    print "\nmaximization message: {}".format(max_res.message)
                    raise ValueError(
                        'maximization for interpolation function fail!')

                if u1 is not None and u2 is not None:
                    if min_value < u1:
                        self.result.status = 'Unsafe'
                        self.result.unsafe_u_point = min_value
                        feas_sol = min_points
                        break
                    elif max_value > u2:
                        self.result.status = 'Unsafe'
                        self.result.unsafe_u_point = max_value
                        feas_sol = max_points
                        break
                elif u1 is None and u2 is not None:
                    if max_value > u2:
                        self.result.status = 'Unsafe'
                        self.result.unsafe_u_point = max_value
                        feas_sol = max_points
                        break
                elif u1 is not None and u2 is None:
                    if min_value < u1:
                        self.result.status = 'Unsafe'
                        self.result.unsafe_u_point = min_value
                        feas_sol = min_points
                        break

            if self.result.status == 'Unsafe':
                fs = feas_sol[0]
                self.result.unsafe_time_point = fs[0]
                self.result.unsafe_x_point = fs[1]
                alpha_value = fs[2]
                beta_value = fs[3]
                print "\nfeas_solution = {}".format(feas_sol)
                break

        # return safe or unsafe and unsafe trace which is a list of function of t
        self.result.unsafe_trace_funcs = []
        if self.result.status == 'Unsafe':
            for j in xrange(0, end_time_step):
                bl_set = bloated_set[j]
                self.result.unsafe_trace_funcs.append(bl_set.get_trace_func(alpha_value, beta_value, self.result.unsafe_x_point))
        else:
            self.result.status = 'Safe'

        return self.result
