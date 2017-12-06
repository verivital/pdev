'''
This module implements continuous/discreted verifier for PDE automaton
Dung Tran: Nov/2017
'''

from scipy.sparse import csc_matrix, lil_matrix, hstack
from engine.pde_automaton import DPdeAutomaton
from engine.set import DReachSet, GeneralSet, RectangleSet2D, RectangleSet3D
from engine.interpolation import Interpolation
from engine.fem import Fem1D


class Verifier(object):
    'verifier for discreted pde automaton'

    # verify the safety of discreted pde automaton from step 0 to step N
    # if unsafe region is reached, produce a trace

    def __init__(self):

        # use for computing and checking reach set of discrete Pde
        self.status_dis = None    # 0=safe /1=unsafe
        self.current_V = None    # Vn = A^n * Vn-1, V0 = U0
        self.current_l = None    # ln = Sigma_[i=0 to i = n-1] (A^i * b)
        # include all reach sets of discrete Pde from 0 to current step
        self.to_current_step_set = []
        self.to_current_step_line_set = []
        # current constraint to check safety based on discrete reach set
        self.current_constraints = None
        self.unsafe_trace = []    # trace for unsafe case of discrete Pdeautomaton

        # use for error analysis
        self.residual_r_u = None

        # use for computing and checking interpolation set (piecewise
        # continuous in space and time)
        self.status_cont = None
        # list of interpolation set in space upto current step
        self.to_cur_step_intpl_inspace_set = []
        # list of interpolation set in both time and space upto current step
        self.to_cur_step_intpl_set = []

    def get_dreach_set(self, dPde, toTimeStep):
        'compute reach set of discrete PDE to the toTimeStep in specific direction'

        assert isinstance(toTimeStep, int)
        assert toTimeStep >= 0
        assert isinstance(dPde, DPdeAutomaton)

        current_set = DReachSet()
        self.current_V = None
        self.current_l = None
        self.to_current_step_set = []

        for i in xrange(0, toTimeStep + 1):

            if i == 0:
                self.current_V = dPde.init_vector
                self.current_l = csc_matrix((dPde.init_vector.shape[0], 1), dtype=float)
            else:
                self.current_V = dPde.matrix_a * self.current_V
                current_vector_b = Fem1D.load_assembler(dPde.xlist, dPde.f_xdom, dPde.time_step, i)
                dPde.set_vector_b(current_vector_b)
                self.current_l = current_vector_b + dPde.matrix_a * self.current_l
            current_set.set_reach_set(dPde.alpha_range, dPde.beta_range, self.current_V, self.current_l)
            self.to_current_step_set.append(current_set)

        return self.to_current_step_set

    def on_fly_check_dPde(self, dPde, toTimeStep):
        'On-the-fly safety checking for discrete Pde automaton '

        assert dPde.matrix_a is not None, 'specify dPde first'
        assert dPde.unsafe_set is not None, 'specify unsafe set first'
        assert dPde.alpha_range is not None, 'specify range of perturbation first'
        assert dPde.beta_range is not None, 'specify perturbation first'

        direct_matrix = dPde.unsafe_set.matrix_c
        unsafe_vector = dPde.unsafe_set.vector_d
        self.current_V = None
        self.current_l = None

        current_constraints = GeneralSet()

        for i in xrange(0, toTimeStep + 1):
            if i == 0:
                self.current_V = dPde.init_vector
                self.current_l = csc_matrix(
                    (dPde.init_vector.shape[0], 1), dtype=float)
                inDirection_Current_V = direct_matrix * self.current_V
                inDirection_Current_l = direct_matrix * self.current_l

            else:
                self.current_V = dPde.matrix_a * self.current_V
                current_vector_b = Fem1D.load_assembler(dPde.xlist, dPde.f_xdom, dPde.time_step, i)
                dPde.set_vector_b(current_vector_b)
                self.current_l = current_vector_b + dPde.matrix_a * self.current_l

                inDirection_Current_V = direct_matrix * self.current_V
                inDirection_Current_l = direct_matrix * self.current_l

            print "\n V_{} = \n{}, \n l_{} = {}".format(i, self.current_V.todense(), i, self.current_l.todense())

            constraint_matrix = hstack([inDirection_Current_V, inDirection_Current_l])
            current_constraints.set_constraints(constraint_matrix, unsafe_vector)    # construct constraints for current step

            # check feasible of current constraint
            feasible_res = current_constraints.check_feasible(dPde.alpha_range, dPde.beta_range)
            if feasible_res.status == 2:
                self.status_dis = 0    # discreted pde system is safe
            elif feasible_res.status == 0:
                self.status_dis = 1    # discreted pde system is unsafe
            elif feasible_res == 1:
                self.status_dis = 2    # iteration limit reached
            elif feasible_res == 3:
                self.status_dis = 3    # problem appears to be unbounded

            if self.status_dis == 0:
                print"\nTimeStep {}: SAFE".format(i)
            elif self.status_dis == 1:
                print"\nTimeStep {}: UNSAFE".format(i)
                feasible_alpha_beta = feasible_res.x
                feasible_alpha = feasible_alpha_beta[0]
                feasible_beta = feasible_alpha_beta[1]
                print "\nalpha = {}, beta = {}".format(feasible_alpha, feasible_beta)
                # produce a trace lead dPde to unsafe region
                self.unsafe_trace = dPde.get_trace(feasible_alpha, feasible_beta, i)
            else:
                print"\nTimeStep{}: Error in checking safe/unsafe"

    def get_interpolation_set(self, dPde, toTimeStep):
        'compute interpolation set to toTimeStep'

        assert isinstance(toTimeStep, int)
        assert toTimeStep >= 1
        assert isinstance(dPde, DPdeAutomaton)

        self.current_V = None
        self.current_l = None

        for i in xrange(0, toTimeStep + 1):

            # get interpolation set in space
            if i == 0:
                self.current_V = dPde.init_vector
                self.current_l = csc_matrix(
                    (dPde.init_vector.shape[0], 1), dtype=float)
            else:
                self.current_V = dPde.matrix_a * self.current_V
                current_vector_b = Fem1D.load_assembler(dPde.xlist, dPde.f_xdom, dPde.time_step, i)
                dPde.set_vector_b(current_vector_b)
                self.current_l = current_vector_b + dPde.matrix_a * self.current_l

            cur_intpl_inspace_set = Interpolation.interpolate_in_space(
                dPde.xlist, self.current_V.todense(), self.current_l.todense())
            self.to_cur_step_intpl_inspace_set.append(cur_intpl_inspace_set)

            # get interpolation set in both time and space
            if i >= 1:
                cur_intpl_set = Interpolation.increm_interpolation(
                    dPde.time_step, i, self.to_cur_step_intpl_inspace_set[i - 1], self.to_cur_step_intpl_inspace_set[i])
                self.to_cur_step_intpl_set.append(cur_intpl_set)

        return self.to_cur_step_intpl_inspace_set, self.to_cur_step_intpl_set

    def get_intpl_inspace_boxes(self, dPde, toTimeStep):
        'get boxes containing interpolation inspace set U_n(x) at time step t = n * step'

        assert isinstance(dPde, DPdeAutomaton)
        self.get_interpolation_set(dPde, toTimeStep)
        assert dPde.alpha_range is not None and dPde.beta_range is not None, 'set range for alpha and beta'
        n = len(self.to_cur_step_intpl_inspace_set)

        boxes_list = []    # list of boxes along time step

        for j in xrange(0, n):
            intpl_inspace_set = self.to_cur_step_intpl_inspace_set[j]
            u_min_vec, _, u_max_vec, _ = intpl_inspace_set.get_min_max(
                dPde.alpha_range, dPde.beta_range)

            boxes = []    # list of boxes along space step
            for i in xrange(0, u_min_vec.shape[0]):
                rect = RectangleSet2D()
                rect.set_bounds(dPde.xlist[i], dPde.xlist[i + 1], u_min_vec[i], u_max_vec[i])
                boxes.append(rect)

            boxes_list.append(boxes)

        return boxes_list

    def get_intpl_boxes(self, dPde, toTimeStep):
        'get boxes containing interpolation set U(x,t)'

        assert isinstance(dPde, DPdeAutomaton)
        self.get_interpolation_set(dPde, toTimeStep)
        assert dPde.alpha_range is not None and dPde.beta_range is not None, 'set range for alpha and beta'
        n = len(self.to_cur_step_intpl_set)

        boxes_list = []

        for j in xrange(0, n):
            intpl_set = self.to_cur_step_intpl_set[j]
            u_min_vec, _, u_max_vec, _ = intpl_set.get_min_max(dPde.alpha_range, dPde.beta_range)

            boxes = []    # list of boxes along space step
            for i in xrange(0, u_min_vec.shape[0]):
                rect3d = RectangleSet3D()
                xmin = dPde.xlist[i]
                xmax = dPde.xlist[i + 1]
                ymin = float(j) * dPde.time_step
                ymax = float(j + 1) * dPde.time_step
                rect3d.set_bounds(xmin, xmax, ymin, ymax, u_min_vec[i], u_max_vec[i])
                boxes.append(rect3d)

            boxes_list.append(boxes)

        return boxes_list

    def compute_residul_r_u(self, dPde):
        'get compute r(u) = beta * f - u, use later to compute the error reachable set'

        assert isinstance(dPde, DPdeAutomaton)
        assert self.to_current_step_set is not None, 'no discrete reach set to comput r(u) = beta * f - u'

        n = len(self.to_current_step_set)
        assert n >= 2, 'need at least two discrete reach set to compute r(u)'
        residual_r_u = DReachSet()
        dPde.residual_r_u = []    # reset list of residual

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

        for i in xrange(1, n):
            prev_set = self.to_current_step_set[i - 1]
            cur_set = self.to_current_step_set[i]

            prev_V1, prev_l1 = get_V1_l1(prev_set.Vn, prev_set.ln, dPde.xlist)
            cur_V1, cur_l1 = get_V1_l1(cur_set.Vn, cur_set.ln, dPde.xlist)

            residual_r_u.Vn = prev_V1 - cur_V1
            residual_r_u.ln = dPde.vector_b[i] + prev_l1 - cur_l1
            dPde.residual_r_u.append(residual_r_u)

        return dPde, dPde.residual_r_u
