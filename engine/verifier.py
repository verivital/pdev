'''
This module implements continuous/discreted verifier for PDE automaton
Dung Tran: Nov/2017
'''

from scipy.sparse import csc_matrix, hstack, vstack
from engine.pde_automaton import DPdeAutomaton
from engine.set import DReachSet, GeneralSet


class Verifier(object):
    'verifier for discreted pde automaton'

    # verify the safety of discreted pde automaton from step 0 to step N
    # if unsafe region is reached, produce a trace

    def __init__(self):

        self.status = None    # 0=safe /1=unsafe
        self.current_V = None    # Vn = A^n * Vn-1, V0 = U0
        self.current_l = None    # ln = Sigma_[i=0 to i = n-1] (A^i * b)
        self.to_current_step_set = []     # include all reach sets from 0 to current step
        self.current_constraints = None    # current constraint to check safety based on discreted reach set
        self.unsafe_trace = []    # trace for unsafe case

    def compute_reach_set(self, dPde, toTimeStep):
        'compute reach set of discreted PDE to the toTimeStep in specific direction'

        assert isinstance(toTimeStep, int)
        assert toTimeStep >= 0
        assert isinstance(dPde, DPdeAutomaton)

        current_set = DReachSet()
        self.current_V = []
        self.current_l = []
        self.to_current_step_set = []

        for i in xrange(0, toTimeStep + 1):

            if i == 0:
                self.current_V = dPde.init_vector
                self.current_l = csc_matrix(
                    (dPde.init_vector.shape[0], 1), dtype=float)
                current_set.set_reach_set(
                    dPde.perturbation, self.current_V, self.current_l)
                self.to_current_step_set.append(current_set)
            else:
                self.current_V = dPde.matrix_a * self.current_V
                self.current_l = dPde.vector_b + dPde.matrix_a * self.current_l
                current_set.set_reach_set(
                    dPde.perturbation, self.current_V, self.current_l)
                self.to_current_step_set.append(current_set)

        return self.to_current_step_set

    def on_fly_check(self, dPde, toTimeStep):
        'On-the-fly safety checking'

        assert dPde.matrix_a is not None, 'specify dPde first'
        assert dPde.unsafe_set is not None, 'specify unsafe set first'

        direct_matrix = dPde.unsafe_set.matrix_c
        unsafe_vector = dPde.unsafe_set.vector_d
        self.current_V = []
        self.current_l = []
        per_set = dPde.perturbation
        per_matrix = per_set.matrix_c
        per_vector = per_set.vector_d

        constraint_vector = vstack([unsafe_vector, per_vector])
        current_constraints = GeneralSet()

        for i in xrange(0, toTimeStep + 1):
            if i == 0:
                self.current_V = dPde.init_vector
                self.current_l = csc_matrix(
                    (dPde.init_vector.shape[0], 1), dtype=float)
                inDirection_Current_V = direct_matrix * self.current_V
                inDirection_Current_l = direct_matrix * self.current_l

                C1 = hstack([inDirection_Current_V, inDirection_Current_l])
                constraint_matrix = vstack([C1, per_matrix])    # construct constrains for current step

            else:
                self.current_V = dPde.matrix_a * self.current_V
                self.current_l = dPde.vector_b + dPde.matrix_a * self.current_l

                inDirection_Current_V = direct_matrix * self.current_V
                inDirection_Current_l = direct_matrix * self.current_l
                C1 = hstack([inDirection_Current_V, inDirection_Current_l])
                constraint_matrix = vstack([C1, per_matrix])

            current_constraints.set_constraints(
                constraint_matrix.tocsc(), constraint_vector.tocsc())    # construct constraints for current step

            # check feasible of current constraint
            feasible_res = current_constraints.check_feasible()
            if feasible_res.status == 2:
                self.status = 0    # discreted pde system is safe
            elif feasible_res.status == 0:
                self.status = 1    # discreted pde system is unsafe
            elif feasible_res == 1:
                self.status = 2    # iteration limit reached
            elif feasible_res == 3:
                self.status = 3    # problem appears to be unbounded

            if self.status == 0:
                print"\nTimeStep {}: SAFE".format(i)
            elif self.status == 1:
                print"\nTimeStep {}: UNSAFE".format(i)
                feasible_alpha = feasible_res.x[0, 0]
                feasible_beta = feasible_res.x[0, 1]
                vector_u0 = dPde.init_vector.multiply(feasible_alpha)
                vector_b0 = dPde.vector_b.multiply(feasible_beta)
                self.unsafe_trace = dPde.get_trace(vector_b0, vector_u0, i)    # produce a trace lead dPde to unsafe region
            else:
                print"\nTimeStep{}: Error in checking safe/unsafe"
