'''
This module implements continuous/discreted verifier for PDE automaton
Dung Tran: Nov/2017
'''

from scipy.sparse import csc_matrix, hstack, vstack
from engine.pde_automaton import DPdeAutomaton
from engine.set import DReachSet, GeneralSet


class DVerifier(object):
    'verifier for discreted pde automaton'

    # verify the safety of discreted pde automaton from step 0 to step N
    # if unsafe region is reached, produce a trace

    def __init__(self):

        self.status = None    # safe / unsafe
        self.next_step = None
        self.current_V = None    # Vn = A^n * Vn-1, V0 = U0
        self.current_l = None    # ln = Sigma_[i=0 to i = n-1] (A^i * b)
        self.to_current_step_set = []     # include all reach sets from 0 to current step
        self.current_constraints = None    # current constraint to check safety
        self.unsafe_trace = []    # trace for unsafe case

    def compute_reach_set(self, dPde, toTimeStep):
        'compute reach set of discreted PDE to the toTimeStep'

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
                self.current_l = csc_matrix((dPde.init_vector.shape[0], 1), dtype=float)
                current_set.set_reach_set(dPde.perturbation, self.current_V, self.current_l)
                self.to_current_step_set.append(current_set)
            else:
                self.current_V = dPde.matrix_a * self.current_V
                self.current_l = dPde.vector_b + dPde.matrix_a * self.current_l
                current_set.set_reach_set(dPde.perturbation, self.current_V, self.current_l)
                self.to_current_step_set.append(current_set)

        return self.to_current_step_set

    def on_fly_check(self, dPde, toTimeStep):
        'On-the-fly safety checking'

        assert dPde.unsafe_set is not None, 'specify unsafe set first'

        direct_matrix = dPde.unsafe_set.matrix_c
        unsafe_vector = dPde.unsafe_set.vector_d
        self.current_V = []
        self.current_l = []
        per_set = dPde.perturbation
        per_matrix = per_set.matrix_c
        per_vector = per_set.vector_d


        for i in xrange(0, toTimeStep + 1):
            if i == 0:
                self.current_V = dPde.init_vector
                self.current_l = csc_matrix((dPde.init_vector.shape[0], 1), dtype=float)
                inDirection_Current_V = direct_matrix * self.current_V
                inDirection_Current_l = direct_matrix * self.current_l

                C1 = hstack([inDirection_Current_V, inDirection_Current_l])
                constraint_matrix = vstack([C1, per_matrix])
                constraint_vector = vstack([unsafe_vector, per_vector])

                current_constraints = GeneralSet()
                current_constraints.set_constraints(constraint_matrix, constraint_vector)

                # check feasible
