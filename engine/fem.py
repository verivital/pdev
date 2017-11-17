'''
This module implements Finite Element Method
Dung Tran Nov/2017

Main references:
    1) An introduction to the Finite Element Method for Differential Equations, M.Asadzaded, 2010
    2) The Finite Element Method: Theory, Implementation and Applications, Mats G. Larson, Fredirik Bengzon
'''

from scipy.sparse import lil_matrix, csc_matrix, linalg, hstack, vstack
from scipy.integrate import quad
import numpy as np
import math


class Fem1D(object):
    'contains functions of finite element method for 1D PDEs'

    @staticmethod
    def mass_assembler(x):
        'compute mass matrix for 1D problem'

        # x is list of discretized mesh points for example x = [0 , 0.1, 0.2, .., 0.9, 1]
        assert isinstance(x, list)
        assert len(x) > 3, 'len(x) should >= 3'

        for i in xrange(0, len(x) - 2):
            assert isinstance(x[i], float), 'x[{}] should be float type'.format(i)
            assert isinstance(x[i + 1], float), 'x[{}] should be float type'.format(i + 1)
            assert x[i + 1] > x[i], 'x[i + 1] = {} should be > x[i] = {}'.format(x[i + 1], x[i])

        n = len(x) - 2    # number of discretized variables
        mass_matrix = lil_matrix((n, n), dtype=float)

        # filling mass_matrix

        for i in xrange(0, n):
            hi = x[i + 1] - x[i]
            hi_plus_1 = x[i + 2] - x[i + 1]

            mass_matrix[i, i] = hi / 3 + hi_plus_1 / 3
            if i + 1 <= n - 1:
                mass_matrix[i, i + 1] = hi_plus_1 / 6
            if i - 1 >= 0:
                mass_matrix[i, i - 1] = hi / 6

        return mass_matrix.tocsc()

    @staticmethod
    def stiff_assembler(x):
        'compute stiff matrix for 1D problem'

        # x is list of discretized mesh points for example x = [0 , 0.1, 0.2, .., 0.9, 1]
        assert isinstance(x, list)
        assert len(x) > 3, 'len(x) should >= 3'

        for i in xrange(0, len(x) - 2):
            assert isinstance(x[i], float), 'x[{}] should be float type'.format(i)
            assert isinstance(x[i + 1], float), 'x[{}] should be float type'.format(i + 1)
            assert x[i + 1] > x[i], 'x[i + 1] = {} should be > x[i] = {}'.format(x[i + 1], x[i])

        n = len(x) - 2    # number of discretized variables
        stiff_matrix = lil_matrix((n, n), dtype=float)

        # filling stiff_matrix

        for i in xrange(0, n):
            hi = x[i + 1] - x[i]
            hi_plus_1 = x[i + 2] - x[i + 1]

            stiff_matrix[i, i] = 1 / hi + 1 / hi_plus_1
            if i + 1 <= n - 1:
                stiff_matrix[i, i + 1] = -1 / hi_plus_1
            if i - 1 >= 0:
                stiff_matrix[i, i - 1] = -1 / hi

        return stiff_matrix.tocsc()

    @staticmethod
    def f_mul_phi(y, fc, pc):
        'return f*phi function for computing load vector'

        # assume f is polynomial function f = c0 + c1*y + c2*y^2 + ... + cm * y^m
        # phi is hat function defined on pc = [pc[0], pc[1], pc[2]]

        # todo: add more general function f which depends on t and x. using np.dblquad to calculate
        # double integration

        assert isinstance(fc, list)
        assert isinstance(pc, list)
        assert len(pc) == 3, 'len(pc) = {} != 3'.format(len(pc))
        for i in xrange(0, len(pc) - 1):
            assert pc[i] >= 0 and pc[i] < pc[i + 1], 'pc[{}] = {} should be larger than 0 and \
                smaller than pc[{}] = {}'.format(i, pc[i], i + 1, pc[i + 1])

        def func(y):
            'piecewise function'
            if y < pc[0]:
                return 0.0
            elif y >= pc[0] and y < pc[1]:
                return (1 / (pc[1] - pc[0])) * (sum((a * y**(i + 1) for i, a in enumerate(fc))) - pc[0] * sum((a * y**i for i, a in enumerate(fc))))

            elif y >= pc[1] and y < pc[2]:
                return (1 / (pc[2] - pc[1])) * (-sum((a * y**(i + 1) for i, a in enumerate(fc))) + pc[2] * sum((a * y**i for i, a in enumerate(fc))))

            elif y >= pc[2]:
                return 0.0

        return np.vectorize(func)

    @staticmethod
    def load_assembler(x, fc, f_dom):
        'compute load vector for 1D problem'

        # x is list of discretized mesh points for example x = [0 , 0.1, 0.2, .., 0.9, 1]
        # y is a variable that used to construct the f * phi function
        # fc is the input function, here we consider polynomial function in space
        # i.e. f = c0 + c1 * x + c2 * x^2 + .. + cm * x^m
        # input fc = [c0, c1, c2, ... cm]
        # f_dom defines the segment where the input function effect,
        # f_domain = [x1, x2] ,  0 <= x1 <= x2 <= x_max
        # return [b_i] = integral (f * phi_i dx)

        assert isinstance(x, list)
        assert isinstance(fc, list)
        assert isinstance(f_dom, list)

        assert len(x) > 3, 'len(x) should >= 3'
        assert len(fc) > 1, 'len(f) should >= 1'
        assert len(f_dom) == 2, 'len(f_domain) should be 2'
        assert (x[0] <= f_dom[0]) and (f_dom[0] <= f_dom[1]) and (f_dom[1] <= x[len(x) - 1]), 'inconsistent domain'

        for i in xrange(0, len(x) - 2):
            assert isinstance(x[i], float), 'x[{}] should be float type'.format(i)
            assert isinstance(x[i + 1], float), 'x[{}] should be float type'.format(i + 1)
            assert x[i + 1] > x[i], 'x[i + 1] = {} should be > x[i] = {}'.format(x[i + 1], x[i])

        n = len(x) - 2    # number of discretized variables

        b = lil_matrix((n, 1), dtype=float)

        y = []

        for i in xrange(0, n):
            pc = [x[i], x[i + 1], x[i + 2]]
            fphi = Fem1D.f_mul_phi(y, fc, pc)
            I = quad(fphi, f_dom[0], f_dom[1])
            b[i, 0] = I[0]

        return b.tocsc()

    @staticmethod
    def get_ode(mass_mat, stiff_mat, load_vec, time_step):
        'obtain discreted ODE model'

        # the discreted ODE model has the form of: U_n = A * U_(n-1) + b

        assert isinstance(mass_mat, csc_matrix)
        assert isinstance(stiff_mat, csc_matrix)
        assert isinstance(load_vec, csc_matrix)
        assert isinstance(time_step, float)
        assert (time_step > 0), 'time step k = {} should be >= 0'.format(time_step)

        matrix_a = linalg.inv((mass_mat + stiff_mat.multiply(time_step / 2))) * \
          (mass_mat - stiff_mat.multiply(time_step / 2))
        vector_b = linalg.inv(mass_mat + stiff_mat.multiply(time_step / 2)) * \
          load_vec.multiply(time_step)

        return matrix_a, vector_b


    @staticmethod
    def u0x_func(x, c):
        'return u(x, 0) = u_0(x) initial function at t = 0'

        # assumpe u_0(x) is a polynomial function defined by u_0(x) = c0 + c1 * x1 + ... + cn * x^n

        assert isinstance(c, list)
        assert len(c) >= 1, 'len(c) = {} should be >= 1'.format(len(c))

        def func(x):
            'function'

            return sum(a * x**i for i, a in enumerate(c))

        return func

    @staticmethod
    def get_init_cond(x, u0x_func):
        'get initial condition from initial condition function'

        # x is list of discreted mesh points, for example x = [0 , 0.1, 0.2, .., 0.9, 1]
        # u0x_func is defined above
        assert isinstance(x, list)
        assert len(x) >= 3, 'len(x) = {} should be >= 3'.format(len(x))

        n = len(x) - 2
        u0 = lil_matrix((n, 1), dtype=float)

        for i in xrange(0, n):
            v = x[i + 1]
            u0[i, 0] = u0x_func(v)

        return u0.tocsc()

    @staticmethod
    def get_trace(matrix_a, vector_b, vector_u0, step, num_steps):
        'produce a trace of the discreted ODE model'

        u_list = []
        times = np.linspace(0, step * num_steps, num_steps + 1)
        print "\n times = {}".format(times)

        n = len(times)

        for i in xrange(0, n):
            print "\ni={}".format(i)
            if i == 0:
                u_list.append(vector_u0)
            else:
                u_n_minus_1 = u_list[i - 1]
                u_n = matrix_a * u_n_minus_1 + vector_b
                u_list.append(u_n)

            print "\n t = {} -> \n U =: \n{}".format(i * step, u_list[i].todense())

        return u_list

    @staticmethod
    def plot_trace(trace, step):
        'plot trace of the discreted ODE model'

        assert isinstance(trace, list)
        n = len(trace)
        assert n >= 2, 'trace should have at least two points, currently it has {} points'.format(n)


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

    def plot_set(self):
        'plot set'
        pass

    def check_feasible(self):
        'check feasible of the set'

        assert self.matrix_c is not None and self.vector_d is not None, 'empty set to check'
        C = [1, 1]

class DPdeAutomaton(object):
    'discreted Pde automaton'

    def __init__(self):

        self.matrix_a = None
        self.vector_b = None
        self.init_vector = None
        self.perturbation = None
        self.unsafe_set = None

    def set_dynamics(self, matrix_a, vector_b, init_vector):
        'set dynamic of discreted pde automaton'

        assert isinstance(matrix_a, csc_matrix)
        assert len(matrix_a.shape) == 2
        assert isinstance(vector_b, csc_matrix)
        assert matrix_a.shape[0] == vector_b.shape[0], 'inconsistency between shapes of matrix_a and \
        vector_b'
        assert vector_b.shape[1] == 1, 'vector_b shape = {} # 1'.format(vector_b.shape[1])

        assert isinstance(init_vector, csc_matrix)
        assert matrix_a.shape[0] == init_vector.shape[0], 'matrix_a and init_vector are inconsistent'
        assert init_vector.shape[1] == 1

        self.matrix_a = matrix_a
        self.vector_b = vector_b
        self.init_vector = init_vector

    def set_perturbation(self, alpha_beta_range):
        'set pertupation on input function f and initial function u0(x)'

        # we consider pertubation on input function f, and initial condition function u_0(x)
        # actual initial condition = u_0(x) + epsilon2 * u_0(x) = alpha * u_0(x), a1 <= alpha <= b2
        # actual input function = f + epsilon1 * f = beta * f, a2 <= beta <= b2

        # the actual initial vector: = alpha * u0
        # the actual load vector: beta * b

        # pertupation defined by a general set C * [alpha beta]^T <= d

        assert isinstance(alpha_beta_range, np.ndarray)
        assert alpha_beta_range.shape == (2, 2), 'alpha_beta_range shape is incorrect'
        assert alpha_beta_range[0, 0] <= alpha_beta_range[1, 0], 'incorrect alpha range'
        assert alpha_beta_range[0, 1] <= alpha_beta_range[1, 1], 'incorrect beta range'

        alpha_beta_matrix = lil_matrix((4, 2), dtype=float)
        alpha_beta_matrix[0, 0] = -1
        alpha_beta_matrix[1, 0] = 1
        alpha_beta_matrix[2, 1] = -1
        alpha_beta_matrix[3, 1] = 1

        alpha_beta_vector = lil_matrix((4, 1), dtype=float)
        alpha_beta_vector[0, 0] = alpha_beta_range[0, 0]
        alpha_beta_vector[1, 0] = alpha_beta_range[0, 1]
        alpha_beta_vector[2, 0] = alpha_beta_range[1, 0]
        alpha_beta_vector[3, 0] = alpha_beta_range[1, 1]

        per_set = GeneralSet()
        per_set.set_constraints(alpha_beta_matrix.tocsc(), alpha_beta_vector.tocsc())

        self.perturbation = per_set

        return self.perturbation

    def set_unsafe_set(self, direction_matrix, unsafe_vector):
        'define the unsafe set of the automaton'

        # unsafe Set defined by direction_matrix * U <= unsafe_vector
        assert isinstance(direction_matrix, csc_matrix)
        assert isinstance(unsafe_vector, csc_matrix)
        assert direction_matrix.shape[0] != unsafe_vector.shape[0], 'inconsistency, \
             direction_matrix.shape[0] = {} != unsafe_vector.shape[0] = {}'\
             .format(direction_matrix.shape[0], unsafe_vector.shape[0])

        self.unsafe_set = GeneralSet()
        self.unsafe_set.set_constraints(direction_matrix, unsafe_vector)

        return self.unsafe_set


class DReachSet(object):
    'Reachable set representation of discreted PDE'

    # the reachable set of discreted PDE has the form of: U_n = alpha * V_n + beta * l_n

    def __init__(self):

        self.perturbation = None
        self.Vn = None
        self.ln = None

    def set_reach_set(self, perturbation_set, vector_Vn, vector_ln):
        'set a specific set'

        assert isinstance(perturbation_set, GeneralSet)
        self.perturbation = perturbation_set

        assert isinstance(vector_Vn, csc_matrix)
        assert isinstance(vector_ln, csc_matrix)
        assert vector_Vn.shape[0] == vector_ln.shape[0], 'inconsistent between Vn and ln vectors'
        assert vector_Vn.shape[1] == vector_ln.shape[1] == 1, 'wrong dimensions for vector Vn and ln'

        self.Vn = vector_Vn
        self.ln = vector_ln

    def plotdReachSet(self):
        'plot dReach set'

        pass


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


if __name__ == '__main__':

    ##################################################
    # test Fem1D class
    FEM = Fem1D()
    mesh_points = [0.0, 0.5, 1.0, 1.5, 2.0]    # generate mesh points

    mass_matrix = FEM.mass_assembler(mesh_points)    # compute mass matrix M
    stiff_matrix = FEM.stiff_assembler(mesh_points)    # compute stiff matrix S

    fc = [1.0, 0.0, 2.0]    # define input function f
    fdom = [0.5, 1.0]    # domain of input function
    load_vector = FEM.load_assembler(mesh_points, fc, fdom)    # compute load vector

    print "\nmass matrix = \n{}".format(mass_matrix.todense())
    print "\nstiff matrix = \n{}".format(stiff_matrix.todense())
    print "\nload vector = \n{}".format(load_vector.todense())

    step = 0.1    # time step of FEM
    A, b = FEM.get_ode(mass_matrix, stiff_matrix, load_vector, step)    # get the discreted ODE model

    print "\nA = {} \nb = {}".format(A.todense(), b.todense())
    print "\ntype of A is {}, type of b is {}".format(type(A), type(b))

    y = []
    c = [1, 2]    # parameters for initial function u0(x)
    u0_func = FEM.u0x_func(y, c)    # define initial function u0(x)
    u0 = FEM.get_init_cond(mesh_points, u0_func)    # compute initial conditions
    print"\nu0 = {}".format(u0.todense())    # initial condition vector u0

    u = FEM.get_trace(A, b, u0, step=0.1, num_steps=4)    # get trace with initial vector u0

    #########################################################
    # test DPdeAutomation object class
    dPde = DPdeAutomaton()
    dPde.set_dynamics(A, b, u0)    # set dynamic of discreted PDE
    alpha_beta_range = np.array([[0, 1], [1, 1]])    # set perturbation range
    dPde.set_perturbation(alpha_beta_range)

    ########################################################
    # test Dverifier class

    verifier = DVerifier()
    toTimeStep = 10
    reachable_set = verifier.compute_reach_set(dPde, toTimeStep)    # compute reachable set

    direction_matrix = lil_matrix((1, A.shape[0]), dtype=float)
    direction_matrix[0, math.ceil(A.shape[0] / 2)] = 1
    unsafe_vector = lil_matrix((1, 1), dtype=float)
    unsafe_vector[0, 0] = -1

    dPde.set_unsafe_set(direction_matrix.tocsc(), unsafe_vector.tocsc())    # set unsafe set
