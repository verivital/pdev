'''
This module implements Finite Element Method
Dung Tran Nov/2017
Tianshu Bao: Jun/2018

Main references:
    1) An introduction to the Finite Element Method for Differential Equations, M.Asadzaded, 2010
    2) The Finite Element Method: Theory, Implementation and Applications, Mats G. Larson, Fredirik Bengzon
'''

from scipy.sparse import lil_matrix, linalg
from sympy import Function, cos, lambdify
from engine.pde_automaton import DPdeAutomaton
from engine.functions import Functions
import numpy as np
from scipy import sparse
from sympy.abc import y

class Fem1Dw(object):
    'contains functions of finite element method for 1D PDEs'

    @staticmethod
    def mass_assembler(x):
        'compute mass matrix for 1D problem'

        # x is list of discretized mesh points for example x = [0 , 0.1, 0.2,
        # .., 0.9, 1]
        assert isinstance(x, list)
        assert len(x) > 3, 'len(x) should >= 3'

        for i in xrange(0, len(x) - 2):
            assert isinstance(
                x[i], float), 'x[{}] should be float type'.format(i)
            assert isinstance(
                x[i + 1], float), 'x[{}] should be float type'.format(i + 1)
            assert x[i +
                     1] > x[i], 'x[i + 1] = {} should be > x[i] = {}'.format(x[i +
                                                                               1], x[i])

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

        # x is list of discretized mesh points for example x = [0 , 0.1, 0.2,
        # .., 0.9, 1]
        assert isinstance(x, list)
        assert len(x) > 3, 'len(x) should >= 3'

        for i in xrange(0, len(x) - 2):
            assert isinstance(
                x[i], float), 'x[{}] should be float type'.format(i)
            assert isinstance(
                x[i + 1], float), 'x[{}] should be float type'.format(i + 1)
            assert x[i +
                     1] > x[i], 'x[i + 1] = {} should be > x[i] = {}'.format(x[i +
                                                                               1], x[i])

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
    def load_assembler(x, x_dom, time_step, current_step):
        'compute load vector for 1D problem'

        # x is list of discretized mesh points for example x = [0 , 0.1, 0.2, .., 0.9, 1]
        # y is a variable that used to construct the f * phi function
        # the input function is defined in engine.functions.Functions class
        # x_dom = [x1, x2] defines the domain where the input function effect,
        # t_dom = (0 <= t<= time_step))
        # return [b_i] = integral (f * phi_i dx), (x1 <= x <= x2))

        assert isinstance(x, list)
        assert isinstance(x_dom, list)

        assert len(x) > 3, 'len(x) should >= 3'
        assert len(x_dom) == 2, 'len(f_domain) should be 2'
        assert (x[0] <= x_dom[0]) and (x_dom[0] <= x_dom[1]) and (
            x_dom[1] <= x[len(x) - 1]), 'inconsistent domain'
	assert current_step >= 1, 'current_step < 1'

        for i in xrange(0, len(x) - 2):
            assert isinstance(
                x[i], float), 'x[{}] should be float type'.format(i)
            assert isinstance(
                x[i + 1], float), 'x[{}] should be float type'.format(i + 1)
            assert x[i +
                     1] > x[i], 'x[i + 1] = {} should be > x[i] = {}'.format(x[i + 1], x[i])

        assert time_step > 0, 'invalid time_step'
        assert isinstance(current_step, int)

        n = len(x) - 2    # number of discretized variables

        b = lil_matrix((2*n, 1), dtype=float)
	
        for i in xrange(0, n):	#we don't intergrate among t
            seg_x = [x[i], x[i + 1], x[i + 2]]
            b[n + i, 0] = Functions.integrate_input_func_mul_phi_in_space(seg_x, x_dom, time_step * current_step)
	    b[n + i, 0] = b[n + i, 0] + Functions.integrate_input_func_mul_phi_in_space(seg_x, x_dom, time_step * (current_step - 1))
	    b[n + i, 0] = b[n + i, 0] * time_step/2

        return b.tocsc()

    @staticmethod
    def load_assembler_err(x, x_dom, time_step, current_step, prev_u, cur_u):
        'compute load vector for 1D problem, we added u double dots on right hand side'

        # x is list of discretized mesh points for example x = [0 , 0.1, 0.2, .., 0.9, 1]
        # y is a variable that used to construct the f * phi function
        # the input function is defined in engine.functions.Functions class
        # x_dom = [x1, x2] defines the domain where the input function effect,
        # t_dom = (0 <= t<= time_step))
        # return [b_i] = integral (f * phi_i dx), (x1 <= x <= x2))

        assert isinstance(x, list)
        assert isinstance(x_dom, list)

        assert len(x) > 3, 'len(x) should >= 3'
        assert len(x_dom) == 2, 'len(f_domain) should be 2'
        assert (x[0] <= x_dom[0]) and (x_dom[0] <= x_dom[1]) and (
            x_dom[1] <= x[len(x) - 1]), 'inconsistent domain'
	assert current_step >= 1, 'current_step < 1'

        for i in xrange(0, len(x) - 2):
            assert isinstance(
                x[i], float), 'x[{}] should be float type'.format(i)
            assert isinstance(
                x[i + 1], float), 'x[{}] should be float type'.format(i + 1)
            assert x[i +
                     1] > x[i], 'x[i + 1] = {} should be > x[i] = {}'.format(x[i + 1], x[i])

        assert time_step > 0, 'invalid time_step'
        assert isinstance(current_step, int)

        n = len(x) - 2    # number of discretized variables
        b = lil_matrix((2*n, 1), dtype=float)

        b_t_curr = lil_matrix((n, 1), dtype=float)

        for i in xrange(0, n):
            seg_x = [x[i], x[i + 1], x[i + 2]]
            b_t_curr[i, 0] = Functions.integrate_input_func_mul_phi_in_space(seg_x, x_dom, time_step * current_step)

	s = Fem1Dw.stiff_assembler(x)

	u_value_curr = cur_u.Vn + cur_u.ln
	
	u_value_curr = u_value_curr[0 : n]

	M_inv = linalg.inv(Fem1Dw.mass_assembler(x))
	
	sec_deri_cur = M_inv * (b_t_curr - s * u_value_curr)

	b_t_prev = lil_matrix((n, 1), dtype=float)

        for i in xrange(0, n):
            seg_x = [x[i], x[i + 1], x[i + 2]]
            b_t_prev[i, 0] = Functions.integrate_input_func_mul_phi_in_space(seg_x, x_dom, time_step * (current_step - 1))

	u_value_prev = prev_u.Vn + prev_u.ln

	u_value_prev = u_value_prev[0 : n]
	
	sec_deri_prev = M_inv * (b_t_prev - s * u_value_prev)



        for i in xrange(0, n):	#we don't intergrate among t

            seg_x = [x[i], x[i + 1], x[i + 2]]

	    b[n + i, 0] = Functions.integrate_input_func_mul_phi_in_space(seg_x, x_dom, time_step * current_step) - sec_deri_cur[i, 0] * 4/3 * time_step  
	    b[n + i, 0] = b[n + i, 0] + Functions.integrate_input_func_mul_phi_in_space(seg_x, x_dom, time_step * (current_step - 1)) - sec_deri_prev[i, 0] * 4/3 * time_step  
	    b[n + i, 0] = b[n + i, 0] * time_step/2	    

        return b.tocsc()



    @staticmethod
    def get_init_cond(x):
        'get initial condition from initial condition function'

        # x is list of discreted mesh points, for example x = [0 , 0.1, 0.2,
        # .., 0.9, 1]
        assert isinstance(x, list)
        assert len(x) >= 3, 'len(x) = {} should be >= 3'.format(len(x))

        n = len(x) - 2
        u0 = lil_matrix((2*n, 1), dtype=float)
        _, init_func = Functions.init_func()

	source_f = Function('func')
	source_f = cos(y)
	f = lambdify(y, source_f)

        for i in xrange(0, n):
            v = x[i + 1]
            u0[i, 0] = init_func(v)


	for i in xrange(0, n):
	    v = x[i + 1]
	    u0[i + n, 0] = f(v)

        return u0.tocsc()

    @staticmethod
    def get_dPde_automaton(x, x_dom, time_step):
        'initialize discreted Pde automaton'
	'A1Un = A2Un-1 + b'
        mass_mat = Fem1Dw.mass_assembler(x)
        stiff_mat = Fem1Dw.stiff_assembler(x)
        load_vec = Fem1Dw.load_assembler(x, x_dom, time_step, 1)
        init_vector = Fem1Dw.get_init_cond(x)

	'matrix before Un, A1'
	temp_matrix_a1 = np.concatenate((mass_mat.transpose().todense(), -mass_mat.multiply(time_step /2).transpose().todense()), axis=0)
	temp_matrix_a1 = temp_matrix_a1.transpose()

	temp_matrix_b1 = np.concatenate((stiff_mat.multiply(time_step /2).transpose().todense(), mass_mat.transpose().todense()), axis=0)
	temp_matrix_b1 = temp_matrix_b1.transpose()

	final_matrix_A1 = np.concatenate((temp_matrix_a1, temp_matrix_b1), axis=0)
	
	final_matrix_A1 = sparse.csc_matrix(final_matrix_A1)
        inv_A1_matrix = linalg.inv(final_matrix_A1)

	'matrix before Un-1, A2'
	temp_matrix_a2 = np.concatenate((mass_mat.transpose().todense(), mass_mat.multiply(time_step /2).transpose().todense()), axis=0)
	temp_matrix_a2 = temp_matrix_a2.transpose()

	temp_matrix_b2 = np.concatenate((-stiff_mat.multiply(time_step /2).transpose().todense(), mass_mat.transpose().todense()), axis=0)
	temp_matrix_b2 = temp_matrix_b2.transpose()

	final_matrix_A2 = np.concatenate((temp_matrix_a2, temp_matrix_b2), axis=0)
	final_matrix_A2 = sparse.csc_matrix(final_matrix_A2)


	'construct A1^{-1}A2'
	matrix_a = inv_A1_matrix * final_matrix_A2
	n = load_vec.shape[0]
	
	'construct A1^{-1}b'
	vector_b = inv_A1_matrix * load_vec
        
	dPde = DPdeAutomaton()
        dPde.set_matrix_a(matrix_a)
        dPde.set_vector_b(vector_b)
        dPde.set_inv_b_matrix(inv_A1_matrix)
        dPde.set_fxdom(x_dom)
        dPde.set_init_condition(init_vector)
        dPde.set_xlist_time_step_w(x, time_step)

        return dPde

if __name__ == '__main__':
    xlist = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    Fem = Fem1Dw()
    M = Fem.mass_assembler(xlist)
    #print "\nmass matrix: \n{}".format(M.todense())
    S = Fem.stiff_assembler(xlist)
    #print "\nstiff matrix:\n{}".format(S.todense())

    timestep_size = 0.1
    cur_step = 1
    b = Fem.load_assembler(xlist, [0.1, 1.1], timestep_size, cur_step)
    #print "\nload vector:\n{}".format(b.todense())

    init_vec = Fem.get_init_cond(xlist)
    #print "\ninit vector :\n{}".format(init_vec.todense())

    pde_auto = Fem.get_dPde_automaton(xlist, [0.1, 1.1], timestep_size)
    #print pde_auto.vector_b.todense()

