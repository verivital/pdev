'''
This module implements Finite Element Method
Dung Tran Nov/2017
Tianshu Bao Mar/2018

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
    def load_assembler(x, x_dom, time_step, current_step):##we update b to 2n here
        'compute load vector for 1D problem'

        # x is list of discretized mesh points for example x = [0 , 0.1, 0.2, .., 0.9, 1]
        # y is a variable that used to construct the f * phi function
        # the input function is defined in engine.functions.Functions class
        # x_dom = [x1, x2] defines the domain where the input function effect,
        # t_dom = (0 <= t<= time_step))
        # return [b_i] = integral (f * phi_i dx dt), ((x1 <= x <= x2), (t[n-1] <=
        # t<= t[n]))

        assert isinstance(x, list)
        assert isinstance(x_dom, list)

        assert len(x) > 3, 'len(x) should >= 3'
        assert len(x_dom) == 2, 'len(f_domain) should be 2'
        assert (x[0] <= x_dom[0]) and (x_dom[0] <= x_dom[1]) and (
            x_dom[1] <= x[len(x) - 1]), 'inconsistent domain'

        for i in xrange(0, len(x) - 2):
            assert isinstance(
                x[i], float), 'x[{}] should be float type'.format(i)
            assert isinstance(
                x[i + 1], float), 'x[{}] should be float type'.format(i + 1)
            assert x[i +
                     1] > x[i], 'x[i + 1] = {} should be > x[i] = {}'.format(x[i + 1], x[i])

        assert time_step > 0, 'invalid time_step'
        assert isinstance(current_step, int)

        n = len(x) - 2    # number of discretized variables, remove head and tail, in this case is 0 and 1

        #b = lil_matrix((n, 1), dtype=float)
        b = lil_matrix((2*n, 1), dtype=float)
	
        for i in xrange(0, n):
            seg_x = [x[i], x[i + 1], x[i + 2]]
            if current_step >= 1:# when current == 0 ???, b = 0
                b[n + i, 0] = Functions.integrate_input_func_mul_phi(
                    seg_x, x_dom, [float(current_step - 1) * time_step, time_step * current_step])

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

	eta = Function('func')
	eta = 1/10.0*cos(y/10.0)
	f = lambdify(y, eta)

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

        mass_mat = Fem1Dw.mass_assembler(x)
        stiff_mat = Fem1Dw.stiff_assembler(x)
        load_vec = Fem1Dw.load_assembler(x, x_dom, time_step, 0)
        init_vector = Fem1Dw.get_init_cond(x)

        #inv_b_matrix = linalg.inv(mass_mat + stiff_mat.multiply(time_step / 2))
	'matrix before Un'
	temp_matrix_a1 = np.concatenate((mass_mat.transpose().todense(), -mass_mat.multiply(time_step /2).transpose().todense()), axis=0)
	temp_matrix_a1 = temp_matrix_a1.transpose()

	temp_matrix_b1 = np.concatenate((stiff_mat.multiply(time_step /2).transpose().todense(), mass_mat.transpose().todense()), axis=0)
	temp_matrix_b1 = temp_matrix_b1.transpose()

	final_matrix = np.concatenate((temp_matrix_a1, temp_matrix_b1), axis=0)
	#print "\n resulting A for wave eq:\n{}".format(final_matrix)
	
	final_matrix = sparse.csc_matrix(final_matrix)
        inv_b_matrix = linalg.inv(final_matrix)

	'matrix before Un-1'
	temp_matrix_a2 = np.concatenate((mass_mat.transpose().todense(), mass_mat.multiply(time_step /2).transpose().todense()), axis=0)
	temp_matrix_a2 = temp_matrix_a2.transpose()

	temp_matrix_b2 = np.concatenate((-stiff_mat.multiply(time_step /2).transpose().todense(), mass_mat.transpose().todense()), axis=0)
	temp_matrix_b2 = temp_matrix_b2.transpose()

	final_matrix2 = np.concatenate((temp_matrix_a2, temp_matrix_b2), axis=0)
	#print "\n resulting A2 for wave eq:\n{}".format(final_matrix2)
	final_matrix2 = sparse.csc_matrix(final_matrix2)

	#print "\n resulting loadvec for wave eq:\n{}".format(load_vec.todense())
	
	matrix_a = inv_b_matrix  * final_matrix2
	n = load_vec.shape[0]
	
	#print n

	#temp_loadvec = np.concatenate((np.zeros((n,1)), load_vec.todense()), axis=0)
	#print "\n resulting loadvec for wave eq:\n{}".format(temp_loadvec)
	#temp_loadvec = sparse.csc_matrix(temp_loadvec)
	vector_b = inv_b_matrix * load_vec
	print "\n vector_b:\n{}".format(vector_b.todense())
 
        #matrix_a = inv_b_matrix * (mass_mat - stiff_mat.multiply(time_step / 2))
        #vector_b = inv_b_matrix * load_vec
        
	dPde = DPdeAutomaton()
        dPde.set_matrix_a(matrix_a)
        dPde.set_vector_b(vector_b)
        dPde.set_inv_b_matrix(inv_b_matrix)
        dPde.set_fxdom(x_dom)
        dPde.set_init_condition(init_vector)#need to reset
        dPde.set_xlist_time_step(x, time_step)

        return dPde

if __name__ == '__main__':
    xlist = [0.0, 0.5, 1.0, 1.5, 2.0]
    Fem = Fem1D()
    M = Fem.mass_assembler(xlist)
    print "\nmass matrix: \n{}".format(M.todense())
    S = Fem.stiff_assembler(xlist)
    print "\nstiff matrix:\n{}".format(S.todense())

    k = 0.1
    cur_step = 1
    b = Fem.load_assembler(xlist, [0.1, 0.2], k, cur_step)
    print "\nload vector:\n{}".format(b.todense())

    init_vec = Fem.get_init_cond(xlist)
    print "\ninit vector :\n{}".format(init_vec.todense())

    pde_auto = Fem.get_dPde_automaton(xlist, [0.1, 0.2], k)
