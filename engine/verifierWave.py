'''
This module implements continuous/discreted verifier for PDE automaton
Dung Tran: Nov/2017
Tianshu Bao: Jun/2018
'''

from scipy.sparse import csc_matrix, lil_matrix
from scipy.optimize import minimize
from engine.pde_automaton import DPdeAutomaton
from engine.set import DReachSet
from engine.interpolation_wave import Interpolation
from engine.femwave import Fem1Dw
from engine.functions import Functions
from engine.specification import SafetySpecification
from engine.plot import Plot
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import Function

class ReachSetAssembler(object):
    'compute all necessary reachable sets'

    def __init__(self):
        self.u_dreachset = []
        self.err_dreachset = []
        self.bloated_dreachset = []

    @staticmethod
    def get_cur_u_dreachset(matrix_a, prev_u, cur_b_vec):
        'compute the current approximate discrete reachable set of u, cur_b_vec = g_n = [0,0,0...0, (b_n + b_(n-1))k/2]^T'

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
    def get_cur_err_dreachset(matrix_a, prev_e, cur_b_vec):
        'compute the current approximate discreate reachable set of error e'

        assert isinstance(matrix_a, csc_matrix)
        assert isinstance(prev_e, DReachSet)
        assert isinstance(cur_b_vec, csc_matrix)
        assert matrix_a.shape[0] == cur_b_vec.shape[0] == prev_e.Vn.shape[0] == prev_e.ln.shape[0], 'inconsistent'

        cur_err_dreachset = DReachSet()
        cur_err_dreachset.Vn = matrix_a * prev_e.Vn
        cur_err_dreachset.ln = matrix_a * prev_e.ln + cur_b_vec
        cur_err_dreachset.alpha_range = prev_e.alpha_range
        cur_err_dreachset.beta_range = prev_e.beta_range

        return cur_err_dreachset

    @staticmethod
    def get_cur_be(prev_u, curr_u, dPde, cur_time):
        'compute b[n], e[n] = A * e[n-1] + be[n]'

        assert isinstance(dPde, DPdeAutomaton)

        cur_b_vec = Fem1Dw.load_assembler_err(dPde.xlist, dPde.f_xdom, dPde.time_step, cur_time, prev_u, curr_u)
	
        return cur_b_vec

    @staticmethod
    def get_dreachset(dPde, toTimeStep):
        'compute approximate discrete reachable set of u and e and the bloated u + e'

        assert isinstance(dPde, DPdeAutomaton)
        assert isinstance(toTimeStep, int) and toTimeStep >= 0

        u_dreachset_list = []
        err_dreachset_list = []
        bloated_dreachset_list = []

        for cur_time in xrange(0, toTimeStep + 1):
            u_dreachset = DReachSet()
            if cur_time == 0:
                u_Vn = dPde.init_vector
                u_ln = csc_matrix((dPde.init_vector.shape[0], 1), dtype=float)
                u_dreachset.set_reach_set(dPde.alpha_range, dPde.beta_range, u_Vn, u_ln)

            else:
                cur_g_vec = Fem1Dw.load_assembler(dPde.xlist, dPde.f_xdom, dPde.time_step, cur_time)
                u_dreachset = ReachSetAssembler.get_cur_u_dreachset(dPde.matrix_a, u_dreachset_list[cur_time - 1], cur_g_vec)																	
	    u_dreachset_list.append(u_dreachset)									
	 

        for cur_time in xrange(0, toTimeStep + 1):
	    err_dreachset = DReachSet()
            if cur_time == 0:
		err_Vn = csc_matrix((dPde.init_vector.shape[0], 1), dtype=float)
		err_ln = csc_matrix((dPde.init_vector.shape[0], 1), dtype=float)
		err_dreachset.set_reach_set(dPde.alpha_range, dPde.beta_range, err_Vn, err_ln)

            else:
		cur_be = ReachSetAssembler.get_cur_be(u_dreachset_list[cur_time - 1], u_dreachset, dPde, cur_time)
		err_dreachset = ReachSetAssembler.get_cur_err_dreachset(dPde.matrix_a, err_dreachset_list[cur_time - 1], cur_be)																
	    
	    err_dreachset_list.append(err_dreachset)
	    bloated_dreachset = DReachSet()
            bloated_dreachset.set_reach_set(dPde.alpha_range, dPde.beta_range, u_dreachset_list[cur_time].Vn + err_dreachset.Vn, u_dreachset_list[cur_time].ln + err_dreachset.ln)
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
            u_set_list.append(Interpolation.increm_interpolation(dPde.time_step, i, u_setinspace_list[i - 1], u_setinspace_list[i], u_dset[i - 1], u_dset[i]))
            e_set_list.append(Interpolation.increm_interpolation(dPde.time_step, i, e_setinspace_list[i - 1], e_setinspace_list[i], e_dset[i - 1], e_dset[i]))
            bl_set_list.append(Interpolation.increm_interpolation(dPde.time_step, i, bl_setinspace_list[i - 1], bl_setinspace_list[i], bl_dset[i - 1], bl_dset[i]))

        return u_setinspace_list, e_setinspace_list, bl_setinspace_list, u_set_list, e_set_list, bl_set_list

class VerificationResult(object):
    'Result object for verification'

    def __init__(self):
        self.status = None                # status Safe/Unsafe
        self.unsafe_time_point = None     # time that system reach unsafe state
        self.unsafe_x_point = None        # position that system reach unsafe state
        self.unsafe_u_point = None        # value of u(x,t) at unsafe state
        self.unsafe_trace_funcs = []    # u(x,t) at unsafe_x_point is a list of functions of t

        self.step = None    # time step
        self.safety_specification = None    # used for plotting the result

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

        time_list.append(n * self.step)
        func = self.unsafe_trace_funcs[n - 1]
        u_list.append(func([n * self.step]))

        return (time_list, u_list)

    def get_unsafe_point(self):
        'return the unsafe point'

        return (self.unsafe_time_point, self.unsafe_x_point, self.unsafe_u_point)

    def get_specification(self):
        'return safety specification'

        return self.safety_specification


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
        self.result.safety_specification = safety_specification
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
                x_range_list.append((xlist[end_point - 1], x2))
            elif start_point < i < end_point - 1:
                x_range_list.append((xlist[i], xlist[i + 1]))

        # decompose T1, T2 into list of t_range
        t_range_list = []
        for j in xrange(start_time_step, end_time_step + 1):
            if j == start_time_step:
                t_range_list.append((T1, (start_time_step + 1) * step))
            elif j == end_time_step:
                t_range_list.append(((end_time_step - 1) * step, T2))
            elif start_time_step < j < end_time_step:
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
		
if __name__ == '__main__':
	FEM = Fem1Dw()
	mesh_points = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]    # generate mesh points
	step = 0.1    # time step of FEM
	x_dom = [0.0, 0.01]    # domain of input function
	xlist = mesh_points

	dPde = Fem1Dw().get_dPde_automaton(mesh_points, x_dom, step)##wave FEM		
	dPde.set_perturbation((0.99,1.01),(0.99,1.01))
		
	toTimeStep = 10
	rsa = ReachSetAssembler()
	#u_dset = rsa.get_dreachset(dPde, toTimeStep)##test new get_dreachset method
	u_setinspace_list, e_setinspace_list, bl_setinspace_list, u_set_list, e_set_list, bl_set_list = rsa.get_interpolationset(dPde, toTimeStep)
	

	pl = Plot()
	fig2, ax = plt.subplots()
	box = bl_setinspace_list[8].get_2D_boxes((0.99,1.01),(0.99,1.01))
	ax = pl.plot_boxes(ax, box, facecolor='cyan', edgecolor='cyan')
	fig2.axes.append(ax)
	plt.show()

	bl_boxes_3d = []
    	for i in xrange(0, len(bl_set_list)):
        	box3d = bl_set_list[i].get_3D_boxes((0.99,1.01), (0.99,1.01))	
        	bl_boxes_3d.append(box3d)

    	fig2 = plt.figure()
    	ax2 = fig2.add_subplot(111, projection='3d')
    	pl2 = Plot()

    	ax2 = pl2.plot_interpolationset(ax2, bl_boxes_3d, facecolor='c', linewidth=0.5, edgecolor='r')
    	ax2.set_xlim(0, 5.0)
    	ax2.set_ylim(0, 2.0)
    	ax2.set_zlim(-1.5, 1.5)
    	ax2.tick_params(axis='z', labelsize=10)
    	ax2.tick_params(axis='x', labelsize=10)
    	ax2.tick_params(axis='y', labelsize=10)
    	ax2.set_xlabel('$x$', fontsize=10)
    	ax2.set_ylabel('$t$', fontsize=10)
    	ax2.set_zlabel(r'$e_h(x,t)$', fontsize=10)
    	fig2.suptitle('3-Dimensional Reachable Set', fontsize=15)
    	fig2.savefig('reachset_3D.pdf')

	plt.show()
	
						
