'''
This module implements general set class and DReachSet class and their methods
Dung Tran: Nov/2017
'''

from scipy.sparse import lil_matrix, csc_matrix, eye, vstack, hstack
from scipy.optimize import linprog
from matplotlib.patches import Rectangle
from matplotlib.axes import Axes


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

    def check_feasible(self):
        'check feasible of the set'

        # todo: implements check feasible using glpk package for large sparse constraints
        # another option can be using commercial Gurobi solver

        assert self.matrix_c is not None and self.vector_d is not None, 'empty set to check'
        min_vector = [1, 1]
        alpha_bounds = (None, None)
        beta_bounds = (None, None)
        res = linprog(
            min_vector,
            A_ub=self.matrix_c.todense(),
            b_ub=self.vector_d.todense(),
            bounds=(
                alpha_bounds,
                beta_bounds),
            options={
                "disp": False})

        return res


class RectangleSet(object):
    'Rectangle Set'

    def __init__(self):
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None

    def set_bounds(self, xmin, xmax, ymin, ymax):
        'specify a rectangle'

        assert isinstance(xmin, float)
        assert isinstance(xmax, float)
        assert isinstance(ymin, float)
        assert isinstance(ymax, float)

        assert xmin < xmax, 'invalid set, xmin = {} is not < than xmax = {}'.format(
            xmin, xmax)
        assert ymin < ymax, 'invalid set, ymin = {} is not < than ymax = {}'.format(
            ymin, ymax)

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax


class DReachSet(object):
    'Reachable set representation of discreted PDE'

    # the reachable set of discreted PDE has the form of: U_n = alpha * V_n +
    # beta * l_n

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

    def range_reach_set(self, direction_matrix):
        'compute range of reach set in a specific direction, i.e.,  x_min[i] <= x[i] <= x_max[i]'

        if self.Vn is None and self.ln is None:
            raise ValueError('empty set')
        elif self.perturbation is None:
            raise ValueError('specify perturbation to plot Reachable Set')

        assert isinstance(direction_matrix, csc_matrix)
        assert direction_matrix.shape[1] == self.Vn.shape[0] == self.ln.shape[0], 'inconsistency between \
            direction matrix and vector Vn and vector ln'

        inDirection_vector_Vn = direction_matrix * self.Vn
        inDirection_vector_ln = direction_matrix * self.ln

        num_var = direction_matrix.shape[0]
        indentity_mat = eye(num_var, dtype=float)
        Aeq_mat = hstack(
            [inDirection_vector_Vn, inDirection_vector_ln, -indentity_mat])
        Beq_mat = csc_matrix((num_var + 2, 1), dtype=float)

        matrix_c = self.perturbation.matrix_c
        vector_d = self.perturbation.vector_d
        zero_mat = csc_matrix((num_var, num_var), dtype=float)
        zero_vec = csc_matrix((num_var, 1), dtype=float)
        Aub_mat = hstack([matrix_c, zero_mat])
        Bub_mat = vstack([vector_d, zero_vec])

        min_range = []    # = [xmin[1], xmin[2], ...xmin[num_var]]
        max_range = []    # = [xmax[1], xmax[2], ...xmax[num_var]]

        # compute min range
        for i in xrange(0, num_var):
            vector_c = lil_matrix((1, num_var + 2), dtype=float)
            vector_c[0, i + 2] = 1
            res = linprog(
                vector_c.todense(),
                A_ub=Aub_mat.todense(),
                b_ub=Bub_mat.todense(),
                A_eq=Aeq_mat.todense(),
                b_eq=Beq_mat.todense())

            if res.status == 0:
                min_range.append(res.fun)
            else:
                print"\nInfeasible or Error when computing min range"

        # compute max range
        for i in xrange(0, num_var):
            vector_c = lil_matrix((1, num_var + 2), dtype=float)
            vector_c[0, i + 2] = -1
            res = linprog(
                vector_c.todense(),
                A_ub=Aub_mat.todense(),
                b_ub=Bub_mat.todense(),
                A_eq=Aeq_mat.todense(),
                b_eq=Beq_mat.todense())

            if res.status == 0:
                max_range.append(-res.fun)
            else:
                print"\nInfeasible or Error when computing max range"

        return min_range, max_range


class Plot(object):
    'implements methods for ploting different kind of set'

    @staticmethod
    def plot_boxes(ax, rectangle_set_list, facecolor, edgecolor):
        'plot reachable set using rectangle boxes'

        # return axes object to plot a figure
        n = len(rectangle_set_list)
        assert n > 0, 'empty set'
        assert isinstance(ax, Axes)
        xmin = []
        xmax = []
        ymin = []
        ymax = []
        for i in xrange(0, n):
            xmin.append(rectangle_set_list[i].xmin)
            xmax.append(rectangle_set_list[i].xmax)
            ymin.append(rectangle_set_list[i].ymin)
            ymax.append(rectangle_set_list[i].ymax)

            patch = Rectangle(
                (xmin[i],
                 ymin[i]),
                xmax[i] - xmin[i],
                ymax[i] - ymin[i],
                facecolor=facecolor,
                edgecolor=edgecolor,
                fill=True)
            ax.add_patch(patch)

        xmin.sort()
        xmax.sort()
        ymin.sort()
        ymax.sort()
        min_x = xmin[0]
        max_x = xmax[len(xmax) - 1]
        min_y = ymin[0]
        max_y = ymax[len(ymax) - 1]

        ax.set_xlim(min_x - 0.1 * abs(min_x), max_x + 0.1 * abs(max_x))
        ax.set_ylim(min_y - 0.1 * abs(min_y), max_y + 0.1 * abs(max_y))

        return ax

    @staticmethod
    def plot_vlines(ax, x_pos_list, ymin_list, ymax_list, colors, linestyles):
        'plot vline at x, where ymin <= y <= ymax'

        assert isinstance(ax, Axes)
        assert isinstance(x_pos_list, list)
        assert isinstance(ymin_list, list)
        assert isinstance(ymax_list, list)

        assert len(x_pos_list) == len(ymin_list) == len(
            ymax_list), 'inconsistent data'
        n = len(x_pos_list)

        for i in xrange(0, n):
            if ymin_list[i] > ymax_list[i]:
                raise ValueError(
                    'invalid values, ymin[{}] = {} > ymax[{} = {}]'.format(
                        i, ymin_list[i], i, ymax_list[i]))
        ax.vlines(
            x_pos_list,
            ymin_list,
            ymax_list,
            colors=colors,
            linestyles=linestyles,
            linewidth=2)

        x_pos_list.sort()
        ymin_list.sort()
        ymax_list.sort()
        xmin = x_pos_list[0]
        xmax = x_pos_list[n - 1]
        ymin = ymin_list[0]
        ymax = ymax_list[n - 1]

        ax.set_xlim(xmin - 0.1 * abs(xmin), xmax + 0.1 * abs(xmax))
        ax.set_ylim(ymin - 0.1 * abs(ymin), ymax + 0.1 * abs(ymax))

        return ax

    @staticmethod
    def plot_interpolation_in_space(ux_list, x_list):
        'plot interpolation set in each segment of x, i.e. u(x) where x[i] <= x <= x[i + 1]'

        # we know a1 < = u(x[i]) <= b1 and a2 <= u(x[i + 1]) <= b2
        # The set between x[i] and x[i + 1] is a linear interpolation of u(x[i]) and u(x[i + 1])
