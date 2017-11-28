'''
This module implements Plot class and its methods
Dung Tran: Nov/2017
'''

from matplotlib.patches import Rectangle
from matplotlib.axes import Axes
from engine.set import RectangleSet2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Plot(object):
    'implements methods for ploting different kind of set'

    @staticmethod
    def plot_boxes(ax, rectangle_set_list, facecolor, edgecolor):
        'plot reachable set using rectangle boxes'

        # return axes object to plot a figure
        n = len(rectangle_set_list)
        assert n > 0, 'empty set'
        assert isinstance(ax, Axes)

        for i in xrange(0, n):
            assert isinstance(rectangle_set_list[i], RectangleSet2D)

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
    def plot_3d_boxes(ax, boxes_list, facecolor, linewidth, edgecolor):
        'plot 3d boxes contain reachable set'

        pass
