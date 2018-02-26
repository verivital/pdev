'''
This module implements relevant data structures for a unstructured mesh in FEM
Dung Tran: 2018/2/19

Main references:
    1) iFEM: AN INNOVATIVE FINITE ELEMENT METHOD PACKAGE IN MATLAB, LONG CHEN, 2009
'''

import numpy as np
from scipy.sparse import csc_matrix, find


class Triangulation_2D(object):
    'two dimensional triangulation'

    def __init__(self, nodes_mat, elements_mat):

        assert isinstance(
            nodes_mat, np.ndarray) and nodes_mat.shape[1] == 2, 'error: invalid nodes matrix for 2D-triangulation'
        assert isinstance(
            elements_mat, np.ndarray) and elements_mat.shape[1] == 3, 'error: invalid elements matrix for 2D-triangulation'

        self.nodes_mat = nodes_mat    # nodes matrix
        self.elements_mat = elements_mat    # elements matrix
        self.num_nodes = nodes_mat.shape[0]    # number of nodes
        self.num_elements = elements_mat.shape[0]    # number of elements

        # auxiliary data of the 2-D triangulation
        self.edges_mat = None    # edges matrix
        self.num_edges = None    # number of edges
        self.bd_edges_mat = None    # boundary edges matrix
        self.num_bd_edges = None    # number of boundary edges

        self.t2v_mat = None    # triangles to vertices matrix
        self.ed2v_mat = None    # edges to vertices matrix
        self.el2led_mat = None    # elements(triangles) to edges matrix
        self.neighbor_mat = None    # neighbor matrix

    def get_edges_mat(self):
        'compute edges and boundary edges matrix'

        edge1 = self.elements_mat[:, [1, 2]]
        edge2 = self.elements_mat[:, [2, 0]]
        edge3 = self.elements_mat[:, [0, 1]]

        edges = np.vstack([edge1, edge2, edge3])
        edges.sort(axis=1, kind='mergesort')   # row sort

        row = edges[:, 0].reshape(edges.shape[0],)
        col = edges[:, 1].reshape(edges.shape[0],)
        data = np.ones((edges.shape[0],), dtype=int)
        s_mat = csc_matrix((data, (row, col)))

        I, J, V = find(s_mat)
        self.edges_mat = np.transpose(np.array([I[:], J[:]]))
        self.num_edges = V.shape[0]
        print "\nedges matrix = \n{}".format(self.edges_mat)
        print "\nnumber of edges is : {}".format(self.num_edges)
        bd_I = []
        bd_J = []
        for i in xrange(0, V.shape[0]):
            if V[i] == 1:
                bd_I.append(I[i])
                bd_J.append(J[i])

        bd_I = np.asarray(bd_I)
        bd_J = np.asarray(bd_J)
        self.bd_edges_mat = np.transpose(np.array([bd_I[:], bd_J[:]]))
        self.num_bd_edges = bd_I.shape[0]
        print "\nboundary edges matrix = \n{}".format(self.bd_edges_mat)
        print "\nnumber of boundary edges is : {}".format(self.num_bd_edges)
        print "\nnumber of interior edges is : {}".format(self.num_edges - self.num_bd_edges)

    def get_t2v_mat(self):
        'construct the incidence matrix between triangles and vertices'

        n = self.num_nodes
        nt = self.num_elements

        el1 = self.elements_mat[:, 0]
        el2 = self.elements_mat[:, 1]
        el3 = self.elements_mat[:, 2]

        col = np.hstack((el1, el2, el3))
        row = np.hstack((range(nt), range(nt), range(nt)))
        data = np.ones((col.shape[0],), dtype=int)

        self.t2v_mat = csc_matrix((data, (row, col)), shape=(nt, n))
        print "\ntriangles to vertices matrix = \n{}".format(self.t2v_mat.toarray())

    def get_ed2v_mat(self):
        'construct the incidence matrix between edges and vertices'

        if self.edges_mat is None:
            self.get_edges_mat()
        else:
            ne = self.num_edges
            n = self.num_nodes
            row = np.hstack((range(ne), range(ne)))
            col = np.hstack((self.edges_mat[:, 0], self.edges_mat[:, 1]))
            data = np.ones((col.shape[0],), dtype=int)
            self.ed2v_mat = csc_matrix((data, (row, col)), shape=(ne, n))
            print "\nedges to vertices matrix = \n{}".format(self.ed2v_mat.toarray())

    def get_el2ed_mat(self):
        'construct the incidence matrix between elements and edges'

        edge1 = self.elements_mat[:, [1, 2]]
        edge2 = self.elements_mat[:, [2, 0]]
        edge3 = self.elements_mat[:, [0, 1]]

        edges = np.vstack([edge1, edge2, edge3])
        edges.sort(axis=1, kind='mergesort')   # row sort
        E, I, J = np.unique(
            edges, axis=0, return_index=True, return_inverse=True)
        nt = self.num_elements
        self.el2ed_mat = np.transpose(J.reshape(3, nt))
        print "\nelements to edges matrix = \n{}".format(self.el2ed_mat)

    def get_ed2el_mat(self):
        'construct edges to elements matrix'

        if self.el2ed_mat is None:
            self.get_el2ed_mat()
        else:
            nt = self.num_elements
            e0 = self.el2ed_mat[:, 0]
            e1 = self.el2ed_mat[:, 1]
            e2 = self.el2ed_mat[:, 2]
            e = np.array([np.hstack((e0, e1, e2))])
            el = np.array([np.hstack((range(nt), range(nt), range(nt)))])
            v = np.array([np.hstack((np.zeros((nt,), dtype=int), np.ones(
                (nt,), dtype=int), 2 * np.ones((nt,), dtype=int)))])
            ed2el = np.hstack(
                (np.transpose(e), np.transpose(el), np.transpose(v)))
            ind = np.argsort(ed2el[:, 0])
            ed2el = ed2el[ind]
            E, C = np.unique(ed2el[:, 0], axis=0, return_counts=True)

            ne = self.num_edges
            t1 = np.zeros((ne,), dtype=int)
            t2 = np.zeros((ne,), dtype=int)
            k1 = np.zeros((ne,), dtype=int)
            k2 = np.zeros((ne,), dtype=int)
            j = 0
            for i in xrange(0, ne):
                if C[i] == 1:
                    t1[i] = ed2el[j, 1]
                    k1[i] = ed2el[j, 2]
                    t2[i] = t1[i]
                    k2[i] = k1[i]
                    j = j + 1
                else:

                    t1[i] = ed2el[j, 1]
                    k1[i] = ed2el[j, 2]
                    t2[i] = ed2el[j + 1, 1]
                    k2[i] = ed2el[j + 1, 2]
                    j = j + 2

            t1 = np.transpose(np.array([t1]))
            t2 = np.transpose(np.array([t2]))
            k1 = np.transpose(np.array([k1]))
            k2 = np.transpose(np.array([k2]))

            self.ed2el_mat = np.hstack((t1, t2, k1, k2))
            print "\nedges to elements matrix: \n{}".format(self.ed2el_mat)

    def get_neighbor_mat(self):
        'construct neighbor matrix to record neighboring triangles for each triangle'

        if self.ed2el_mat is None:
            self.get_ed2el_mat()

        ed2el = self.ed2el_mat
        el2ed = self.el2ed_mat
        nt = self.num_elements
        neighbor = np.zeros((nt, 3), dtype=int)
        for i in xrange(0, nt):
            for j in xrange(0, 3):
                ed = el2ed[i, j]
                e1 = ed2el[ed, 0]
                e2 = ed2el[ed, 1]
                if e1 != e2:
                    if e1 != i:
                        neighbor[i, j] = e1
                    elif e2 != i:
                        neighbor[i, j] = e2

                else:
                    neighbor[i, j] = e1

        self.neighbor_mat = neighbor
        print "\nneighbor = {}".format(neighbor)

    def get_area(self, el_index):
        'compute the area of the ith-triangle'

        assert isinstance(
            el_index, int) and 0 <= el_index <= self.num_elements - 1, 'error: invalid element index'

        element = self.elements_mat[el_index, :]
        node1_ind = element[0]
        node2_ind = element[1]
        node3_ind = element[2]

        node1 = self.nodes_mat[node1_ind, :]
        node2 = self.nodes_mat[node2_ind, :]
        node3 = self.nodes_mat[node3_ind, :]

        # area = x1y2 + x2y3 + x3y1 - x1y3 - x2y1 - x3y2
        area = 0.5 * (node1[0] * node2[1] + node2[0] * node3[1] + node3[0] * node1[1] -
                      node1[0] * node3[1] - node2[0] * node1[1] - node3[0] * node2[1])

        return area


def test():
    'test mesh.py'

    nodes = np.array([[1, 0], [1, 1], [0, 1], [-1, 1],
                      [-1, 0], [-1, -1], [0, -1], [0, 0]])
    elements = np.array([[0, 1, 7], [2, 7, 1], [7, 2, 4],
                         [3, 4, 2], [6, 7, 5], [4, 5, 7]])

    mesh = Triangulation_2D(nodes, elements)

    print "\nnumber of nodes of the mess is : {}".format(mesh.num_nodes)
    print "\nnumber of elements of the mess is : {}".format(mesh.num_elements)
    print "\nnodes matrix of the mess is : \n{}".format(mesh.nodes_mat)
    print "\nelements matrix of the mess is : \n{}".format(mesh.elements_mat)

    mesh.get_edges_mat()
    mesh.get_t2v_mat()
    mesh.get_ed2v_mat()
    mesh.get_el2ed_mat()
    mesh.get_ed2el_mat()
    mesh.get_neighbor_mat()
    mesh.get_area(0)


if __name__ == '__main__':

    test()
