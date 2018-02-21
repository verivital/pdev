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

        assert isinstance(nodes_mat, np.ndarray) and nodes_mat.shape[1] == 2, 'error: invalid nodes matrix for 2D-triangulation'
        assert isinstance(elements_mat, np.ndarray) and elements_mat.shape[1] == 3, 'error: invalid elements matrix for 2D-triangulation'

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
        self.el2led_mat = None    # elements to edges matrix
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
        E, I, J = np.unique(edges, axis=0, return_index=True, return_inverse=True)
        nt = self.num_elements
        self.el2ed_mat = np.transpose(J.reshape(3, nt))
        print "\nelements to edges matrix = \n{}".format(self.el2ed_mat)

    def get_ed2el_mat(self):
        'construct edges to elements matrix'

        pass

    def get_neighbor_mat(self):
        'construct neighbor matrix to record neighboring triangles for each triangle'

        pass


def test():
    'test mesh.py'

    nodes = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [0, 0]])
    elements = np.array([[0, 1, 7], [2, 7, 1], [7, 2, 4], [3, 4, 2], [6, 7, 5], [4, 5, 7]])

    mesh = Triangulation_2D(nodes, elements)

    print "\nnumber of nodes of the mess is : {}".format(mesh.num_nodes)
    print "\nnumber of elements of the mess is : {}".format(mesh.num_elements)
    print "\nnodes matrix of the mess is : \n{}".format(mesh.nodes_mat)
    print "\nelements matrix of the mess is : \n{}".format(mesh.elements_mat)

    mesh.get_edges_mat()
    mesh.get_t2v_mat()
    mesh.get_ed2v_mat()
    mesh.get_el2ed_mat()

if __name__ == '__main__':

    test()
