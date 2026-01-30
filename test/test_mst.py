import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'

    # Added assertions:
    
    # how many edges should a minimum spanning tree have? should be n-1
    n = mst.shape[0]
    num_edges = np.count_nonzero(np.triu(mst, k=1))  # count undirected edges once
    assert num_edges == n - 1, f'Proposed MST has incorrect number of edges (got {num_edges}, expected {n-1})'

    # minimum spanning trees should always be connected. 
    
    # from Wikipedia (link in README citation section):
    # 'The algebraic connectivity of a graph G is the second-smallest eigenvalue 
    # of the Laplacian matrix of G. This eigenvalue is greater than 0 if and only if G is a connected graph.'

    # from stanford pdf (link in README citation section):
    # 'the multiplicity of the zeroth eigenvalue reveals the number of connected components of the graph'

    # this means that the number of times 0 appears as an eigenvalue (the multiplicity of the 0 eigenvalue) 
    # in the Laplacian matrix L=D-A exactly equals the number of connected components in the graph.
    # A graph is connected if and only if the smallest eigenvalue is 0 and the second smallest eigenvalue is strictly greater than 0

    # Laplacian L = D - A, where D is the degree matrix and A is the adjacency matrix

    A = (mst != 0).astype(float) # convert the proposed mst adjacency matrix (which is weighted) into an unweighted adj matrix
    D = np.diag(A.sum(axis=1)) # degree matrix; the degree of a node is the num of edges indident to it = sum of row i in the adjacency matrix
    L = D - A

    # for a connected graph: there is exactly one connected component, and one zero eigeinvalue.
    # instead of actually computing the eigenvalues, compute the rank of the matrix (this is easier)
    # the rank is basically the number of independent directions in which the matrix acts, ie the
    # total size of the matrix - the number of zero eigenvalues
    assert np.linalg.matrix_rank(L) == n - 1, "Proposed MST is not connected"
    

def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """
    The MST should select the two lowest-weight edges. Total MST weight should be 3 + 4 = 7.
    """
    adj = np.array([
        [0, 3, 5],
        [3, 0, 4],
        [5, 4, 0]
    ], dtype=float)

    g = Graph(adj)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, expected_weight=7.0)



def check_if_disconnected():
    """
    given a bad MST that is disconnected, do I catch it?
    """
    adj_mat = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
    ], dtype=float)

    # Proposed MST
    mst = adj_mat.copy()

    expected_weight = 3 # total weight of upper-triangle is 3

    with pytest.raises(AssertionError, match="Proposed MST is not connected"):
        check_mst(adj_mat, mst, expected_weight)


def check_if_square():
    '''
    Make sure the appropriate error is raised in construct_mst() if the input is not square.
    '''
    mat = np.array([[1, 1, 1]])
    with pytest.raises(ValueError, match=f"adjacency matrix must be a square"):
        g = Graph(mat)
        g.construct_mst()


def check_if_symmetric():
    '''
    An undirected graph must be symmetric. Make sure the appropriate error is raised in 
    construct_mst() if it is not. 
    '''
    mat = np.array([[1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3]])
    with pytest.raises(ValueError, match=f"adjacency matrix must be symmetric"):
        g = Graph(mat)
        g.construct_mst()
