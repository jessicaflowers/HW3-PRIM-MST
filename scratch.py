import numpy as np
import heapq
from typing import Union
from mst.graph import Graph
from sklearn.metrics import pairwise_distances


# g = Graph("data/citation_network.adjlist")  # start with mini
# traversal = g.bfs("Tony Capra")


# file_path = './data/slingshot_example.txt'
# coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
# dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
# g = Graph(dist_mat)
# g.construct_mst()
# g.adj_mat, g.mst


mat = np.array([[1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3]])

g = Graph(mat)
g.construct_mst()
# A = (mst != 0).astype(float)          # treat any nonzero weight as an edge
# D = np.diag(A.sum(axis=1))            # degree matrix
# L = D - A                             # Laplacian
# n = mst.shape[0]
# print(L)
# print(n)
# print(np.linalg.matrix_rank(L))# assert np.linalg.matrix_rank(L) == n - 1, "Proposed MST is not connected"
