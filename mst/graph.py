import numpy as np
import heapq
from typing import Union


class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        W = self.adj_mat

        # Adjacency matrix must be a square 2D array
        if not isinstance(W, np.ndarray) or W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("adjacency matrix must be a square")
            
        # adjacency matrix must be symmetric 
        diff = W - W.T
        if not np.all(np.abs(diff) <= 0.001):
            raise ValueError("adjacency matrix must be symmetric")

        n = len(W) # number of nodes
        mst = np.zeros((n, n), dtype=float) # initialize mst
        start = 0
        visited = set() # gonna keep track of nodes that i visited
        visited.add(start) # add start node 

        # priority queue
        heap = []

        # add all edges from the start node to the heap
        for v in range(n):
            if v == start: #skip self loops
                continue
            w = W[start, v] # this indexes matrix at start, v
            if w == 0: 
                continue 
            heap.append((w,start,v)) # add edge start-v as canditate 

        heapq.heapify(heap)
        
        edges_added = 0 # Track how many edges i have placed into the MST

        # this is the main block of prim's algo, should go on n-1 steps
        while heap and edges_added < (n - 1):

            # Pop the min  weight edge 
            w, u, v = heapq.heappop(heap)
            if v in visited: # If v is already in visited, skip it and keep going.
                continue
            # Add v to visited
            visited.add(v)

            # Add the edge to the mst adj matrix  
            mst[u, v] = w
            mst[v, u] = w

            edges_added += 1

            # look for edges ending in nodes that have not been visited yet
            for x in range(n):
                if x == v:
                    continue
                if x in visited:
                    continue

                wx = W[v, x] # this indexes the adjacency matrix at v and x
                if wx == 0:
                    continue
                
                heapq.heappush(heap, (wx, v, x)) # push edge into heap


        # checks:
        # graph couldnt have been connected if i didnt add n-1 edges
        if edges_added != (n - 1):
            raise ValueError("Input graph must be connected to have an MST")
        

        self.mst = mst





