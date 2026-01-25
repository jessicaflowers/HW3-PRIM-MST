import numpy as np
import heapq
from typing import Union
from mst.graph import Graph

# g = Graph("data/citation_network.adjlist")  # start with mini
# traversal = g.bfs("Tony Capra")


file_path = './data/small.csv'
g = Graph(file_path)
g.construct_mst()