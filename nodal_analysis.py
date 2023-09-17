import numpy as np

NUM_OF_NODES = 3

matrix = np.zeros((NUM_OF_NODES,NUM_OF_NODES))
vector = np.zeros((NUM_OF_NODES,1))
resistors = np.array([[10,1,2],
                      [5,0,2],
                      [10,2,3],
                      [15,0,3]])  #R, node 1, node 2

independent_current_sources = np.array([[2.5, 0, 1],])  #I, node 1, node 2

dependent_current_sources = np.array([[10,3,2]])  #k, end_node, begin_node

for resistor in resistors:
    G = 1/resistor[0]
    node1 = resistor[1]
    node2 = resistor[2]
    matrix[node1][node1] += G
    matrix[node1][node2] -= G
    matrix[node2][node1] -= G
    matrix[node2][node2] += G

for independent_current_source in independent_current_sources:
    I = independent_current_source[0]
    node1 = independent_current_source[1]
    node2 = independent_current_source[2]
