# Lab6
from Core.istance_network import *
import random
from Core.elements import Connection
from Core.parameters import *

print('Node A, switching matrix:')
print(network.nodes['A'].switching_matrix)  # Example of switching matrix
print(network.nodes['A'].switching_matrix['B']['C'])  # Example of switching matrix
con_dict = []
# random connection
for i in range(10):
    i_node = random.choice(list(network.nodes))  # Random input node
    o_node = random.choice(list(network.nodes))  # Random output node
    while i_node == o_node:   # Output node must be different from input ones
        o_node = random.choice(list(network.nodes))
    con_dict.append(Connection({'input': i_node, 'output': o_node, 'signal_power': 1}))
network.stream(con_dict, lat_snr)
print('Route space:')
print(network.route_space)
