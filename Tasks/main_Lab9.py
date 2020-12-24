import random
from Core.elements import Connection, Line
from Core.istance_network import *


rnd_con = 100
con_dict = []
# random connection
for i in range(rnd_con):
    i_node = random.choice(list(network.nodes))  # Random input node
    o_node = random.choice(list(network.nodes))  # Random output node
    while i_node == o_node:   # Output node must be different from input ones
        o_node = random.choice(list(network.nodes))
    con_dict.append(Connection({'input': i_node, 'output': o_node, 'signal_power': 1}))
network.stream(con_dict, 'snr')  # run for the snr
print('End lab9')
