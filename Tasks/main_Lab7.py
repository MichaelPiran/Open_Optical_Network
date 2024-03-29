# Lab7
import random
import matplotlib.pyplot as plt
from Core.elements import Connection
from Core.istance_network import *
from Core.parameters import *

rnd_con = 150
con_dict = []
# random connection
for i in range(rnd_con):
    i_node = random.choice(list(network.nodes))  # Random input node
    o_node = random.choice(list(network.nodes))  # Random output node
    while i_node == o_node:   # Output node must be different from input ones
        o_node = random.choice(list(network.nodes))
    con_dict.append(Connection({'input': i_node, 'output': o_node, 'signal_power': 1}))
network.stream(con_dict, lat_snr)
print('Route space:')
print(network.route_space)
lbl_axes = []
lat_axes = []
snr_axes = []
for i in range(rnd_con):
    lbl_axes.append(con_dict[i].input + con_dict[i].output)
    lat_axes.append(con_dict[i].latency)
    snr_axes.append(con_dict[i].snr)
plt.figure(figsize=(9, 3))
if lat_snr == 'latency':
    plt.bar(lbl_axes, lat_axes)  # latency distribution
    plt.ylabel('latency')
else:
    plt.bar(lbl_axes, snr_axes)  # snr distribution
    plt.ylabel('snr')
plt.show()

print('Rejected request: ', len(network.rejected_request))
for j in range(len(network.rejected_request)):
    print(network.rejected_request[j].input + network.rejected_request[j].output)
print('End lab7')
