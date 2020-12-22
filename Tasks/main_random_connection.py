# Ex6 Lab 4
import random
import matplotlib.pyplot as plt
from Core.istance_network import *
from Core.elements import Connection
from Core.parameters import *

con_dict = []
# random connection
for i in range(100):
    i_node = random.choice(list(network.nodes))  # Random input node
    o_node = random.choice(list(network.nodes))  # Random output node
    while i_node == o_node:   # Output node must be different from input ones
        o_node = random.choice(list(network.nodes))
    con_dict.append(Connection({'input': i_node, 'output': o_node, 'signal_power': 1}))

network.stream(con_dict, lat_snr)
# network.stream(con_dict, 'latency')
# network.stream(con_dict, 'snr')
lbl_axes = []
lat_axes = []
snr_axes = []
for i in range(100):
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
print('End lab4')
# plt.subplot(121)
# plt.bar(lbl_axes, lat_axes)  # latency distribution
# plt.ylabel('latency')
# plt.subplot(122)
# plt.bar(lbl_axes, snr_axes)  # snr distribution
# plt.ylabel('snr')
# plt.show()
