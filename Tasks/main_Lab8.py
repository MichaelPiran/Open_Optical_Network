# Lab8
import random
import matplotlib.pyplot as plt
from Core.elements import Connection
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
lbl_axes = []
rb_axes = []
lat_axes = []
snr_axes = []
sum_rb = 0
for i in range(rnd_con):
    lbl_axes.append(con_dict[i].input + con_dict[i].output)
    rb_axes.append(con_dict[i].bit_rate)
    lat_axes.append(con_dict[i].latency)
    snr_axes.append(con_dict[i].snr)
    sum_rb += con_dict[i].bit_rate
plt.figure(figsize=(9, 3))
plt.bar(lbl_axes, rb_axes)  # bit rate distribution
plt.ylabel('bit_rate [b/s]')

plt.figure(figsize=(9, 3))
plt.bar(lbl_axes, lat_axes)  # Latency distribution
plt.ylabel('Latency [s]')

plt.figure(figsize=(9, 3))
plt.bar(lbl_axes, snr_axes)  # snr distribution
plt.ylabel('snr [dB]')
plt.show()
avg_rb = sum_rb/rnd_con
print('The average bit rate is: ', avg_rb, 'bit/s')
print('The total capacity allocated: ', sum_rb, 'bit/s')
print('The rejected request are ', len(network.rejected_request), ': ')
for r in range(len(network.rejected_request)):
    print('In: ', network.rejected_request[r].input, ' Out: ', network.rejected_request[r].output)
print('End lab8')
