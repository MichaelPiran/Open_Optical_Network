# import random
import matplotlib.pyplot as plt
from Core.elements import Connection
from Core.istance_network import *
from Core.utils import create_file_result

rnd_con = 100
con_dict = []
# Uniform traffic matrix
ret_node = network.manage_traffic_mtx()
eff_n_conn = ret_node[2]
if eff_n_conn > 0:
    for i in range(eff_n_conn):
        i_node = ret_node[0][i]
        o_node = ret_node[1][i]
        con_dict.append(Connection({'input': i_node, 'output': o_node, 'signal_power': 1}))
    print("Connection considered: ", eff_n_conn, "out of: ", nconn_t_mtx, " connection")
    network.stream(con_dict, lat_snr)  # run for the snr
else:
    print("Network saturation, no available connection")
    exit()

lbl_axes = []
rb_axes = []
lat_axes = []
snr_axes = []
sum_rb = 0
for i in range(eff_n_conn):
    lbl_axes.append(con_dict[i].input + con_dict[i].output)
    rb_axes.append(con_dict[i].bit_rate)
    lat_axes.append(con_dict[i].latency)
    snr_axes.append(con_dict[i].snr)
    sum_rb += con_dict[i].bit_rate
if CreateFileFlag == 'True':
    # Create output data file
    create_file_result(lbl_axes, rb_axes, lat_axes, snr_axes, eff_n_conn)
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
print('The rejected request are ', len(network.rejected_request), ' -- No path available ')
for r in range(len(network.rejected_request)):
    print('In: ', network.rejected_request[r].input, ' Out: ', network.rejected_request[r].output)
print('End lab10')
