"""
General function that we use in the code.
"""
# from scipy.constants import c
import numpy as np
import json
import time
from Core.parameters import *


def channel_available(current_line):
    # the first channel free is choosen as comunication channel
    flag = 'false'  # indicate if a channel is already choosen
    channel = 0
    for i in range(len(current_line.state)):  # scan all the state of that pice of line
        if (current_line.state[i] == free) and (flag == 'false'):  # first available free channel
            channel = i
            flag = 'true'  # free channel selected
    # occupy the channel "line.state[n_ch]= occupy"
    # current_line.state[channel] = 'occupied'
    return channel


def set_static_switch_mtx(node_dict):
    new_switch_mtx = {}
    for i_node in node_dict['switching_matrix'].keys():
        possible_o_node = node_dict['switching_matrix'].get(i_node)
        new_switch_mtx[i_node] = possible_o_node
        for o_node in possible_o_node.keys():
            resized_nch = node_dict['switching_matrix'].get(i_node).get(o_node)[:n_ch]  # consider only nch elements
            new_switch_mtx[i_node][o_node] = resized_nch
    return new_switch_mtx


def update_route_space(route_space, nodes, lines, path):
    """
    If the path is composed of two nodes, we rewrite the routing space with
    the state of the line. Instead if we have more than two nodes, we should consider the
    availability of each intermediate node in the switching matrix.
    0 -> Occupied
    1 -> Free
    """
    if len(path) > 2:  # we have more than two nodes in the path
        for i in range(len(path)-1):
            label = path[i]+path[i+1]  # span two nodes at time
            if i == 0:  # first node
                route_space.loc[label] = np.array(lines[label].state)  # update route_space
            else:
                x = np.transpose(nodes[path[i]].switching_matrix[path[i-1]][path[i+1]][:n_ch])
                result = lines[label].state * x  # 1x10 * 10x1
                route_space.loc[label] = np.array(result)  # update route_space
    else:
        # routing space rewritten equal to line
        route_space.loc[path] = np.array(lines[path].state)  # update route_space
    return route_space


def create_file_result(lbl_index, rb_index, lat_index, snr_index, n_conn):
    timestr = time.strftime("%Y_%m_%d_h%Hm%Ms%S")
    namefile = timestr + '.json'
    root_res = Path(__file__).parent.parent
    folder_res = root_res / 'Results' / 'File_results'
    file_res = folder_res / namefile
    data_file = {}
    sub_data_file = {}
    for i in range(n_conn):
        sub_data_file['Bit rate [b/s]'] = rb_index[i]
        sub_data_file['Latency [s]'] = lat_index[i]
        sub_data_file['Snr [dB]'] = snr_index[i]
        data_file[lbl_index[i]] = sub_data_file
    json_object = json.dumps(data_file, indent=4)
    with open(file_res, 'w') as outfile:
        outfile.write(json_object)
