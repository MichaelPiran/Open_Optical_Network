"""
General function that we use in the code.
"""
# from scipy.constants import c
import numpy as np
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


# def stream_propagate(lightpath, lines):
#     # given a line between two nodes, propagate a lightpath
#     if len(lightpath.path) > 1:
#         line_label = lightpath.path[:2]  # consider two node at time
#         current_line = lines[line_label]  # select current line
#         length = current_line.length  # take lenght of that piece of line
#         latency = length / (c * 2 / 3)  # evaluate the latency
#         noise = 1e-3 * lightpath.signal_power * length  # evaluate the noise
#         lightpath.add_latency(latency)  # latency of lightpath
#         lightpath.add_noise(noise)  # noise of lightpath
#
#         free_channel = channel_available(current_line)  # select first channel available
#         # occupy the channel "line.state[n_ch]= occupy"
#         current_line.state[free_channel] = occupied
#         # update the routing space
#         # route_space.loc[line_label, free_channel] = 'occupied'  # update routing space
#
#         lightpath.set_channel(free_channel)
#         lightpath.next()  # consider next lightpath
#         lightpath = stream_propagate(lightpath, lines)  # Call successive element propagate method
#     return lightpath


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
