"""
General function that we use.
"""
from scipy.constants import c


def channel_available(current_line):
    # the first channel free is choosen as comunication channel
    flag = 'false'  # indicate if a channel is already choosen
    channel = 0
    for i in range(len(current_line.state)):  # scan all the state of that pice of line
        if (current_line.state[i] == 'free') and (flag == 'false'):  # first available free channel
            channel = i
            flag = 'true'  # free channel selected
    # occupy the channel "line.state[n_ch]= occupy"
    current_line.state[channel] = 'occupied'
    return channel


def stream_propagate(lightpath, lines):
    # given a line between two nodes, propagate a lightpath
    if len(lightpath.path) > 1:
        line_label = lightpath.path[:2]  # consider two node at time
        current_line = lines[line_label]  # select current line
        length = current_line.length  # take lenght of that piece of line
        latency = length / (c * 2 / 3)  # evaluate the latency
        noise = 1e-3 * lightpath.signal_power * length  # evaluate the noise
        lightpath.add_latency(latency)  # latency of lightpath
        lightpath.add_noise(noise)  # noise of lightpath

        free_channel = channel_available(current_line)  # select first channel available
        lightpath.set_channel(free_channel)
        lightpath.next()  # consider next lightpath
        lightpath = stream_propagate(lightpath, lines)  # Call successive element propagate method
    return lightpath
