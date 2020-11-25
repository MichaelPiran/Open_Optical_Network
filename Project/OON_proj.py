import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
import random


class SignalInformation(object):
    def __init__(self, power, path):
        self._signal_power = power
        self._noise_power = 0.0
        self._latency = 0.0
        self._path = path

    @property
    def signal_power(self):
        return self._signal_power

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    @property
    def noise_power(self):
        return self._noise_power

    @noise_power.setter
    def noise_power(self, noise):
        self._noise_power = noise

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    def add_noise(self, noise):  # Update the noise power
        self.noise_power += noise

    def add_latency(self, latency):  # Update the latency
        self.latency += latency

    def next(self):  # Update the path
        self.path = self.path[1:]


###################################################################


class Node(object):
    def __init__(self, node_dic):
        """"
        input_node_dic = {'label': string , 'position ' : tuple(float,float), 'connected_nodes': list[string],
                    'successive': dict[Line]}
        """
        self._label = node_dic['label']
        self._position = node_dic['position']
        self._connected_nodes = node_dic['connected_nodes']
        self._successive = {}

    @property
    def label(self):
        return self._label

    @property
    def position(self):
        return self._position

    @property
    def connected_nodes(self):
        return self._connected_nodes

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    def propagate(self, signal_information):
        path = signal_information.path
        if len(path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]
            signal_information.next()
            signal_information = line.propagate(signal_information)  # Call successive element propagate method
        return signal_information

####################################################################


class Line(object):
    def __init__(self, line_dict):
        self._label = line_dict['label']
        self._length = line_dict['length']
        self._successive = {}
        self._state = 'free'  # free or occupied

    @property
    def label(self):
        return self._label

    @property
    def length(self):
        return self._length

    @property
    def successive(self):
        return self._successive

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    def latency_generation(self):
        latency = self.length / (c * 2 / 3)
        return latency

    def noise_generation(self, signal_power):
        noise = 1e-3 * signal_power * self.length
        return noise

    def propagate(self, signal_information):
        latency = self.latency_generation()
        signal_information.add_latency(latency)

        signal_power = signal_information.signal_power
        noise = self.noise_generation(signal_power)
        signal_information.add_noise(noise)

        node = self.successive[signal_information.path[0]]
        signal_information = node.propagate(signal_information)  # Call successive element propagate method
        return signal_information


##################################################################


class Network(object):
    def __init__(self, json_path):
        node_json = json.load(open(json_path, 'r'))  # give path of the file
        self._nodes = {}  # Dict of Node
        self._lines = {}  # Dict of Line
        self._weighted_paths = pd.DataFrame()

        for node_label in node_json:
            # Create the node instance
            node_dict = node_json[node_label]
            node_dict['label'] = node_label
            node = Node(node_dict)
            self._nodes[node_label] = node
            # Create the line instances
            for connected_node_label in node_dict['connected_nodes']:
                line_dict = {}
                line_label = node_label + connected_node_label
                line_dict['label'] = line_label
                node_position = np.array(node_json[node_label]['position'])
                connected_node_position = np.array(node_json[connected_node_label]['position'])
                line_dict['length'] = np.sqrt(np.sum((node_position - connected_node_position) ** 2))
                line = Line(line_dict)
                self._lines[line_label] = line

    @property
    def weighted_paths(self):
        return self._weighted_paths

    @property
    def nodes(self):
        return self._nodes

    @property
    def lines(self):
        return self._lines

    def draw(self):
        nodes = self.nodes
        for node_label in nodes:
            n0 = nodes[node_label]
            x0 = n0.position[0]
            y0 = n0.position[1]
            plt.plot(x0, y0, 'go ', markersize=10)
            plt.text(x0 + 20, y0 + 20, node_label)
            for connected_node_label in n0.connected_nodes:
                n1 = nodes[connected_node_label]
                x1 = n1.position[0]
                y1 = n1.position[1]
                plt.plot([x0, x1], [y0, y1], 'b')
        plt.title('Network ')
        plt.show()

    def find_paths(self, label1, label2):
        cross_nodes = [key for key in self.nodes.keys() if ((key != label1) & (key != label2))]
        cross_lines = self.lines.keys()
        inner_paths = {'0': label1}
        for i in range(len(cross_nodes) + 1):
            inner_paths[str(i + 1)] = []
            for inner_path in inner_paths[str(i)]:
                inner_paths[str(i + 1)] += [inner_path + cross_node for cross_node in cross_nodes
                                            if ((inner_path[-1] + cross_node in cross_lines) &
                                                (cross_node not in inner_path))]
        paths = []
        for i in range(len(cross_nodes) + 1):
            for path in inner_paths[str(i)]:
                if path[-1] + label2 in cross_lines:
                    paths.append(path + label2)
        return paths

    def connect(self):
        nodes_dict = self.nodes
        lines_dict = self.lines
        for node_label in nodes_dict:
            node = nodes_dict[node_label]
            for connected_node in node.connected_nodes:
                line_label = node_label + connected_node
                line = lines_dict[line_label]
                line.successive[connected_node] = nodes_dict[connected_node]
                node.successive[line_label] = lines_dict[line_label]

    def propagate(self, signal_information):
        path = signal_information.path
        start_node = self.nodes[path[0]]
        propagated_signal_information = start_node.propagate(signal_information)
        return propagated_signal_information

    def set_weighted_paths(self, paths, latencies, noises, snrs):  # Create the weighted graph
        self._weighted_paths['path'] = paths
        self._weighted_paths['latency'] = latencies
        self._weighted_paths['noise'] = noises
        self._weighted_paths['snr'] = snrs

    def find_best_snr(self, i_node, o_node):  # Find path with higher snr between two node
        path_list = []
        snr_list = []
        for path in Network.find_paths(self, i_node.label, o_node.label):  # Retrieve all paths
            for row in self._weighted_paths.itertuples():  # Search inside the dataframe
                if row.path == path:
                    # If the line between each node is occupied don't consider it
                    flag_state = 'free'
                    for i in range(len(row.path)-1):
                        if self._lines[row.path[i]+row.path[i+1]].state == 'false':
                            flag_state = 'false'
                    if flag_state == 'free':
                        path_list.append(row.path)
                        snr_list.append(row.snr)
        max_path = path_list[snr_list.index(max(snr_list))]
        return max_path

    def find_best_latency(self, i_node, o_node):  # Find path with lower latency between two node
        path_list = []
        lat_list = []
        for path in Network.find_paths(self, i_node.label, o_node.label):  # Retrieve all paths
            for row in self._weighted_paths.itertuples():  # Search inside the dataframe
                if row.path == path:
                    # If the line between each node is occupied don't consider it
                    flag_state = 'free'
                    for i in range(len(row.path)-1):
                        if self._lines[row.path[i]+row.path[i+1]].state == 'false':
                            flag_state = 'false'
                    if flag_state == 'free':
                        path_list.append(row.path)
                        lat_list.append(row.latency)
        min_path = path_list[lat_list.index(min(lat_list))]
        return min_path

    def stream(self, conn_list, lat_snr_label):
        path = []
        for elem in conn_list:  # Check all connection istance
            if lat_snr_label == 'latency':  # If check for latency
                path = Network.find_best_latency(self, self._nodes[elem.input], self._nodes[elem.output])
            elif lat_snr_label == 'snr':  # If check for snr
                path = Network.find_best_snr(self, self._nodes[elem.input], self._nodes[elem.output])
            if len(path) != 0 :  # There is almost a free path available
                signal_information = SignalInformation(1, path)
                signal_information = Network.propagate(self, signal_information)
                if lat_snr_label == 'latency':  # If check for latency
                    elem.latency = signal_information.latency
                elif lat_snr_label == 'snr':
                    elem.snr = 10 * np.log10(signal_information.signal_power / signal_information.noise_power)
            else:
                elem.latency = None
                elem.snr = 0


#############################################################


class Connection(object):
    def __init__(self, conn_dict):
        self._input = conn_dict['input']
        self._output = conn_dict['output']
        self._signal_power = conn_dict['signal_power']
        self._latency = 0.0
        self._snr = 0.0

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def signal_power(self):
        return self._signal_power

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @property
    def snr(self):
        return self._snr

    @snr.setter
    def snr(self, snr):
        self._snr = snr


#############################################################

def main():
    network = Network('nodes.json')

    network.connect()
    node_labels = network.nodes.keys()  # A,B,C,D;E,F
    pairs = []
    for label1 in node_labels:
        for label2 in node_labels:
            if label1 != label2:
                pairs.append(label1 + label2)
    # columns = ['path', 'latency', 'noise', 'snr']
    df = pd.DataFrame()
    paths = []
    n_paths = []  # paths without '->'
    latencies = []
    noises = []
    snrs = []
    for pair in pairs:
        for path in network.find_paths(pair[0], pair[1]):
            n_paths.append(path)
            path_string = ''
            for node in path:
                path_string += node + '->'
            paths.append(path_string[: -2])
            # Propagation
            signal_information = SignalInformation(1, path)
            signal_information = network.propagate(signal_information)
            latencies.append(signal_information.latency)
            noises.append(signal_information.noise_power)
            snrs.append(
                10 * np.log10(
                    signal_information.signal_power / signal_information.noise_power
                )
            )
    df['path'] = paths
    df['latency'] = latencies
    df['noise'] = noises
    df['snr'] = snrs

    # Es1 lab4
    network.set_weighted_paths(n_paths, latencies, noises, snrs)  # Instance weighted graph

    # Es2 lab4
    # snr_custom_nodes = network.find_best_snr(network.nodes['C'], network.nodes['A'])  # Search for path with bst snr
    # print('Path with best snr between the two input nodes is: ', snr_custom_nodes)

    # Es3 lab4
    # lat_custom_nodes = network.find_best_latency(network.nodes['C'], network.nodes['A'])  # Search bst snr
    # print('Path with best latency between the two input nodes is: ', lat_custom_nodes)
    # Es5 lab4

    # con1 = Connection({'input': 'E', 'output': 'B', 'signal_power': 1})
    # con2 = Connection({'input': 'B', 'output': 'D', 'signal_power': 1})
    # con3 = Connection({'input': 'C', 'output': 'A', 'signal_power': 1})
    # con4 = Connection({'input': 'A', 'output': 'C', 'signal_power': 1})
    # con_dict = [con1, con2, con3, con4]
    # network.stream(con_dict, 'snr')   # Latency or snr

    # Es6 lab4
    # con_dict = []
    # for i in range(100):
    #     i_node = random.choice(list(network.nodes))  # Random input node
    #     o_node = random.choice(list(network.nodes))  # Random output node
    #     while i_node == o_node:   # Output node must be different from input ones
    #         o_node = random.choice(list(network.nodes))
    #     con_dict.append(Connection({'input': i_node, 'output': o_node, 'signal_power': 1}))
    # network.stream(con_dict, 'latency')
    # network.stream(con_dict, 'snr')
    # lbl_axes = []
    # lat_axes = []
    # snr_axes = []
    # for i in range(100):
    #     lbl_axes.append(con_dict[i].input + con_dict[i].output)
    #     lat_axes.append(con_dict[i].latency)
    #     snr_axes.append(con_dict[i].snr)
    # plt.figure(figsize=(9, 3))
    # plt.subplot(121)
    # plt.bar(lbl_axes, lat_axes)  # latency distribution
    # plt.ylabel('latency')
    # plt.subplot(122)
    # plt.bar(lbl_axes, snr_axes)  # snr distribution
    # plt.ylabel('snr')
    # plt.show()

    # Es7 lab4


if __name__ == "__main__":
    main()
