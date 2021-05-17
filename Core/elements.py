import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc
import random
from scipy.constants import c, Planck
from Core.utils import update_route_space, set_static_switch_mtx
from Core.parameters import *
from Core.conversion import conv_db_to_linear, conv_kilo_to_unit


###################################################################


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


class Lightpath(SignalInformation):  # Eredith from class Siglan Information
    def __init__(self, power, path):
        super().__init__(power, path)  # to eredith also the method
        self._channel = 0  # Indicate which frequency slot the signal occupy
        self._previous_node = 'empty'  # indicate the previous node crossed by lightpath
        self._rs = Rs  # Simbol Rate
        self._df = Df  # Frequency spacing between channel

    @property
    def channel(self):
        return self._channel

    def set_channel(self, channel):
        self._channel = channel

    @property
    def previous_node(self):
        return self._previous_node

    def set_previous_node(self, p_node):
        self._previous_node = p_node

    @property
    def rs(self):
        return self._rs

    @property
    def df(self):
        return self._df


###################################################################


class Node(object):
    def __init__(self, node_dic):
        self._label = node_dic['label']
        self._position = node_dic['position']
        self._connected_nodes = node_dic['connected_nodes']
        self._successive = {}
        self._switching_matrix = {}
        self._transceiver = ''

    @property
    def transceiver(self):
        return self._transceiver

    @transceiver.setter
    def transceiver(self, strategy):
        self._transceiver = strategy

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

    @property
    def switching_matrix(self):
        return self._switching_matrix

    @switching_matrix.setter
    def switching_matrix(self, matrix):
        self._switching_matrix = matrix

    def propagate(self, signal):
        path = signal.path
        if len(path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]
            line.optimal_launch_power = line.optimized_launch_power()  # Pch
            """ 
            Check if we are propagating a Lightpath o a SignalInformation.
            If the former:
                -Check free channel
                -Occupy the line state
                -Update switching matrix
            """
            if hasattr(signal, 'channel'):
                # Node propagate Lightpath
                free_channel = signal.channel
                line.state[free_channel] = occupied  # occupy the channel

                flag_availability = 'true'
                if signal.previous_node != 'empty':
                    # Update switching matrix of the node
                    mtx_node = self.switching_matrix
                    sw_mtx_position = mtx_node[signal.previous_node][line_label[1]]  # sw matrix to check free channel
                    if free_channel - 1 >= 0:  # Occupy previous channel
                        if sw_mtx_position[free_channel-1] == free:
                            # self._switching_matrix[signal.previous_node][line_label[1]][free_channel-1] = occupied
                            mtx_node[signal.previous_node][line_label[1]][free_channel - 1] = occupied
                            self.switching_matrix = mtx_node  # update sw mtx
                        else:
                            flag_availability = 'false'

                    if sw_mtx_position[free_channel] == free:
                        # self._switching_matrix[signal.previous_node][line_label[1]][free_channel] = occupied
                        mtx_node[signal.previous_node][line_label[1]][free_channel] = occupied
                        self.switching_matrix = mtx_node  # update sw mtx
                    else:
                        flag_availability = 'false'

                    if free_channel + 1 < n_ch:  # Occupy next channel
                        if sw_mtx_position[free_channel+1] == free:
                            # self._switching_matrix[signal.previous_node][line_label[1]][free_channel + 1] = occupied
                            mtx_node[signal.previous_node][line_label[1]][free_channel + 1] = occupied
                            self.switching_matrix = mtx_node  # update sw mtx
                        else:
                            flag_availability = 'false'
                # can block channel and its neighbourghs
                if flag_availability == 'true':
                    signal.set_previous_node(path[0])  # update previous node of the lightpath
                    signal.next()
                    signal = line.propagate(signal)  # Call successive element propagate method
                    # otherwise loose the propagation
            else:
                # Node propagate SignalInformation
                signal.next()
                signal = line.propagate(signal)  # Call successive element propagate method

        return signal


####################################################################


class Line(object):
    def __init__(self, line_dict):
        self._label = line_dict['label']
        self._length = line_dict['length']
        self._successive = {}
        self._state = np.ones(n_ch, int)  # list of 10 channel 'free' or 'occupied'
        self._n_amplifiers = np.ceil(line_dict['length']/conv_kilo_to_unit(80))  # one amplifier at each 80km
        self._gain = conv_db_to_linear(16)  # G
        self._noise_figure = conv_db_to_linear(3)  # NF
        self._alpha = 0.2/(20*np.log10(np.exp(1))*1000)  # alpha = 0.2 dB/Km
        self._beta2_abs = 2.13e-26  # beta2_abs = 2.13e-26 (m*(Hz^2))^-1
        self._gamma = 1.27e-3  # gamma = 1.27 (W m)^-1
        self._optimal_launch_power = 0

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

    @property
    def n_amplifiers(self):
        return self._n_amplifiers

    @property
    def gain(self):
        return self._gain

    @property
    def noise_figure(self):
        return self._noise_figure

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta2_abs(self):
        return self._beta2_abs

    @property
    def gamma(self):
        return self._gamma

    @property
    def optimal_launch_power(self):
        return self._optimal_launch_power

    @optimal_launch_power.setter
    def optimal_launch_power(self, power):
        self._optimal_launch_power = power

    def latency_generation(self):
        latency = self.length / (c * 2 / 3)  # delay = L/(c/n)
        return latency

    def noise_generation(self):
        # noise = 1e-9 * signal_power * self.length
        ase = Line.ase_generation(self)
        nli = Line.nli_generation(self)[1]
        noise = ase + nli
        return noise

    def propagate(self, signal):
        latency = self.latency_generation()
        signal.add_latency(latency)

        noise = self.noise_generation()  # ASE + NLI
        pch = Line.optimized_launch_power(self)
        gsnr = pch / noise
        signal.add_noise(1/gsnr)

        node = self.successive[signal.path[0]]
        signal = node.propagate(signal)  # Call successive element propagate method
        return signal

    def ase_generation(self):
        ase = self._n_amplifiers * Planck * freq * Bn * self._noise_figure * (self._gain - 1)  # ASE
        return ase

    def nli_generation(self):
        # eta_nli = 8e-9
        # nli = 2e-7
        nli_struct = []
        n_span = self._n_amplifiers - 1
        x1 = 0.5 * (np.pi**2) * self._beta2_abs * (Rs**2) * (1/self._alpha) * n_ch**(2*(Rs/Df))
        x2 = (self._gamma ** 2) / (4 * self._alpha * self._beta2_abs * (Rs ** 3))
        eta_nli = (16/(27 * np.pi)) * np.log10(x1) * x2
        # p_ch = (self.ase_generation() / (2 * eta_nli))**(1/3)
        p_ch = ((Planck*freq*Bn*self._noise_figure*self._alpha*self.length)/(2*Bn*eta_nli))**(1/3)
        # print("channel power: ", p_ch)
        nli = eta_nli * n_span * (p_ch ** 3) * Bn
        nli_struct.append(eta_nli)
        nli_struct.append(nli)
        # print("eta nli:  ", eta_nli)
        # print("nli:  ", nli)
        return nli_struct

    def optimized_launch_power(self):
        p_ase = self.ase_generation()
        eta_nli = self.nli_generation()[0]
        optimal_launch_power = (p_ase/(2*eta_nli))**(1/3)  # Pch
        return optimal_launch_power


##################################################################


class Network(object):
    def __init__(self, json_path):
        node_json = json.load(open(json_path, 'r'))  # give path of the file
        self._nodes = {}  # Dict of Node
        self._lines = {}  # Dict of Line
        self._weighted_paths = pd.DataFrame()
        self._route_space = pd.DataFrame()
        self._static_switch_mtx = {}  # complete switching network from file
        self._rejected_request = []
        self._traffic_matrix = {}

        routing_state = []  # state of each path for the routing space
        routing_index = []  # path for the routing space
        for node_label in node_json:
            # Create the node instance
            node_dict = node_json[node_label]
            node_dict['label'] = node_label
            node = Node(node_dict)
            self._nodes[node_label] = node
            # Create the line    instances
            for connected_node_label in node_dict['connected_nodes']:
                line_dict = {}
                line_label = node_label + connected_node_label
                line_dict['label'] = line_label
                node_position = np.array(node_json[node_label]['position'])
                connected_node_position = np.array(node_json[connected_node_label]['position'])
                line_dict['length'] = np.sqrt(np.sum((node_position - connected_node_position) ** 2))
                line = Line(line_dict)
                routing_state.append(line.state)
                routing_index.append(line_label)
                self._lines[line_label] = line
            # store static switching matrix of whole network. Take only the wanted n of channel
            self._static_switch_mtx[node_label] = set_static_switch_mtx(node_dict)
            # Set node transceiver attribute
            if 'transceiver' in node_dict:
                self._nodes[node_label].transceiver = node_dict['transceiver']
            else:  # default value
                self._nodes[node_label].transceiver = 'fixed_rate'
        # Define the route space
        self._route_space = pd.DataFrame(routing_state, routing_index)  # Route space data frame

    @property
    def nodes(self):
        return self._nodes

    @property
    def lines(self):
        return self._lines

    @property
    def weighted_paths(self):
        return self._weighted_paths

    def set_weighted_paths(self):
        node_labels = self._nodes.keys()  # A,B,C,D;E,F
        pairs = []
        for label1 in node_labels:
            for label2 in node_labels:
                if label1 != label2:
                    pairs.append(label1 + label2)
        df = pd.DataFrame()
        paths = []
        n_paths = []  # paths without '->'
        latencies = []
        noises = []
        snrs = []
        for pair in pairs:
            for path in Network.find_paths(self, pair[0], pair[1]):
                n_paths.append(path)  # normal path without ->
                path_string = ''
                for node in path:
                    path_string += node + '->'
                paths.append(path_string[: -2])
                # Propagation
                signal_information = SignalInformation(1, path)
                signal_information = Network.propagate(self, signal_information)
                latencies.append(signal_information.latency)
                noises.append(signal_information.noise_power)
                # snrs.append(10 * np.log10(signal_information.signal_power / signal_information.noise_power))
                # snrs.append(signal_information.noise_power)
                snrs.append(1/signal_information.noise_power)
        df['path'] = paths
        df['latency'] = latencies
        df['noise'] = noises
        df['snr'] = snrs
        self._weighted_paths['path'] = n_paths
        self._weighted_paths['latency'] = latencies
        self._weighted_paths['noise'] = noises
        self._weighted_paths['snr'] = snrs

    @property
    def route_space(self):
        return self._route_space

    @property
    def static_switch_mtx(self):
        return self._static_switch_mtx

    @property
    def rejected_request(self):
        return self._rejected_request

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
            # Define also the switching_matrix for each node
            node.switching_matrix = self._static_switch_mtx[node_label]

    def propagate(self, signal):
        path = signal.path
        start_node = self.nodes[path[0]]
        propagated_signal = start_node.propagate(signal)
        return propagated_signal

    def find_best_snr(self, path_struc):  # Find path with higher snr between two node
        path_list = []
        snr_list = []
        max_path = {}
        for path in path_struc.keys():  # Retrieve all paths
            # Retrieve the row from the weighted path structure
            row_df = self._weighted_paths.loc[self._weighted_paths['path'] == path]
            row = row_df['path'].values[0]
            snr_list.append(row_df['snr'].values[0])
            path_list.append(row)
        if len(path_struc) > 1:
            path = path_list[snr_list.index(max(snr_list))]
            max_path[path] = path_struc[path]
        else:
            max_path = path_struc
        return max_path

    def find_best_latency(self, path_struc):  # Find path with lower latency between two node
        lat_list = []
        path_list = []
        min_path = {}
        for path in path_struc.keys():  # Retrieve all paths
            row_df = self._weighted_paths.loc[self._weighted_paths['path'] == path]
            row = row_df['path'].values[0]
            lat_list.append(row_df['latency'].values[0])
            path_list.append(row)
        if len(path_struc) > 1:
            path = path_list[lat_list.index(min(lat_list))]
            min_path[path] = path_struc[path]  # look the min
        else:
            min_path = path_struc
        return min_path

    def availability(self, i_node, o_node):
        """
        Control the availability of a specific request
        """
        network_availability = {}
        for path in Network.find_paths(self, i_node.label, o_node.label):  # find all possible path
            ch_free = []
            if len(path) > 2:  # the path has more than two nodes
                for i in range(len(path)-1):
                    label = path[i] + path[i + 1]  # span two nodes at time
                    if i == 0:  # first node
                        for j in range(n_ch):
                            if self.lines[label].state[j] == free:
                                ch_free.append(j)  # append all free channel
                    else:  # otherwise
                        x = self._nodes[path[i]].switching_matrix[path[i - 1]][path[i + 1]]
                        for j in ch_free:  # look only the available
                            if (self.lines[label].state[j] == occupied) or (x[j] == occupied):
                                ch_free.remove(j)  # remove if no wavelenght continuity
            else:  # Direct connection check only the availability of the line
                for i in range(n_ch):
                    if self.lines[path].state[i] == free:
                        ch_free.append(i)  # append all free channel
            if len(ch_free) > 0:
                # I can garantee wavelength continuity
                network_availability[path] = ch_free[0]  # choose the first available channel
        return network_availability

    def stream(self, conn_list, lat_snr_label):
        """
        1 - Check availability
        2 - Propagate the line, update the state
        4 - Propagate the node, update switching matrices
        5 - Update the routing space
        """
        for elem in conn_list:  # Check all connection istance
            path_ch = {}
            lightpath = None
            path = ''
            # Check availability
            possible_path = Network.availability(self, self._nodes[elem.input], self._nodes[elem.output])

            if len(possible_path) != 0:  # There is almost a free path available
                bit_rate = -1  # default value
                while (bit_rate <= 0) and (len(possible_path) > 0):
                    if lat_snr_label == 'latency':  # If check for latency
                        path_ch = Network.find_best_latency(self, possible_path)
                    elif lat_snr_label == 'snr':  # If check for snr
                        path_ch = Network.find_best_snr(self, possible_path)
                    path = list(path_ch.keys())[0]  # retrieve path
                    channel = list(path_ch.values())[0]  # retrieve channel

                    lightpath = Lightpath(1, path)  # deploy the lightpath
                    lightpath.set_channel(channel)  # set the channel
                    strategy = self._nodes[path[0]].transceiver
                    # Evaluate bit-rate
                    bit_rate = Network.calculate_bit_rate(self, lightpath, strategy)
                    # print(bit_rate)
                    if bit_rate <= 0:
                        possible_path.pop(path, channel)
                        self._rejected_request.append(path)  # consider as rejected request
                    else:
                        elem.bit_rate = bit_rate  # Assign bit rate to connection element
                        start_node = self.nodes[path[0]]
                        lightpath = start_node.propagate(lightpath)

                # update routing space
                self._route_space = update_route_space(self._route_space, self._nodes, self._lines, path)
                # restore the initial node switching matrix
                node_dic = json.load(open(file, 'r'))
                for node_label in path:
                    self._nodes[node_label].switching_matrix = node_dic[node_label]['switching_matrix']

                elem.latency = lightpath.latency
                elem.snr = 10 * np.log10(1 / lightpath.noise_power)
                # if lat_snr_label == 'latency':  # If check for latency
                #     elem.latency = lightpath.latency
                # elif lat_snr_label == 'snr':
                #     elem.snr = 10 * np.log10(1 / lightpath.noise_power)
            else:  # no free path available
                # Specific request is rejected
                self._rejected_request.append(elem)
                elem.latency = 0
                elem.snr = 0

    def calculate_bit_rate(self, lightpath, strategy):
        row_df = self._weighted_paths.loc[self._weighted_paths['path'] == lightpath.path]
        gsnr = row_df['snr'].values[0]  # Retrieve the gsnr for a specific path for the first node
        rb = 0
        rs = lightpath.rs  # symbol rate
        # 1 fixed-rate
        if strategy == 'fixed_rate':
            if gsnr >= 2*(sc.erfcinv(2*BERt)**2)*(rs/Bn):
                rb = 100e9  # Bit-rate 100Gbps
            else:
                rb = 0
        # 2 flex-rate
        if strategy == 'flex_rate':
            if gsnr < 2*(2*BERt)*(rs/Bn):
                rb = 0
            if (gsnr >= 2*(sc.erfcinv(2*BERt)**2)*(rs/Bn)) and (gsnr < (14/3)*(sc.erfcinv((3/2)*BERt)**2)*(rs/Bn)):
                rb = 100e9  # Rb = 100Gbps
            if (gsnr >= (14/3)*(sc.erfcinv((3/2)*BERt)**2)*(rs/Bn)) and (gsnr < 10*(sc.erfcinv((8/2)*BERt)**2)*(rs/Bn)):
                rb = 200e9  # Rb = 200Gbps
            if gsnr >= 10*(sc.erfcinv((8/2)*BERt)**2)*(rs/Bn):
                rb = 400e9  # Rb = 400Gbps
        # 3 shannon
        if strategy == 'shannon':
            rb = 2*rs*np.log2(1+gsnr*(Bn/rs))*1e9
        return rb

    def manage_traffic_mtx(self):
        # Create the traffic matrix
        traffic_json = json.load(open(Path(__file__).parent.parent/'Resources'/'traffic_matrix_file.json', "r"))
        for t_node in traffic_json:
            self._traffic_matrix[t_node] = traffic_json[t_node]
        # Manage connections
        inode_arr = []
        onode_arr = []
        node_ret_arr = []
        eff_n_con = 0  # count the effective number of connection
        rej_conn = 0   # count the rejected  number of connection
        print('Traffic matrix generation: ')
        for i in range(nconn_t_mtx):
            i_node = random.choice(list(self._nodes))  # Random input node
            o_node = random.choice(list(self._nodes))  # Random output node
            if self._traffic_matrix[i_node][o_node][0] >= (M*100e9):
                eff_n_con += 1
                # update traffic matrix
                self._traffic_matrix[i_node][o_node][0] = self._traffic_matrix[i_node][o_node][0] - M * 100e9
                inode_arr.append(i_node)
                onode_arr.append(o_node)
            else:
                rej_conn += 1
                print('Rejected connection ', i_node, '->', o_node, '  --  Traffic request not supported')
        print('Total rejected connection: ', rej_conn, ' over ', nconn_t_mtx)
        node_ret_arr.append(inode_arr)
        node_ret_arr.append(onode_arr)
        node_ret_arr.append(eff_n_con)
        return node_ret_arr

#############################################################


class Connection(object):
    def __init__(self, conn_dict):
        self._input = conn_dict['input']
        self._output = conn_dict['output']
        self._signal_power = conn_dict['signal_power']
        self._latency = 0.0
        self._snr = 0.0
        self._bit_rate = 0

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

    @property
    def bit_rate(self):
        return self._bit_rate

    @bit_rate.setter
    def bit_rate(self, bit_rate):
        self._bit_rate = bit_rate


#############################################################
