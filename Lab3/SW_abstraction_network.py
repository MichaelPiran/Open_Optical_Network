class Signal_information(object):
    def __init__(self, power, path):
        self._signal_power = power
        self._noise_power = 0.0
        self._latency = 0.0
        self._path = path #[A, B, C, D, E, F]
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

    def add_noise(self, noise): #Update the noise power
        self.noise_power += noise
    def add_latency(self, latency): #Update the latency
        self.latency += latency
    def next(self): #Update the path
        self.path = self.path [1:]
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

    def propagate (self, signal_information):
        path = signal_information.path
        if len(path) >1 :
            line_label = path [:2]
            line = self.successive [line_label]
            signal_information.next()
            signal_information = line.propagate(signal_information)
        return signal_information
####################################################################
class Line(object):
    def __init__(self, line_dict):
        self._label = line_dict['label']
        self._length = line_dict['length']
        self._successive = {}
    @property
    def label(self):
        return self._label
    @property
    def length(self):
        return self._length
    @property
    def successive(self):
        return self._successive

    def latency_generation(self):
        latency = self.latency /(c * 2/3)
        return latency

    def noise_generation(self, signal_power):
        noise =  1e-3 * signal_power * self.length
        return noise
    def propagate(self, signal_information):
        latency = self.latency_generation()
        signal_information.add_latency(latency)

        signal_power = signal_information.signal_power
        noise = self.noise_generation(signal_power)
        signal_information.add_noise(noise)

        node = self.successive [signal_information.path[0]]
        signal_information = node.propagate(signal_information)
        return signal_information

def main():
    #error

if __name__ == "__main__":
    main()
