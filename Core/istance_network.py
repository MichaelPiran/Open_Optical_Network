"""
Istance of the Network
"""
import pandas as pd
import numpy as np
from pathlib import Path
from Core.elements import Network, SignalInformation

root = Path(__file__).parent.parent
folder = root/'Resources'
file = folder/'nodes.json'
network = Network(file)

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
        snrs.append(10 * np.log10(signal_information.signal_power / signal_information.noise_power))
df['path'] = paths
df['latency'] = latencies
df['noise'] = noises
df['snr'] = snrs
network.set_weighted_paths(n_paths, latencies, noises, snrs)  # Instance weighted graph
