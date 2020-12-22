# Global variables
from pathlib import Path

n_ch = 10  # number of channel
free = 1  # the channel is free
occupied = 0  # the channel is occupied
lat_snr = 'latency'  # run stream() for latency or snr

# Read the json file input
root = Path(__file__).parent.parent
folder = root/'Resources'
# file = folder/'nodes.json'  # Obsolete
file = folder/'nodes_full.json'  # All path available in the switching matrices
# file = folder/'nodes_not_full.json'  # Not all path available in the switching matrices
