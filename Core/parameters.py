# Global variables
from pathlib import Path

n_ch = 10  # number of channel
free = 1  # the channel is free
occupied = 0  # the channel is occupied
lat_snr = 'snr'  # run stream() for latency or snr

# Read the json file input
root = Path(__file__).parent.parent
folder = root/'Resources'
# file = folder/'nodes.json'  # Obsolete
# file = folder/'nodes_full.json'  # All path available in the switching matrices
# file = folder/'nodes_not_full.json'  # Not all path available in the switching matrices
# file = folder/'nodes_full_fixed_rate.json'
# file = folder/'nodes_full_flex_rate.json'
file = folder/'nodes_full_shannon.json'

Rs = 32*(10**9)  # Symbol-rate of the lightpath : 32GHz
Bn = 12.5*(10**9)  # Noise bandwidth : 12.5GHz
BERt = 10e-3  # maximun error rate
freq = 193.414e12  # C-band center
Bopt = 4.4e12  # C-band width (195.6-191.2)THz
Df = 50e9  # Bopt/n_ch
