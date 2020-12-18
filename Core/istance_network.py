"""
Create an istance of the Network
"""
from pathlib import Path
from Core.elements import Network

# Read the json file
root = Path(__file__).parent.parent
folder = root/'Resources'
# file = folder/'nodes.json'  # Obsolete
file = folder/'nodes_full.json'  # All path available in the switching matrices
# file = folder/'nodes_not_full.json'  # Not all path available in the switching matrices
network = Network(file)

network.connect()
network.set_weighted_paths()
