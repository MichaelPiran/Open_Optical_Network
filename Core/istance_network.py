"""
Create an istance of the Network
"""
from Core.elements import Network
from Core.parameters import *

network = Network(file)

network.connect()
network.set_weighted_paths()
