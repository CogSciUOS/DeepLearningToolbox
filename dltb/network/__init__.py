"""Abstract base classes for working withs networks.
"""
from .exception import NetworkParsingError
from .base import Layerlike, as_layer, layer_key
from .layer import Layer
from .layer import StridingLayer, Dense, Conv2D, MaxPooling2D
from .layer import Dropout, Flatten
from .network import Network, Networklike, network_key, as_network
from .image import ImageNetwork
from .classifier import Classifier
from .autoencoder import Autoencoder, VariationalAutoencoder
