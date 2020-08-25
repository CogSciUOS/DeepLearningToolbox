from .network import Network
from .network import NetworkController as Controller, NetworkView as View
from .network import (Classifier, Autoencoder, VariationalAutoencoder, AutoencoderController)
from .layers.layers import Layer, NeuralLayer, StridingLayer, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from .resize import *


Network.register_key('alexnet-tf', 'network.tensorflow', 'Alexnet')
# FIXME[old]:                     extend=Classifier, scheme='ImageNet')

Network.register_key('resnet-keras', 'network.keras',
                     'ApplicationsNetwork', model='ResNet50')

Network.register_key('resnet-keras-tf', 'network.keras_tensorflow',
                     'ApplicationsNetwork', model='ResNet50')

Network.register_key('mnist-keras-tf', 'models.example_keras_advex_mnist',
                     'KerasMnistClassifier')

Network.register_key('resnet-torch', 'network.torch', 'DemoResnetNetwork')

# FIXME[hack]: hack_new_model
#  if self.data is None:
#    self.hack_load_mnist()
#
# original_dim = self.data[0][0].size
original_dim = 28 * 28
# print(f"Hack 1: new model with original_dim={original_dim}")
#intermediate_dim = 512
#latent_dim = 2
Network.register_key('mnist-vae', 'models.example_keras_vae_mnist',
                     'KerasAutoencoder', original_dim)


# FIXME[todo]: torch
#Network.register_key('mnist-keras', 'models.example_keras_advex_mnist', '?')

# FIXME[todo]: there are also some stored models:
#  - models/example_caffe_network_deploy.prototxt
#  - models/example_keras_mnist_model.h5  <- network.examples.keras
#  - models/example_tf_alexnet/
#  - models/example_tf_mnist_model/
#  - models/example_torch_mnist_model.pth
#  - models/mnist.caffemodel

# FIXME[todo]: there are also examples in network/examples.py:
#  - keras  -> models/example_keras_mnist_model.h5
#  - torch
