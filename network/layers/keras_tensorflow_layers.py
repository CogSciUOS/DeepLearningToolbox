# FIXME[old]: currently only used by
#  - ./network/tests/test_keras_tensorflow_network.py

from . import keras_layers

class KerasTensorFlowLayerMixin:

    @property
    def input(self):
        return self._keras_layer_objs[0].input

    @property
    def output(self):
        return self._keras_layer_objs[-1].output


# Mix in acess to input and output into every KerasLayer class.

class KerasTensorFlowLayer(keras_layers.KerasLayer, KerasTensorFlowLayerMixin):
    pass

class KerasTensorFlowNeuralLayer(keras_layers.KerasNeuralLayer, KerasTensorFlowLayerMixin):
    pass

class KerasTensorFlowStridingLayer(keras_layers.KerasStridingLayer, KerasTensorFlowLayerMixin):
    pass


class KerasTensorFlowDense(keras_layers.KerasDense, KerasTensorFlowLayerMixin):
    pass

class KerasTensorFlowConv2D(keras_layers.KerasConv2D, KerasTensorFlowLayerMixin):
   pass


class KerasTensorFlowMaxPooling2D(keras_layers.KerasMaxPooling2D, KerasTensorFlowLayerMixin):
   pass


class KerasTensorFlowDropout(keras_layers.KerasDropout, KerasTensorFlowLayerMixin):
    pass

class KerasTensorFlowFlatten(keras_layers.KerasFlatten, KerasTensorFlowLayerMixin):
    pass
