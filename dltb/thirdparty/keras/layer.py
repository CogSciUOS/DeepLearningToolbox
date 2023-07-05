from typing import List
from packaging import version
from . import keras

from dltb.network import Network, layer, Layer as BaseLayer

if version.parse(keras.__version__) >= version.parse('2.0.0'):
    from keras.layers import Conv2D
else:
    from keras.layers import Convolution2D

def conv_2d(filters, kernel_shape, strides, padding, input_shape=None):
    """
    Defines the right convolutional layer according to the
    version of Keras that is installed.
    :param filters: (required integer) the dimensionality of the output
        space (i.e. the number output of filters in the convolution)
    :param kernel_shape: (required tuple or list of 2 integers) specifies
        the strides of the convolution along the width and
        height.
    :param padding: (required string) can be either 'valid' (no padding around
        input or feature map) or 'same' (pad to ensure that the
        output feature map size is identical to the layer input)
    :param input_shape: (optional) give input shape if this is the first
        layer of the model
    :return: the Keras layer
    """
    if version.parse(keras.__version__) >= version.parse('2.0.0'):
        if input_shape is not None:
            return Conv2D(filters=filters, kernel_size=kernel_shape,
                          strides=strides, padding=padding,
                          input_shape=input_shape)
        else:
            return Conv2D(filters=filters, kernel_size=kernel_shape,
                          strides=strides, padding=padding)
    else:
        if input_shape is not None:
            return Convolution2D(filters, kernel_shape[0], kernel_shape[1],
                                 subsample=strides, border_mode=padding,
                                 input_shape=input_shape)
        else:
            return Convolution2D(filters, kernel_shape[0], kernel_shape[1],
                                 subsample=strides, border_mode=padding)


class Layer(BaseLayer):
    """A keras :py:class:`Layer` implements the abstract layer class
    based on keras layer object.

    A `py:class:`Layer` may bundle multiple keras layer objects of the
    unterlying Keras model, e.g., when convolution and activation
    function are realized be separate Keras layers.
    """

    def __init__(self, network: Network,
                 keras_layer_objs: List[keras.layers.Layer]) -> None:
        super().__init__(network)
        if len(keras_layer_objs) > 6:  # FIXME[hack]: was 2 not 6
            raise ValueError('A layer should at most contain 2 keras layer'
                             'objects: normal plus activation')
        self._keras_layer_objs = keras_layer_objs

    @property
    def input_shape(self):
        """The input shape is obtained from the first Keras layer bundled
        in this `py:class:`Layer`.
        """
        keras_layer = self._keras_layer_objs[0]
        input_shape = keras_layer.input_shape
        # FIXME[hack]: keray_layer.input_shape may be a list of tuples
        # (at least for ResNet50). But why? - I could not find an
        # API documentation for Layer.input_shape
        # return self._keras_layer_objs[0].input_shape
        # if isinstance(input_shape, list):
        if isinstance(keras_layer, keras.layers.InputLayer):
            print(f"{type(self._keras_layer_objs[0])}:",
                  [layer.input_shape for layer in self._keras_layer_objs])
            input_shape = input_shape[0]
        return input_shape

    @property
    def output_shape(self):
        """The output shape is obtained from the last Keras layer bundled
        in this `py:class:`Layer`.
        """
        return self._keras_layer_objs[-1].output_shape


class NeuralLayer(Layer, layer.NeuralLayer):
    """A keras `NeuralLayer` consists of one or two keras.layers.Layer
    objects.
    """

    @property
    def parameters(self):
        return self._keras_layer_objs[0].get_weights()

    @property
    def num_parameters(self):
        return self._keras_layer_objs[0].count_params()

    @property
    def weights(self):
        return self._keras_layer_objs[0].get_weights()[0]

    @property
    def bias(self):
        return self._keras_layer_objs[0].get_weights()[1]


class StridingLayer(Layer, layer.StridingLayer):

    @property
    def strides(self):
        return self._keras_layer_objs[0].strides

    @property
    def padding(self):
        return self._keras_layer_objs[0].padding


class Dense(NeuralLayer, layer.Dense):
    pass


class Conv2D(NeuralLayer, StridingLayer, layer.Conv2D):

    @property
    def kernel_size(self):
        return self._keras_layer_objs[0].kernel_size

    @property
    def filters(self):
        return self._keras_layer_objs[0].filters


class MaxPooling2D(StridingLayer, layer.MaxPooling2D):

    @property
    def pool_size(self):
        return self._keras_layer_objs[0].pool_size


class Dropout(Layer, layer.Dropout):
    pass


class Flatten(Layer, layer.Flatten):
    pass

