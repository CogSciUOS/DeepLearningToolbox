import layers.layers as layers

class KerasLayer(layers.Layer):

    def __init__(self, network, keras_layer_objs):
        super().__init__(network)
        if len(keras_layer_objs) > 2:
            raise ValueError('A layer should at most contain 2 keras layer'
                             'objects: normal plus activation')
        self._keras_layer_objs = keras_layer_objs


    @property
    def input_shape(self):
        return self._keras_layer_objs[0].input_shape

    @property
    def output_shape(self):
        return self._keras_layer_objs[-1].output_shape

class KerasNeuralLayer(KerasLayer, layers.NeuralLayer):

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

class KerasStridingLayer(KerasLayer, layers.StridingLayer):
    @property
    def strides(self):
        return self._keras_layer_objs[0].strides

    @property
    def padding(self):
        return self._keras_layer_objs[0].padding

class KerasDense(KerasNeuralLayer, layers.Dense):
    pass

class KerasConv2D(KerasNeuralLayer, KerasStridingLayer, layers.Conv2D):


    @property
    def kernel_size(self):
        return self._keras_layer_objs[0].kernel_size


class KerasMaxPooling2D(KerasStridingLayer, layers.MaxPooling2D):

    @property
    def pool_size(self):
        return self._keras_layer_objs[0].pool_size


class KerasDropout(KerasLayer, layers.Dropout):
    pass

class KerasFlatten(KerasLayer, layers.Flatten):
    pass