from . import layers


class TensorFlowLayer(layers.Layer):

    def __init__(self, network, ops):
        super().__init__(network)
        self._ops = ops

    @property
    def type(self):
        return self._type

    # Assume that there is only one input/output that matters.
    @property
    def input_shape(self):
        return tuple(self._ops[0].inputs[0].shape.as_list())

    @property
    def output_shape(self):
        return tuple(self._ops[-1].outputs[0].shape.as_list())


class TensorFlowNeuralLayer(TensorFlowLayer, layers.NeuralLayer):

    @property
    def parameters(self):
        return (self.weights, self.bias)

    @property
    def num_parameters(self):
        return self.weights.size + self.bias.size

    @property
    def weights(self):
        return self._network._sess.run(self.weight_tensor)

    @property
    def bias(self):
        return self._network._sess.run(self.bias_tensor)

    @property
    def activation_tensor(self):
        """The tensor that contains the activations of the layer."""
        # For now assume that the last operation is the activation.
        # Maybe differentiate with subclasses later.
        return self._ops[-1].outputs[0]

    @property
    def net_input_tensor(self):
        return self._ops[-2].outputs[0]

    @property
    def weight_tensor(self):
        # The last input of the first operation should correspond to the weights.
        return self._ops[0].inputs[-1]

    @property
    def bias_tensor(self):
        # The last input of the second operation should correspond to the bias.
        return self._ops[1].inputs[-1]


class TensorFlowStridingLayer(TensorFlowLayer, layers.StridingLayer):
    @property
    def strides(self):
        striding_op = self._ops[0]
        strides = striding_op.node_def.attr['strides']
        return (strides.list.i[1], strides.list.i[2])

    @property
    def padding(self):
        striding_op = self._ops[0]
        return striding_op.node_def.attr['padding'].s.decode('utf8')


class TensorFlowDense(TensorFlowNeuralLayer, layers.Dense):
    pass


class TensorFlowConv2D(TensorFlowNeuralLayer, TensorFlowStridingLayer, layers.Conv2D):

    @property
    def kernel_size(self):
        # The kernel size is not directly saved in the `node_def`, but has to be read from
        # the shape of the weights.

        return tuple(self.weight_tensor.shape.as_list()[:2])

    @property
    def filters(self):
        return self.output_shape[-1]

class TensorFlowMaxPooling2D(TensorFlowStridingLayer, layers.MaxPooling2D):

    @property
    def pool_size(self):
        kernel = self._ops[0].node_def.attr['ksize']
        return (kernel.list.i[1], kernel.list.i[2])


class TensorFlowDropout(TensorFlowLayer, layers.Dropout):
    pass

class TensorFlowFlatten(TensorFlowLayer, layers.Flatten):
    pass
