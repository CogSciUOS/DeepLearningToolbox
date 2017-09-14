import os
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from network.network import BaseNetwork
from network.exceptions import ParsingError

_OPERATION_TYPES = {
        'activation_functions': ['Relu',
                                  'Relu6',
                                  'Relu',
                                  'Elu',
                                  'Softplus',
                                  'Softsign',
                                  'Sigmoid',
                                  'Tanh'],
        'classification': [],
        'convolution': ['Conv2D',
                     'DepthwiseConv2dNative',
                     'DepthwiseConv2dNative',
                     'Conv2DBackpropFilter',
                     'Conv2DBackpropInput',
                     'DepthwiseConv2dNativeBackpropFilter',
                     'DepthwiseConv2dNativeBackpropInput'],
        'embeddings': [],
        'evaluation': ['TopKV2'],
        'losses': ['L2Loss'],
        'morphological_filtering': [],
        'normalization': ['LRN'],
        'pooling': ['AvgPool',
                 'MaxPool',
                 'MaxPoolWithArgmax',
                 'FractionalAvgPool',
                 'FractionalMaxPool'],
        'recurrent_neural_networks': []
    }


class NonMatchingLayerDefinition(Exception):
    pass


class TensorFlowLayer:


    def __init__(self, layer_type, ops):
        self._type = layer_type
        self._ops = ops

    @property
    def type(self):
        return self._type

    # Assume that there is only one input/output that matters.
    @property
    def input_shape(self):
        return tuple(self._ops[0].inputs[0].get_shape().as_list())

    @property
    def output_shape(self):
        return tuple(self._ops[-1].outputs[0].get_shape().as_list())

    @property
    def activation_tensor(self):
        """The tensor that contains the activations of the layer."""
        # For now assume that the last operation is the activation.
        # Maybe differentiate with subclasses later.
        if self._ops[-1].type in _OPERATION_TYPES['activation_functions']:
            return self.ops[-1].outputs[0]
        else:
            raise ValueError('No activation function')

    @property
    def net_input_tensor(self):
        if self.ops[-3].type == 'MatMul':
            return self.ops[-3].outputs[0]

    @property
    def weight_tensor(self):
        # The last input of the first operation should correspond to the weights.
        return self._ops[0].inputs[-1]




class TensorFlowNetwork(BaseNetwork):
    """Network interface to TensorFlow."""


    _LAYER_DEFS = {
        'Conv2D': [
            _OPERATION_TYPES['convolution'],
            ['Add', 'BiasAdd'],
            _OPERATION_TYPES['activation_functions']
        ],
        'MaxPooling2D': ['MaxPool'],
        'Dense': [
            ['MatMul'],
            ['Add', 'BiasAdd'],
            _OPERATION_TYPES['activation_functions']
        ]
    }

    def __init__(self, model_file: str=None, sess: tf.Session=None):
        if model_file is not None and sess is None:
            # Restore the tensorflow model from a file.
            self._sess = tf.Session()
            saver = tf.train.import_meta_graph(os.path.join(model_file, '.meta'))
            saver.restore(sess, model_file)
        elif model_file is None and sess is not None:
            # Just store the session since the model is already there.
            self._sess = sess

        # Try to parse layers.
        self._layers = self._create_layer_dict()

    @property
    def layer_ids(self) -> list:
        """Get list of layer ids.
        Returns
        -------

        """
        return list(self._layers.keys())

    def get_layer_input_shape(self, layer_id) -> tuple:
        """
        Give the shape of the input of the given layer.

        Parameters
        ----------
        layer_id

        Returns
        -------
        """
        return tuple(self._layers[0].inputs[0].get_shape.as_list())

    def get_layer_output_shape(self, layer_id) -> tuple:
        """
        Give the shape of the output of the given layer.

        Parameters
        ----------
        layer_id

        Returns
        -------

        """
        return tuple(self._layers[layer_id].output)


    def get_activations(self, layer_ids, input_samples: np.ndarray) -> list:
        """
        Gives activations values of the loaded_network/model
        for a given layername and an input (inputsample).
        Parameters
        ----------
        layer_ids
        input_samples

        Returns
        -------

        """
        layer_ids, input_samples = super().get_activations(layer_ids, input_samples)
        # Get the tensors that actually hold the activations.
        activation_tensors = []
        for layer_id in layer_ids:
            activation_tensors.append(self._layers[layer_id].activation_tensor)
        network_input = self._sess.graph.get_operations()[0].values()
        return self._sess.run(fetches=activation_tensors, feed_dict={network_input: input_samples})



    def get_layer_weights(self, layer_id) -> np.ndarray:
        """
        Returns weights INCOMING to the
        layer (layername) of the model
        shape of the weights variable should be
        coherent with the get_layer_output_shape function.
        Parameters
        ----------
        layer_id :
             An identifier for a layer.

        Returns
        -------
        ndarray
            Weights of the layer.

        """
        return self._sess.run(self._layers[layer_id])

    def _create_layer_dict(self) -> OrderedDict:
        """Try to find the sequences in operations of the graph that
        match the idea of a layer.

        Returns
        -------
        A mapping of layer_ids to layer objects.
        """
        layers = OrderedDict()
        layer_counts = {layer_type: 0 for layer_type in self._LAYER_DEFS.keys()}
        for op_idx in range(len(self._sess.graph.get_operations())):
            try:
                for layer_type, layer_def in self._LAYER_DEFS.items():
                    matching_ops = self._match_layer_def(op_idx, layer_def)
                    # Increment count for layer type.
                    layer_counts[layer_type] += 1
                    layer_name = '{}_{}'.format(layer_type, layer_counts[layer_type]).lower()
                    layers[layer_name] = TensorFlowLayer(layer_type, matching_ops)
            except NonMatchingLayerDefinition:
                continue

        if not layers:
            raise ParsingError('Could not find any layers in TensorFlow graph.')

        return layers


    def _match_layer_def(self, op_idx: int, layer_def: dict) -> list:
        """Check whether the layer definition match the operations starting from a
        certain index.

        Parameters
        ----------
        op_idx
        layer_def

        Returns
        -------

        """
        ops = self._sess.graph.get_operations()
        num_matched_ops = 0
        num_ops_to_match = len(layer_def)
        for i, op_group in enumerate(layer_def):
            if ops[i] in op_group:
                continue
            else:
                raise NonMatchingLayerDefinition
        # If all operations could be matched, return the respective operations.
        return ops[op_idx - len(layer_def): op_idx]


