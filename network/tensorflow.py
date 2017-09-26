from __future__ import absolute_import

import os
import re
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from frozendict import FrozenOrderedDict

from . import Network as BaseNetwork
from .exceptions import ParsingError
from .layers.tensorflow_layers import TensorFlowLayer as Layer
from .layers.tensorflow_layers import TensorFlowNeuralLayer as NeuralLayer
from .layers.tensorflow_layers import TensorFlowStridingLayer as StridingLayer
from .layers.tensorflow_layers import TensorFlowDense as Dense
from .layers.tensorflow_layers import TensorFlowConv2D as Conv2D
from .layers.tensorflow_layers import TensorFlowMaxPooling2D as MaxPooling2D
from .layers.tensorflow_layers import TensorFlowDropout as Dropout
from .layers.tensorflow_layers import TensorFlowFlatten as Flatten

class NonMatchingLayerDefinition(Exception):
    pass

class Network(BaseNetwork):
    """Network interface to TensorFlow."""

    _OPERATION_TYPES = {
        'activation_functions': {'Relu',
                                 'Relu6',
                                 'Relu',
                                 'Elu',
                                 'Softplus',
                                 'Softsign',
                                 'Sigmoid',
                                 'Tanh',
                                 'Softmax' # Should not really count as an activation function, as it is computed
                                           # base on the whole layer inputs. It is convenient though.
                                 },
        'classification': [],
        'convolution': {'Conv2D',
                        'DepthwiseConv2dNative',
                        'DepthwiseConv2dNative',
                        'Conv2DBackpropFilter',
                        'Conv2DBackpropInput',
                        'DepthwiseConv2dNativeBackpropFilter',
                        'DepthwiseConv2dNativeBackpropInput'},
        'embeddings': set(),
        'evaluation': {'TopKV2'},
        'losses': {'L2Loss'},
        'morphological_filtering': set(),
        'normalization': {'LRN'},
        'pooling': {'AvgPool',
                    'MaxPool',
                    'MaxPoolWithArgmax',
                    'FractionalAvgPool',
                    'FractionalMaxPool'},
        'recurrent_neural_networks': set()
    }



    _LAYER_TYPES_TO_CLASSES = {
        'Conv2D': Conv2D,
        'Dense': Dense,
        'MaxPooling2D': MaxPooling2D,
        'Dropout': Dropout,
        'Flatten': Flatten
    }

    _LAYER_DEFS = {
        'Conv2D': [
            _OPERATION_TYPES['convolution'],
            {'Add', 'BiasAdd'},
            _OPERATION_TYPES['activation_functions']
        ],
        'MaxPooling2D': {'MaxPool'},
        'Dense': [
            {'MatMul'},
            {'Add', 'BiasAdd'},
            _OPERATION_TYPES['activation_functions'] | {'?'},
        ],
        'Flatten': ['Shape',
                    'Const',
                    'Const',
                    'Slice',
                    'Const',
                    'Const',
                    'Slice',
                    'Const',
                    'Prod',
                    'Const',
                    'ExpandDims',
                    'Const',
                    'ConcatV2',
                    'Reshape'],
        'Dropout': ['Shape',
                    'Const',
                    'Const',
                    'RandomUniform',
                    'Sub',
                    'Mul',
                    'Add',
                    'Add',
                    'Floor',
                    'RealDiv',
                    'Mul']
    }

    keras_name_regex = re.compile(r'(.)([A-Z][a-z0-9]+)')

    def __init__(self, **kwargs):
        checkpoint = kwargs.get('checkpoint', None)
        sess = kwargs.get('session', None)
        if checkpoint is not None and sess is None:
            # Restore the tensorflow model from a file.
            self._sess = tf.Session()
            saver = tf.train.import_meta_graph(checkpoint + '.meta')
            model_dir = os.path.split(checkpoint)[0]
            saver.restore(self._sess, tf.train.latest_checkpoint(model_dir))
        elif checkpoint is None and sess is not None:
            # Just store the session since the model is already there.
            self._sess = sess
        # TensorFlow uses channels last as data format by default. This can however be changed
        # by the user.
        # TODO make this more flexible.
        kwargs['data_format'] = 'channels_last'
        super().__init__(**kwargs)

    def _create_layer_dict(self) -> FrozenOrderedDict:
        """Try to find the sequences in operations of the graph that
        match the idea of a layer.

        Returns
        -------
        A mapping of layer_ids to layer objects.
        """
        layer_dict = OrderedDict()
        layer_counts = {layer_type: 0 for layer_type in self._LAYER_DEFS.keys()}
        op_idx = 0
        while op_idx < len(self._sess.graph.get_operations()):

            for layer_type, layer_def in self._LAYER_DEFS.items():
                try:
                    matching_ops = self._match_layer_def(op_idx, layer_def)
                    # Increment count for layer type.
                    layer_counts[layer_type] += 1
                    layer_name = '{}_{}'.format(self._to_keras_name(layer_type), layer_counts[layer_type])
                    layer_dict[layer_name] = self._LAYER_TYPES_TO_CLASSES[layer_type](self, matching_ops)
                    # If the layer definition was successfully
                    # matched, advance the number of ops that were
                    # matched. Don't try to match another layer
                    # definition at the same op by breaking the for
                    # loop.
                    op_idx += len(layer_def) - 1
                    break
                except NonMatchingLayerDefinition:
                    continue
            # Try to match at the next op.
            op_idx += 1

        if not layer_dict:
            raise ParsingError('Could not find any layers in TensorFlow graph.')

        layer_dict = FrozenOrderedDict(layer_dict)
        return layer_dict

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
        matched_ops = []
        for i, op_group in enumerate(layer_def):
            if ops[op_idx + i].type in op_group:
                matched_ops.append(ops[op_idx + i])
                continue
            # The operation at this position is optional. This is only the case if we are dealing with a linear
            # activation function. In this case the last MatMul operation should just be duplicated as
            # the activation.
            elif '?' in op_group:
                matched_ops.append(ops[op_idx + i - 1])
                continue
            else:
                raise NonMatchingLayerDefinition
        # If all operations could be matched, return the respective operations.
        return matched_ops

    def _to_keras_name(self, name):
        return self.keras_name_regex.sub(r'\1_\2', name).lower()

    def _compute_activations(self, layer_ids: list, input_samples: np.ndarray) -> list:
        """
        Parameters
        ----------
        layer_ids
        input_samples

        Returns
        -------

        """
        # Get the tensors that actually hold the activations.
        activation_tensors = []
        for layer_id in layer_ids:
            activation_tensors.append(self.layer_dict[layer_id].activation_tensor)
        print('activation tensors', activation_tensors)
        return self._feed_input(activation_tensors, input_samples)

    def _compute_net_input(self, layer_ids: list, input_samples: np.ndarray):
        net_input_tensors = []
        for layer_id in layer_ids:
            net_input_tensors.append(self.layer_dict[layer_id].net_input_tensor)
        print('net tensors', net_input_tensors)

        return self._feed_input(net_input_tensors, input_samples)

    def _feed_input(self, fetches: list, input_samples: np.ndarray):
        network_input_tensor = self._sess.graph.get_operations()[0].outputs[0] # Assuming the first op is the input.
        return self._sess.run(fetches=fetches, feed_dict={network_input_tensor: input_samples})
