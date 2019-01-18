from __future__ import absolute_import

import os
import re

from collections import OrderedDict
from frozendict import FrozenOrderedDict

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import util

import numpy as np
import tensorflow as tf

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

class TensorflowException(Exception):
    pass


class Network(BaseNetwork):
    """Network interface to TensorFlow.


    _graph: tf.Graph
        The internal representation of the computational graph.
        Can be initialized from the graph_def by calling
        `tf.import_graph_def(graph_def, name='')`
        May be modified by for special needs, e.g. by adding nodes
        (operations).

    _session: tf.Session
        The session running the graph. Before any computation can be
        perfomed, a session has to be initialized.
    """

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
        """Initialize a TensorFlow network.

        Parameters
        ----------
        checkpoint: str
        session: tf.Session

        graph_def: tf.GraphDef
            A Protobuf represetnation of the model. This representation
            is immutable. One may however use it to create new GraphDefs
            by calling methods from `tf.graph_util`.
    
            A GraphDef can be loaded from a protobuf file by calling
              graph_def = tf.GraphDef()
              graph_def.ParseFromString(f.read())


        Returns
        -------
        A mapping of layer_ids to layer objects.


        TensorFlow provides two ways to store models: checkpoints
        and SavedModel.
        """

        self._graph = None
        self._session = None

        if 'graph_def' in kwargs:
            self._init_from_graph_def(kwargs['graph_def'])
        elif 'checkpoint' in kwargs:
            self._init_from_checkpoint(kwargs['checkpoint'])
        elif 'session' in kwargs:
            self._init_from_session(kwargs['session'])

        # TensorFlow uses channels last as data format by
        # default. This can however be changed by the user.
        # FIXME[todo]: make this more flexible.
        
        kwargs['data_format'] = 'channels_last'
        super().__init__(**kwargs)

    def __del__(self):
        self._offline()

    def _init_from_graph_def(self, graph_def: tf.GraphDef):
        """Initialize this tensorflow.Network from a TensorFlow
        Graph Definition.
        """
        logger.info("Initialize TensorflowNetwork from GraphDef")
        self._offline()
        self._graph = tf.Graph()
        with self._graph.as_default():
            tf.import_graph_def(graph_def, name='')


    def _init_from_checkpoint(self, checkpoint: str):
        """Restore the tensorflow model from checkpoint files.
      
        These files consists basically of two parts:
        (a) The meta graph ({checkpoint}.meta):
            a protocol buffer containing the complete Tensorflow
            graph, i.e. all variables, operations, collections, etc.
        (b) The actual checkpoint files
        (b1) {checkpoint}.data-00000-of-00001
             a binary file containing all weights, biases,
             gradients, etc.
        (b2) {checkpoint}.index
        (b3) a textfile called 'checkpoint' keeping record of
             the latest checkpoint files saved

        Arguments
        ---------
        checkpoint: str
            Name identifying a Checkpoint file.

        """
        logger.info(f"Initialize TensorflowNetwork from Checkpoint ({checkpoint})")
        self._graph = tf.Graph()
        self._online()
        with self._graph.as_default():
            saver = tf.train.import_meta_graph(checkpoint + '.meta',
                                               clear_devices=True)
            model_dir = os.path.dirname(checkpoint)
            # the following can fail (for different reasons):
            # - not enough memory on session device
            # - checkpoint files have been corrupted
            # FIXME[todo]: we should raise some exception here!
            saver.restore(self._session, tf.train.latest_checkpoint(model_dir))

    def _init_from_session(self, session: tf.Session):
        """Initialize this :py:class:`Network` from a TensorFlow session.
        This will use the given session to run the model, that is the
        Network will immediatly be online (no neeed to call
        :py:meth:`Network._online` explicitly.
        """
        logger.info("Initialize TensorflowNetwork from Session")
        self._offline()
        self._session = session
        self._graph = session.graph

    def _online(self) -> None:
        if self._session is None:
            logger.info("online -> starting tf.Session")
            # tf_config = tf.ConfigProto(log_device_placement=True)
            if util.use_cpu:
                tf_config = tf.ConfigProto(device_count = {'GPU': 0})
            else:
                tf_config = tf.ConfigProto()
            self._session = tf.Session(graph = self._graph, config=tf_config)


    def _offline(self) -> None:
        """Put this model in offline mode. In offline mode, no
        inference will be possible.

        In TensorFlow offline mode means that no tf.Session is
        available. The tf.Graph may however still be present.
        """
        if self._session is not None:
            self._session.close()
            self._session = None

    def _create_layer_dict(self) -> FrozenOrderedDict:
        """Try to find the sequences in operations of the graph that
        match the idea of a layer.

        Returns
        -------
        A mapping of layer_ids to layer objects.
        """
        layer_dict = OrderedDict()
        layer_counts = {
            layer_type: 0 for layer_type in self._LAYER_DEFS.keys()
        }

        graph_def = self._graph.as_graph_def()
        
        # Output information about the underlying graph:
        logger.debug(f"Network: {self.get_id()} ({type(self)}):") 
        for i, tensor in enumerate(graph_def.node):
            logger.debug(f"  {i}) {tensor.name}: {type(tensor)}")
            #logger.debug(tf.get_default_graph().get_tensor_by_name(tensor.name+":0")[0])
        logger.debug(f"debug: Layer dict:") 


        ops = self._graph.get_operations()
        op_idx = 0
        self._input_placeholder = None


        while op_idx < len(ops):
            logger.debug(f"debug:  op-{op_idx}: "
                         # "{type(ops[op_idx])}, "
                         # always <class 'tensorflow.python.framework.ops.Operation'>
                         f"{ops[op_idx].type} with name '{ops[op_idx].name}', "
                         f"{ops[op_idx].values()}")
                
            if self._input_placeholder is None:
                if ops[op_idx].type == 'Placeholder':
                    self._input_placeholder = ops[op_idx]
                op_idx += 1
                continue

            for layer_type, layer_def in self._LAYER_DEFS.items():
                try:
                    matching_ops = self._match_layer_def(op_idx, layer_def)
                    # Increment count for layer type.
                    layer_counts[layer_type] += 1
                    layer_name = '{}_{}'.format(self._to_keras_name(layer_type), layer_counts[layer_type])
                    layer_dict[layer_name] = self._LAYER_TYPES_TO_CLASSES[layer_type](self, matching_ops)
                    logger.debug(f"debug:   ** {layer_name}"
                                 f" => {type(layer_dict[layer_name])}")
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
        """Check whether the layer definition match the operations
        starting from a certain index.

        Parameters
        ----------
        op_idx
        layer_def

        Returns
        -------

        """
        ops = self._graph.get_operations()
        matched_ops = []
        for i, op_group in enumerate(layer_def):
            if ops[op_idx + i].type in op_group:
                matched_ops.append(ops[op_idx + i])
                continue
            # The operation at this position is optional. This is only
            # the case if we are dealing with a linear activation
            # function. In this case the last MatMul operation should
            # just be duplicated as the activation.
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
        return self._feed_input(activation_tensors, input_samples)

    def _compute_net_input(self, layer_ids: list, input_samples: np.ndarray):
        net_input_tensors = []
        for layer_id in layer_ids:
            net_input_tensors.append(self.layer_dict[layer_id].net_input_tensor)

        return self._feed_input(net_input_tensors, input_samples)

    def get_input_tensor(self, include_batch=False):
        """Determine the input node of the network.
        This should be a Placeholder, that can be used to feed the network.

        Returns
        -------
        The tf.Placeholder object representing the network input.
        """
        tensor = self._input_placeholder.outputs
        if not include_batch:
            tensor = tensor[0]
        return tensor

    def get_output_tensor(self, pre_activation=False, include_batch=False):
        """Determine the output node of the network.

        Returns
        -------
        The tf.Placeholder object representing the network input.
        """
        output_layer = self.layer_dict[self.output_layer_id()]
        tensor = (output_layer.net_input_tensor if pre_activation else
                  output_layer.activation_tensor)
        if not include_batch:
            tensor = tensor[0]
        return tensor

    def _feed_input(self, fetches: list, input_samples: np.ndarray):
        if self._session is None:
            raise TensorflowException(f"{self.id()} was not prepared.")
        input = self.get_input_tensor()
        return self._session.run(fetches=fetches,
                                 feed_dict={input: input_samples})

    def get_input_shape(self, include_batch = True) -> tuple:
        """Get the shape of the input data for the network.
        """
        shape = tuple(self.get_input_tensor().shape.as_list())
        return shape if include_batch else shape[1:]


