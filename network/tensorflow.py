
# standard imports
from __future__ import absolute_import
from typing import Union
from collections import OrderedDict
import os
import re
import logging

# third party imports
import numpy as np
from dltb.thirdparty.tensorflow import v1 as tf

# toolbox imports
import util
from datasource.data import ClassScheme
from . import Network as BaseNetwork, Classifier
from .exceptions import ParsingError
from .layers.tensorflow_layers import TensorFlowLayer as Layer
from .layers.tensorflow_layers import TensorFlowNeuralLayer as NeuralLayer
from .layers.tensorflow_layers import TensorFlowStridingLayer as StridingLayer
from .layers.tensorflow_layers import TensorFlowDense as Dense
from .layers.tensorflow_layers import TensorFlowConv2D as Conv2D
from .layers.tensorflow_layers import TensorFlowMaxPooling2D as MaxPooling2D
from .layers.tensorflow_layers import TensorFlowDropout as Dropout
from .layers.tensorflow_layers import TensorFlowFlatten as Flatten
from dltb.util.image import imread, imresize
from tools.classify import ImageClassifier

# logging
LOG = logging.getLogger(__name__)
del logging


class NonMatchingLayerDefinition(Exception):
    pass

class TensorflowException(Exception):
    pass


class Network(BaseNetwork):
    """Network interface to TensorFlow.

    Attributes
    ----------

    **Class attributes:**

    _OPERATION_TYPES: dict
        Auxiliary mapping of custom operation types (str) 
        to sets of TensorFlow operation types (str).
        This is only used to build up the _LAYER_DEFS (see below).

    _LAYER_TYPES_TO_CLASSES: dict
        Mapping layer class names (str) to corresponding Layer types (type).
        This is used to instantiate Layers when building up the layer_dict.

    _LAYER_DEFS: dict
        Mapping layer class names (str) to a list of TensorFlow operations
        types (str). This information is used when analyzing
        ("parsing") the graph to extract the network structure.
    
    Attributes
    ----------
    
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
                                 'Softmax' # Should not really count
                                           # as an activation
                                           # function, as it is
                                           # computed based on the
                                           # whole layer inputs. It is
                                           # convenient though.
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
        'normalization': {'LRN'},  # local response normalization
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
    
    _LAYER_DEFS2 = {
        'Conv2D': [
            _OPERATION_TYPES['convolution'],
            {'Add', 'BiasAdd'},
            _OPERATION_TYPES['activation_functions'],
            {'LRN', '?'}
        ],
        'MaxPooling2D': {'MaxPool'},
        'Dense': [
            {'MatMul'},
            {'Add', 'BiasAdd'},
            _OPERATION_TYPES['activation_functions'],
        ]
        # FIXME[todo]: 'Flatten'
        # FIXME[todo]: 'Dropout'
    }

    keras_name_regex = re.compile(r'(.)([A-Z][a-z0-9]+)')

    def __init__(self, checkpoint=None, graph_def=None, session=None, **kwargs):
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
        LOG.info("NEW Tensorflow Network: %s", kwargs)

        self._graph = None
        self._session = None

        self._init_graph_def = graph_def
        self._init_checkpoint = checkpoint
        self._init_session = session


        # TensorFlow uses channels last as data format by
        # default. This can however be changed by the user.
        # FIXME[todo]: make this more flexible.        
        kwargs['data_format'] = 'channels_last'        
        super().__init__(**kwargs)

    def _prepare(self):
        from tensorflow.python.client import device_lib
        LOG.info(device_lib.list_local_devices())
        self._prepare_graph()
            
        # Now we can call the superclass constructor to initialize the
        # network. This will make use of the _graph (by calling
        # _create_layer_dict).
        super()._prepare()
        LOG.debug("Prepare Tensorflow Network Session")
        self._prepare_session()

    def _prepare_graph(self):
        if self._graph is not None:
            return
        # FIXME[hack]: we should integrate the following better into the
        # preparation scheme (the following may as well initialize the session!)
        if self._init_graph_def is not None:
            self._init_from_graph_def(self._init_graph_def)
        elif self._init_checkpoint is not None:
            self._init_from_checkpoint(self._init_checkpoint)
        elif self._init_session is not None:
            self._init_from_session(self._init_session)
        else:
            self._graph = tf.Graph()

    def _prepare_session(self):
        LOG.debug("Tensorflow: _prepare_session: calling _online()")
        self._online()

    def _unprepare(self):
        self._offline()
        super()._unprepare()

    def _prepared(self):
        return (self._graph is not None and self._session is not None
                and super()._prepared())

    def _init_from_graph_def(self, graph_def: tf.GraphDef):
        """Initialize this tensorflow.Network from a TensorFlow
        Graph Definition.
        """
        LOG.info("Initialize TensorflowNetwork from GraphDef")
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
            Path to a Checkpoint file. All files described above should
            be available there.

        Raises
        ------
        ValueError:
            If checkpoint is None or not a valid checkpoint.
        """
        LOG.info(f"Initialize TensorflowNetwork from Checkpoint ({checkpoint})")
        self._graph = tf.Graph()
        self._online()
        with self._graph.as_default():
            saver = tf.train.import_meta_graph(checkpoint + '.meta',
                                               clear_devices=True)
            model_dir = os.path.dirname(checkpoint)

            # saver.restore runs the ops added by the constructor for
            # restoring variables. It requires a session in which
            # the graph was launched.
            # The variables to restore do not have to been initialized,
            # as restoring is itself a way to initialize variables.
            # Restoring can can fail (for different reasons):
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
        LOG.info("Initialize TensorflowNetwork from Session")
        self._offline()
        self._session = session
        self._graph = session.graph

    def _online(self) -> None:
        """Setup the tensorflow Session. This assumes that the computational
        graph has been prepared (in property :py:prop:`_graph`).

        """
        if self._session is None:
            LOG.info("online -> starting tf.Session")
            # tf_config = tf.ConfigProto(log_device_placement=True)
            if util.use_cpu:
                tf_config = tf.ConfigProto(device_count = {'GPU': 0})
            else:
                tf_config = tf.ConfigProto()
            LOG.info("Tensorflow: Creating new session for graph")
            self._session = tf.Session(graph=self._graph, config=tf_config)
        else:
            LOG.info("Tensorflow: Using already existing session")


    def _offline(self) -> None:
        """Put this model in offline mode. In offline mode, no
        inference will be possible.

        In TensorFlow offline mode means that no tf.Session is
        available. The tf.Graph may however still be present.
        """
        if self._session is not None:
            self._session.close()
            self._session = None

    #
    # Extracting the network structure from the graph
    #

    def _create_layer_dict(self) -> OrderedDict:
        """Try to find the sequences in operations of the graph that
        match the idea of a Layer.

        Returns
        -------
        A mapping of layer_ids to layer objects.
        """

        # Collect the relevant operations ...
        self._ops = self._get_operations()

        # ... and then build Layers from these operations.
        self._input_placeholder = self._ops[0]
        layer_dict = self._layers_from_operations(self._ops[1:])

        return OrderedDict(layer_dict)

    def _layers_from_operations(self, ops: list) -> OrderedDict:
        """Group TensorFlow operations to form Layers.

        Parameters
        ----------
        ops: list
            A list of TensorFlow operations.

        Returns
        ------
        layer_dict: OrderedDict
            An ordered dictionary, mapping layer names to Layers.
        """

        # The layer_dict maps layer names to Layers
        layer_dict = OrderedDict()

        # we will count the number of Layers of each type to generate
        # unique names.
        layer_counts = {
            layer_type: 0 for layer_type in self._LAYER_DEFS.keys()
        }

        # The current index in the operator list
        op_idx = 0
        while op_idx < len(ops):
            op = ops[op_idx]
            LOG.debug(f"parsing op-{op_idx}: "
                         f"{op.type} with name '{op.name}', "
                         f"{op.values()}")

            for layer_type, layer_def in self._LAYER_DEFS2.items():
                try:
                    # try to match the layer definition with the
                    # given operations
                    matching_ops, op_idx_last = \
                        self._match_layer_def(op_idx, layer_def)

                    # No exeception -> we are fine: create a new Layer...

                    # Invent a name for the new Layer
                    layer_counts[layer_type] += 1
                    layer_name = (f'{self._to_keras_name(layer_type)}'
                                  f'_{layer_counts[layer_type]}')

                    # Instantiate the new Layer
                    layer_cls = self._LAYER_TYPES_TO_CLASSES[layer_type]
                    layer_dict[layer_name] = layer_cls(self, matching_ops,
                                                       id=layer_name)

                    LOG.debug(f"debug:   ** {layer_name}"
                                 f" => {type(layer_dict[layer_name])}")

                    # If the layer definition was successfully
                    # matched, advance the number of ops that were
                    # matched. Don't try to match another layer
                    # definition at the same op by breaking the for
                    # loop.
                    op_idx = op_idx_last
                    break  # Do not try more layer types for this operation
                except NonMatchingLayerDefinition:
                    continue  # Try the next layer type

            # Try to match at the next op.
            op_idx += 1

        if not layer_dict:
            raise ParsingError('Could not find any layers in TensorFlow graph.')

        if False:
            self._debug_layer_dict(layer_dict)

        return layer_dict

    def _to_keras_name(self, name):
        return self.keras_name_regex.sub(r'\1_\2', name).lower()

    def _get_operations(self) -> list:
        """Get the (relevant) operations from the TensorFlow graph.  An
        operation is judged as relevant, if it lies on a path between
        the input (the first placeholder) and the output (the last
        activation function).

        Returns
        ------
        operations: list
            A list of TensorFlow operations, ordered by their input-output
            relations, i.e., all inputs to an operation will occur in this
            list before that operation.
        """

        # get the graph as a list of operations
        ops = self._graph.get_operations()
        
        # depth_dict will map TensorFlow operator names to their depth,
        # that is the length of the path from the input. The
        # input (the first placeholder) will get depth 1.
        depth_dict = OrderedDict()

        # walk through this list and extract the relevant operations:
        for op_idx, op in enumerate(ops):
            # The first placeholder we find will be our input layer:
            if not depth_dict:
                # depth_dict is not initialized, that means, no input
                # (placeholder) has been detected yet.
                # Ignore this op unless it is a Placeholder
                if op.type == 'Placeholder':
                    depth_dict[op.name] = 1
                continue

            # ok, we had found an input layer, that mean, this op can
            # be in the path. Check if one of its predecessors is
            # in the path:
            for input_name in [t.op.name for t in op.inputs]:
                if input_name in depth_dict:
                    depth_dict[op.name] = depth_dict[input_name]+1
                    if op.type == 'ConcatV2':
                        break

        if False:
            self._debug_depth_dict(depth_dict)
            
        return [self._graph.get_operation_by_name(name)
                for name in depth_dict.keys()]



    def _match_layer_def(self, op_idx: int, layer_def: dict) -> list:
        """Check whether the layer definition match the operations
        starting from a certain index.

        Parameters
        ----------
        op_idx: int
            The index of the first operation to match.
        layer_def: dict
            A dictionary mapping Layer classes to layer definitions.
            The index of the first operation to match.

        Returns
        -------
        matched_ops: list
            The list of TensorFlow operations matched.
            This list can be used to intialize a TensorFlow Layer.
        last_idx: int
            The index of the last operation included in the matched_ops
            list.

        Raises
        ------
        NonMatchingLayerDefinition:
            The operations fount in the TensorFlow operations list do
            not match the layer definition.
        """
        ops = self._ops
        matched_ops = []
        for i, op_group in enumerate(layer_def):

            if op_idx >= len(ops):
                raise NonMatchingLayerDefinition
            op = ops[op_idx]

            # skip irrelevant layers
            while op.type in ('Split', 'ConcatV2', 'Reshape'):
                op_idx += 1
                matched_ops.append(op)
                if op_idx >= len(ops):
                    raise NonMatchingLayerDefinition
                op = ops[op_idx]
            
            if op.type in op_group:
                while op_idx < len(ops) and ops[op_idx].type in op_group:
                    matched_ops.append(ops[op_idx])
                    op_idx += 1
            elif '?' in op_group:
                pass  # The operation group at this position is optional.
            else:
                raise NonMatchingLayerDefinition

        # If all operations could be matched, return the respective operations.
        return matched_ops, op_idx-1

    #
    # Old code for layer parsing ...
    #

    def _parse_old(self):
        while op_idx < len(ops):
            LOG.debug(f"debug:  op-{op_idx}: "
                         # "{type(ops[op_idx])}, "
                         # always <class 'tensorflow.python.framework.ops.Operation'>
                         f"{ops[op_idx].type} with name '{ops[op_idx].name}', "
                         f"{ops[op_idx].values()}")

            op = ops[op_idx]
            # The first placeholder we find will be our input layer:
            if self._input_placeholder is None:
                if op.type == 'Placeholder':
                    self._input_placeholder = op
                    op_idx += 1

            for layer_type, layer_def in self._LAYER_DEFS.items():
                try:
                    # try to match the layer definition with the
                    # given operations
                    matching_ops, op_idx_last = \
                        self._match_layer_def(op_idx, layer_def)

                    # No exeception -> we are fine: create a new Layer...

                    # Invent a name for the new Layer
                    layer_counts[layer_type] += 1
                    layer_name = (f'{self._to_keras_name(layer_type)}'
                                  f'_{layer_counts[layer_type]}')

                    # Instantiate the new Layer
                    layer_cls = self._LAYER_TYPES_TO_CLASSES[layer_type]
                    layer_dict[layer_name] = layer_cls(self, matching_ops)

                    LOG.debug(f"debug:   ** {layer_name}"
                                 f" => {type(layer_dict[layer_name])}")

                    # If the layer definition was successfully
                    # matched, advance the number of ops that were
                    # matched. Don't try to match another layer
                    # definition at the same op by breaking the for
                    # loop.
                    op_idx += len(layer_def) - 1
                    break
                except NonMatchingLayerDefinition:
                    continue  # Try the next layer type
            # Try to match at the next op.
            op_idx += 1

    def _match_layer_def_old(self, op_idx: int, layer_def: dict) -> list:
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

    #
    # Debug
    #

    def _debug_graph_def(self) -> None:
        """Output information about the underlying TensorFlow graph
        definition.
        """
        graph_def = self._graph.as_graph_def()

        LOG.debug(f"Network: {self.get_id()} ({type(self)}):") 
        for i, tensor in enumerate(graph_def.node):
            LOG.debug(f"  {i}) {tensor.name}: {type(tensor)}")
        LOG.debug(f"debug: Layer dict:") 

    def _debug_layer_dict(self, layer_dict: dict=None) -> None:
        """Output the given layer dict.

        Parameters
        ----------
        layer_dict: dict
            A dictionary mapping layer names to Layers.
        """
        if layer_dict is None:
            layer_dict = self.layer_dict
        if layer_dict is None:
            LOG.debug("No layer dictionary available.")
        for i, (name, layer) in enumerate(layer_dict.items()):
            LOG.debug("%d. %s: %s", i, name, layer)

    def _debug_depth_dict(self, depth_dict: dict) -> None:
        """Output a depth dictionary of TensorFlow operations.  Some
        operations will be annotated with additional information like
        input and output shape, strides, kernel size, etc.
        """
        indent = ""
        for name, depth in depth_dict.items():
            op = self._graph.get_operation_by_name(name)
            if op.type == 'ConcatV2':
                info = (f"{op.inputs[0].shape} + {op.inputs[1].shape}"
                        f" -> {op.outputs[0].shape}")
                indent = ""
            elif op.type == 'Split':
                info = (f"{op.inputs[1].shape}"
                        " -> {op.outputs[0].shape} x {op.outputs[1].shape}")
            elif op.type == 'Conv2D':
                info = (f"{op.inputs[0].shape} * {op.inputs[1].shape}"
                        f" -> {op.outputs[0].shape}; "
                        f"strides: {op.get_attr('strides')}, "
                        f"padding: {op.get_attr('padding')}, "
                        f"data_format: {op.get_attr('data_format')}")
            elif op.type == 'MaxPool':
                info = (f"{op.inputs[0].shape}"
                        f" -> {op.outputs[0].shape}; "
                        f"strides: {op.get_attr('strides')}, "
                        f"padding: {op.get_attr('padding')}, "
                        f"data_format: {op.get_attr('data_format')}")
            elif op.type == 'Reshape':
                info = f"{op.inputs[0].shape} -> {op.outputs[0].shape}"
            else:
                info = ""

            LOG.debug("%d: %s%s (%s): [%s]",
                      depth, indent, op.name, op.type, info)

            if op.type == 'Split':
                indent = "  "

    #
    # Network operations
    #

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

    def _get_input_shape(self) -> tuple:
        """Get the shape of the input data for the network.
        """
        return tuple(self.get_input_tensor().shape.as_list())


class Alexnet(ImageClassifier, Classifier, Network):
    """
    AlexNet trained on ImageNet data (TensorFlow).
    """

    def __init__(self, *args, **kwargs):
        LOG.debug("alexnet: import tensorflow")
        checkpoint = os.path.join('models', 'example_tf_alexnet',
                                  'bvlc_alexnet.ckpt')
        LOG.debug("alexnet: TensorFlowNetwork")
        super().__init__(*args, checkpoint=checkpoint, id='AlexNet',
                         scheme='ImageNet', lookup='caffe', **kwargs)

        # FIXME[old]:
        # if 'ALEXNET_MODEL' in os.environ:
        #    model_path = os.getenv('ALEXNET_MODEL', '.')
        # else:
        #    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
        #                              'models', 'example_tf_alexnet')
        #
        # checkpoint = os.path.join(model_path, 'bvlc_alexnet.ckpt')
        # if not os.path.isfile(checkpoint + '.meta'):
        #     raise ValueError("AlexNet checkpoint files do not exist. "
        #                      "You can generate them by downloading "
        #                      "'bvlc_alexnet.npy' and then running "
        #                      "models/example_tf_alexnet/alexnet.py.")

    def _prepare(self):
        super()._prepare()
        LOG.debug("alexnet: prepare")
        self._online()

        LOG.debug("alexnet: Load Class Names")
        # from datasource import Datasource
        # imagenet = Datasource.register_initialize_key('imagenet-val')
        # self.set_labels(imagenet, format='caffe')
        LOG.debug("alexnet: Done")

        # FIXME[old]:
        #
        # if toolbox.contains_tool('activation'):
        #    tool = self.get_tool('activation')
        #    tool.set_network(network)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        """
        # image = (image[0][:, :, :3]).astype(np.float32)
        image = image[:, :, :3].astype(np.float32)
        image = imresize(image, (227, 227))
        image = image - image.mean()
        image[:, :, 0], image[:, :, 2] = image[:, :, 2], image[:, :, 0]
        return image

    def _image_as_batch(self, image: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image, str):
            # FIXME[todo]: general resizing/preprocessing strategy for networks
            size = self.get_input_shape(include_batch=False,
                                        include_channel=False)
            image = imread(image)  # , size=size

        image = self.preprocess_image(image)

        # add batch dimension
        image = np.expand_dims(image, axis=0)
        return image
