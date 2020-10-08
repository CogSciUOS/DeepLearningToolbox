"""Basic network definition.
"""

# standard imports
from typing import Tuple, Any, Union, List, Iterator
from collections import OrderedDict
import functools
import operator
import logging
import importlib

# third party imports
import numpy as np

# toolbox imports
from ..base.register import RegisterClass
from ..base.data import Data, Datalike
from ..base.image import Imagelike, ImageExtension
from ..tool import Tool
from ..tool.classifier import SoftClassifier
from base import Identifiable, Extendable, Preparable

# logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


# FIXME[design]: we should decide on some points:
#
#  * there is some confusion about the ordering of channels and
#    we should provide a general strategy how to deal with this.
#    also reporting batch size ...
#
#  * provide some information on the loaded_network
#     - version of the framework
#     - general info: number of parameters / memory usage
#     - GPU usage
#
#  * provide some information on the layer
#


class Network(Identifiable, Extendable, Preparable, method='network_changed',
              changes={'state_changed', 'weights_changed'},
              metaclass=RegisterClass):
    """Abstract Network interface for all frameworks.



    ** Data format **

    There seem to be (at least) two formats in use in different
    frameworks: `channels_first` means that the channel axis is
    before additional (spatial) axes, that is for 2-dimensional layers
    the axis would be ``(batch, channel, height, width)``, short `NCHW`.
    `channels_last` means that the channel axis is the last data axes,
    that is the axes for a 2-dimensional layer wourd be
    ``(batch, channel, height, width)``, short `NHWC`.

    We have decided to use `channel_last` as a standard ordering, as this
    seems to be the natural ordering for RGB images. However, there
    may also be arguments against this ordering. Rüdiger has mentioned
    that cuDNN `requires channel first data
    <https://caffe2.ai/docs/tutorial-image-pre-processing.html#null__caffe-prefers-chw-order>`

    The Network API will allow to order the dimensions in data arrays
    in a way independent of the underlying Network
    implementation. There seem to be different orderings applied in
    different frameworks:

      * tensorflow default is to use channel last (``NHWC``)
        (`can be changed to channel first
         <https://www.tensorflow.org/api_docs/python/tf/nn/conv2d>`_:
         ``data_format = "NCHW"``)

      * `pytorch only supports channel first
         <http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d>`_ (``NCHW``)

      * `pycaffe seems to default to
         <http://caffe.berkeleyvision.org/tutorial/net_layer_blob.html>`_
         (``NCHW``)

      * theano: ?

    To stay flexible, we introdude two properties to adapt this
    behaviour:

    _channel_axis:
        The format for data passed to the network via the public
        interface. Also data returned by public methods, like
        :py:meth:`get_activations` will be in that format.

    _channel_internal:
        The internal data format used by the underlying implementation.
        This is the format expected and returned by all internal
        methods like :py:meth:`_get_activations`.

    Public methods may in addition accept an optional `channel_axis`
    argument, allowing to overwrite the :py:class:`Network`s default.

    .. note::
        It may be more useful to be able to specify the desired order,
        either globally for a Network, in each method that gets or
        returns array data.

    Attributes
    ----------
    layer_dict: Dict[str,Layer]

    Changes
    -------
    'busy_changed':
        the network state changed
    'state_changed':
        the network state changed
    'weights_changed':
        network weights were changed

    """

    # FIXME[problem]:
    # There seems to be a mismatch between layer.id and the layer_id that
    # is used as key in the layer_dict.
    # (1) the name 'id' is alread used by the python buildin function id()
    #     that provides a unique number for each object.
    # (2) we want a unique key for each layer in the layer dict
    #     probably a string (as we also want numerical index access)
    # (3) maybe we can use the register interface?
    # (4) we also want a string to be presented in user interfaces:
    #     can that be the key or should we introduce an additional name?
    #     -> Probably it is best to just use one string ...
    # Action plan:
    # -> remove the 'Identifiable' -> replace with register
    #     - only used in
    #        ./network/network.py
    #        ./network/layers/layers.py
    # -> replace
    #       layer.id -> layer.key
    #       layer_id -> layer_key
    #       layer.name -> layer_key
    #       network.id -> network_key

    @classmethod
    def import_framework(cls):
        """Import the framework (python modules) required by this network.
        """
        pass  # to be implemented by subclasses

    # FIXME[todo]: add additional parameters, e.g. cpu / gpu
    # FIXME[concept]: there are at least two different concepts of loading:
    #  - load the architecture
    #  - load the parameters
    #  - load architecure and parameters
    # Also: loading could be done like here (as class method) or
    # as instance method - it should then call
    #   self.change('weights_changed')
    @classmethod
    def load(cls, module_name, *args, **kwargs):
        LOG.info("Loading module: %s", module_name)
        module = importlib.import_module(module_name)
        network_class = getattr(module, "Network")
        network_class.import_framework()
        instance = network_class(*args, **kwargs)
        return instance

    # ------------ Public interface ----------------

    CHANNEL_AXIS_FIRST = 'channels_first'
    CHANNEL_AXIS_LAST = 'channels_last'

    _channel_axis: str = 'channels_last'  # or 'channels_first'
    # network/tests/test_network.py
    # ./network/tensorflow.py
    # ./network/torch.py
    # ./network/caffe.py
    # ./network/layers/caffe_layers.py
    # ./dltb/network/network.py
    # --
    # ./network/keras.py
    # --
    # keras.backend.image_data_format()  # 'channels_last'
    #
    # FIXME[todo]: channel ordering is relevant when processing
    # actiations

    def __init__(self, id=None, channel_axis: str = None,
                 **kwargs) -> None:
        """

        Parameters
        ----------
        **kwargs
            channel_axis: {'channels_last', 'channels_first'}
                The place of the color channels in the tensors.
        """
        LOG.debug("Network.__init__: %s [%s]", kwargs, self.__class__.__mro__)
        # Prohibited instantiation of base class.
        if self.__class__ == Network:  # FIXME[todo]: use python ABC mechanism
            raise NotImplementedError('Abstract base class Network '
                                      'cannot be used directly.')
        super().__init__(id, **kwargs)

        # Every loaded_network should know which data format it is using.
        # Default is channels_last.
        if channel_axis is not None:
            self._channel_axis = channel_axis
        self.layer_dict = None

    @property
    def channel_axis(self) -> str:
        return self._channel_axis

    @channel_axis.setter
    def channel_axis(self, axis: str) -> None:
        self._channel_axis = axis

    #
    # Sized interface
    #

    def __len__(self) -> int:
        """The length of a network is the number of layers.
        """
        return len(self.layer_dict)

    def __iter__(self) -> Iterator['Layer']:
        """Iterate the layers of the network.
        """
        return iter(self.layer_dict.values())

    def __contains__(self, key: Any) -> bool:
        """Check if the network contains the given layer.
        """
        if isinstance(key, str):
            return key in self.layer_dict
        elif isinstance(key, Identifiable):
            return key.get_id() in self.layer_dict
        return False

    def __getitem__(self, key: Any) -> 'Layer':
        """Obtain the given layer from this :py:class:`Network`.
        """
        if isinstance(key, str):
            return self.layer_dict[key]
        elif isinstance[key, int]:
            return list(self.layer_dict.items())[key]
        elif isinstance(key, Identifiable):
            return self.layer_dict[key.get_id()]
        raise KeyError(f"No layer for key '{key}' in network.")

    #
    # Preparation
    #

    def _prepare(self):
        super()._prepare()
        #
        # Create the layer representation.
        #

        self._prepare_layers()

    def _prepare_layers(self) -> None:

        # default implementation: create a layer_dict
        # subclasses may overwrite this to provide a different layer
        # mechanism.
        self.layer_dict = self._create_layer_dict()

        # connect layers in the dictionary
        for i in range(len(self.layer_dict)):
            if i > 0:
                self[i]._predecessor = self[i-1]
            if i+1 < len(self.layer_dict):
                self[i]._successor = self[i+1]

    def _unprepare(self):
        self.layer_dict = None
        super()._unprepare()

    def _prepared(self) -> bool:
        return (self.layer_dict is not None) and super()._prepared()

    def __getitem__(self, layer: Union[int, str]) -> 'Layer':
        """Provide access to the layers by number. Access by id is provided
        via :py:attr:`layer_dict`."""
        if not self.prepared:
            return RuntimeError("Trying to access unprepared Network "
                                f"'{self.key}'")
        if isinstance(layer, str):
            return self.layer_dict[layer]
        if isinstance(layer, int):
            return tuple(self.layer_dict.values())[layer]
        raise TypeError("Network index argument layer has invalid type "
                        f"{type(layer)}, should by str or int")

    # @busy("getting activations")
    def get_activations(self, inputs: np.ndarray,
                        layer_ids: Any = None,
                        channel_axis: str = 'channels_last',
                        image: bool = False
                        ) -> Union[np.ndarray, List[np.ndarray]]:
        """Gives activations values of the loaded_network/model
        for given layers and an input sample.

        Parameters
        ----------
        inputs:
            For multi-channel, two-dimensional data, we expect the
            input data to be given in with channel last, that is
            (N,H,W,C). For plain data of dimensionality D we expect
            batch first (N,D).
        layer_ids:
            The layers the activations should be fetched for. Single
            layer_id or list of layer_ids.
        channel_axis: {'channels_last', 'channels_first'}
            The format in which the data is provided.
            Either "channels_first" or "channels_last".
        image: bool
            The data provided are an image.

        Returns
        -------
        Array of shape
        (input_samples, image_height, image_width, feature_maps).

        """
        internal, is_batch, is_internal = \
            self._transform_input(inputs, channel_axis)

        # Check whether the layer_ids are actually a list.
        layer_ids, is_list = self._force_list(layer_ids)

        # Transform the input_sample appropriate for the loaded_network.
        LOG.debug("Transforming input data from %s to %s.",
                  inputs.shape, channel_axis)
        activations = self._get_activations(internal, layer_ids)

        # Transform the output to stick to the canocial interface.
        activations = [self._transform_outputs(activation, channel_axis,
                                               is_batch, is_internal)
                       for activation in activations]
        # If it was just asked for the activations of a single layer,
        # return just an array.
        if not is_list:
            activations = activations[0]
        return activations

    def get_net_input(self, layer_ids: Any,
                      input_samples: np.ndarray,
                      channel_axis: str = 'channels_last'
                      ) -> Union[np.ndarray, List[np.ndarray]]:
        """Gives the net input (inner product + bias) values of the network
        for given layers and an input sample.

        Parameters
        ----------
        layer_ids
            The layers the activations should be fetched for. Single
            layer_id or list of layer_ids.
        input_samples
             For multi-channel, two-dimensional data, we expect the
             input data to be given in with channel last, that is
             (N,H,W,C). For plain data of dimensionality D we expect
             batch first (N,D).

        Returns
        -------
        Array of shape
        (input_samples, image_height, image_width, feature_maps).

        """
        # Check whether the layer_ids are actually a list.
        layer_ids, is_list = self._force_list(layer_ids)
        # Transform the input_sample appropriate for the loaded_network.
        internal, is_batch, is_internal = \
            self._transform_input(input_samples, channel_axis)
        activations = self._compute_net_input(layer_ids, input_samples)
        # Transform the output to stick to the canocial interface.
        activations = [self._transform_outputs(activation, channel_axis,
                                               is_batch, is_internal)
                       for activation in activations]
        # If it was just asked for the activations of a single layer,
        # return just an array.
        if not is_list:
            activations = activations[0]
        return activations

    def layers(self) -> Iterator['Layer']:
        return self.layer_dict.values()

    def get_layer_info(self, layername):
        """FIXME[todo]: we still have to decide on some info API

        Parameters
        ----------
        layername

        Returns
        -------
        """
        raise NotImplementedError

    # ------------------- Things to be implmeneted by subclasses --------------

    def _get_activations(self, input_samples: Any,
                         layer_ids: list) -> List[Any]:
        """To be implemented by subclasses.
        Computes a list of activations from a list of layer ids.
        """
        raise NotImplementedError

    def _compute_net_input(self, layer_ids: list,
                           input_samples: np.ndarray) -> np.ndarray:
        """To be implemented by subclasses.
        Computes a list of net inputs from a list of layer ids."""
        raise NotImplementedError

    def _create_layer_dict(self) -> OrderedDict:
        """Create the mapping from layer ids to layer objects.

        Returns
        -------

        """
        # FIXME[hack]: we need this to initialize the Network
        # remove this once there is a proper initialization concept
        #raise NotImplementedError
        return {}

    # ---------------------- Private helper functions -------------------------

    def _transform_input(self, inputs: np.ndarray,
                         channel_axis: str) -> np.ndarray:
        """Fills up the ranks of the input, e.g. if no batch size was
        specified and converts the input to the data format of the model.

        Parameters
        ----------
        inputs
            The inputs fed to the network.
        channel_axis: {'channels_last', 'channels_first'}
            The data format of inputs.

        Returns
        -------
        internal:
            The transformed input.
        is_batch: bool
            The data was already as a batch.
        is_internal: bool
            The data was already given in the internal format.
        """
        is_internal = True

        shape = self.get_input_shape()
        if len(inputs.shape) == len(shape):
            is_batch = True
        elif len(inputs.shape) == len(shape) - 1:
            inputs = inputs[np.newaxis]
            is_batch = False
        else:
            raise ValueError(f"Invalid input of shape {inputs.shape} "
                             f"for network with imput shape {shape}")

        channel_index = -1 if self._channel_axis == 'channels_last' else 1
        channels = shape[channel_index]

        if channels == inputs.shape[channel_index]:
            # channel_swap = False
            pass
        elif (self._channel_axis == 'channels_last' and
              inputs.shape[1] == channels):
            # channel_swap = True
            inputs = inputs.transpose((0, 2, 3, 1))
        elif (self._channel_axis == 'channels_first' and
              inputs.shape[-1] == channels):
            # channel_swap = True
            inputs = inputs.transpose((0, 3, 1, 2))
        else:
            raise ValueError(f"Cannot find channels ({channels}) "
                             f"in input of show {inputs.shape}")

        return inputs, is_batch, is_internal

    def _transform_outputs(self, outputs: np.ndarray,
                           channel_axis: str,
                           batch: bool, internal: bool) -> np.ndarray:
        """Convert output values into a desired data format.
        """
        if not internal:
            outputs = transpose_channel_axis(outputs,
                                             input_format=self._channel_axis,
                                             output_format=channel_axis)
        if not batch:
            outputs = outputs[0]
        return outputs

    def _fill_up_ranks(self, inputs: np.ndarray) -> np.ndarray:
        """Fill up the ranks of the input tensor in case no batch or
        color dimension is provided.

        Parameters
        ----------
        inputs
            The inputs fed to the network.
        channel_axis: {'channels_last', 'channels_first'}
            The data format to transform to.


        Returns
        -------

        """

        #network_input_shape = self[0].input_shape
        network_input_shape = self.get_input_shape()
        # Checking whether input samples was provided with all for channels.
        if len(inputs.shape) == 2:
            # Only width and height means we are dealing with one
            # grayscale image.
            inputs = inputs[np.newaxis, :, :, np.newaxis]
        elif len(inputs.shape) == 3:
            # We have to decide whether the batch or the channel
            # dimension is missing.
            #
            # Ask the loaded_network what shape it expects. Since
            # either the last three dimensions (in case channel was
            # provided) or second and third dimension (in case batch)
            # was provided have to match.

            if self._is_channel_provided(inputs.shape, network_input_shape):
                inputs = inputs[np.newaxis, ...]
            elif self._is_batch_provided(inputs.shape, network_input_shape):
                inputs = inputs[..., np.newaxis]
            else:
                raise ValueError('Non matching input dimensions: '
                                 'network={} vs data={}'.
                                 format(network_input_shape, inputs.shape))
        elif len(inputs.shape) > 4:
            raise ValueError('Too many input dimensions.'
                             'Should be maximally 4 instead of {}.'
                             .format(len(inputs.shape)))

        return inputs

    @staticmethod
    def _is_channel_provided(input_sample_shape: tuple,
                             network_input_shape: tuple) -> bool:
        """Check if a given shape includes a channel dimension.
        The channel dimension is assumed to be at the first axis.
        """
        return input_sample_shape == network_input_shape[1:]

    @staticmethod
    def _is_batch_provided(input_sample_shape: tuple,
                           network_input_shape: tuple) -> bool:
        """Check if a given shape includes a batch dimension.
        The batch dimension is assumed to be at the first axis.
        """
        return input_sample_shape[1:3] == network_input_shape[1:3]

    def _force_list(self, maybe_list: Union[List[str], Any]) -> \
            Tuple[List['str'], bool]:
        """Turn something into a list, if it is none.

        Returns
        -------
        maybe_list: list
            The input, turned into a list if necessary.
        is_list: bool
            A flag indicating whether the input was a list or not

        """
        if maybe_list is None:
            return list(self.layer_dict.keys()), True

        if isinstance(maybe_list, list):
            return maybe_list, True

        return [maybe_list], False

    # -------------------------- methods from Ulf ----------------------

    def _canonical_input_shape(self, input_shape: tuple) -> tuple:
        """Transform an input shape into the canonical form. For
        convolutional layers, this is channel last ordering (N,H,W,C).
        For flat input of dimension D it is (D,N).

        This method is intended to be used to determine the actual
        layer shapes for networks, that do not provide this
        information.

        Parameters
        ---------
        input_shape:
            The shape of an input sample. May or may not include batch (B)
            or channel (C) dimension. If so, channel should be last, i.e.
            (N,H,W,C)

        Returns
        -------
        tuple
            The input_shape in canonical form.

        Raises
        ------
        ValueError
            The provided shape is incorrect
        """
        network_input_channels = self._get_number_of_input_channels()
        if len(input_shape) == 2:
            ## Only width and height, so we will add the channel information
            ## from the loaded_network input.
            input_shape = (1, *input_shape, network_input_channels)
        elif len(input_shape) == 3:
            if input_shape[-1] == network_input_channels:
                ## channel information is provided, add batch
                input_shape = (1, *input_shape)
            else:
                ## channel information is not provided, add it
                input_shape = (*input_shape, network_input_channels)
        elif len(input_shape) != 4:
            raise ValueError('Incorrect input shape {}, len should be {}'
                             .format(input_shape, 4))
        elif input_shape[-1] != input_network_input_channels:
            raise ValueError('Invalid input shape {}: channels should be {}'
                             .format(input_shape, network_input_channels))
        return input_shape

    def _get_number_of_input_channels(self) -> int:
        """Get the number of input channels for this loaded_network.
        This is the number of channels each input given to the loaded_network
        should have.  Usually this coincides with the number of
        channels in the first layer of the loaded_network.

        The standard implementation just extracts this from the input
        shape. However, in some networks the input shape is not
        available upon initialization and those networks should
        reimplement this method to provide the channel number,
        which should always be known.

        Returns
        -------
        int
            The number of input channels or 0 if the loaded_network does not
            have input channels.
        """
        network_input_shape = self.get_layer_input_shape(self.layer_ids[0])
        return network_input_shape[-1] if len(network_input_shape) > 2 else 0

    #
    # methods for accessing layer attributes
    #

    def empty(self) -> bool:
        return self.layer_dict is None or not bool(self.layer_dict)

    def input_layer_id(self) -> str:
        first_layer_id = next(iter(self.layer_dict.keys()))
        return first_layer_id

    def output_layer_id(self):
        for last_layer_id in iter(self.layer_dict.keys()):
            pass
        return last_layer_id

    def get_input_layer(self):
        return self.layer_dict[self.input_layer_id()]

    def get_output_layer(self):
        return self.layer_dict[self.output_layer_id()]

    def get_input_shape(self, include_batch: bool = True,
                        include_channel: bool = True) -> tuple:
        """Get the shape of the input data for the network.
        """
        shape = self._get_input_shape()
        if include_batch and include_channel:
            return shape
        if include_channel:  # and not include_batch
            return shape[1:]
        if not include_batch:  # and not include_channel
            return (shape[1:-1] if self._channel_axis == 'channels_last'
                    else shape[2:])
        # include_batch and not include_channel
        if self._channel_axis == 'channels_last':
            return shape[:-1]
        return (shape[0],) + shape[2:]

    def _get_input_shape(self) -> tuple:
        return self.get_layer_input_shape(self.input_layer_id())

    def get_output_shape(self, include_batch: bool = True) -> tuple:
        """Get the shape of the output data for the network.
        """
        shape = self.get_layer_output_shape(self.output_layer_id())
        return shape if include_batch else shape[1:]

    def is_classifier(self):
        """Check if this network is a classifier.
        This is just a guess, based on simple heuristics.
        There is no way to give a definitive answer here.
        Subclasses may overwrite this method if additional
        information is available.
        """
        return len(self.get_output_shape(False)) == 1

    def get_layer_input_shape(self, layer_id) -> tuple:
        """
        Give the shape of the input of the given layer.

        Parameters
        ----------
        layer_id

        Returns
        -------
        tuple:
            For convolutional layers, this will be channel last (H,W,C).
        """
        return self.layer_dict[layer_id].input_shape

    def get_layer_output_shape(self, layer_id) -> tuple:
        """
        Give the shape of the output of the given layer.

        Parameters
        ----------
        layer_id

        Returns
        -------
        tuple:
            For convolutional layers, this will be channel last (H,W,C).
        """
        return self.layer_dict[layer_id].output_shape

    def get_layer_weights(self, layer_id) -> np.ndarray:
        """Returns weights INCOMING to the layer (layer_id) of the model
        shape of the weights variable should be coherent with the
        get_layer_output_shape function.

        Parameters
        ----------
        layer_id :
             An identifier for a layer.

        Returns
        -------
        ndarray
            Weights of the layer. For convolutional layers,
            this will be channel last (H,W,C_in,C_out).
        """
        return self.layer_dict[layer_id].weights

    def _get_layer_weights_shape(self, layer_id) -> tuple:
        weights = self.get_layer_weights(layer_id)
        if weights is None:
            return None
        # FIXME[hack/old]: old implementation returns list: (weights,bias)
        if isinstance(weights, list):
            if len(weights) == 0:  # e.g. dropout layer
                return None
            weights = weights[0]
        return weights.shape

    def get_layer_biases(self, layer_id) -> np.ndarray:
        """Returns the bias values for the layer (layer_id) of the model.

        Parameters
        ----------
        layer_id :
             An identifier for a layer.

        Returns
        -------
        ndarray
            A one-dimensional array of bias values.
            For dense layer, this will be one per output unit,
            for convolutional layers, this will be one per channel.
        """
        return self.layer_dict[layer_id].bias

    def get_layer_number_of_parameters(self, layer_id) -> int:
        """Returns the number of (learnable) parameters for a layer.
        For normal layers, the parameters are the weights and the
        bias values.

        Parameters
        ----------
        layer_id :
             An identifier for a layer.

        Returns
        -------
        int
            The number of parameters.
        """
        return self.layer_dict[layer_id].num_parameters

    def get_layer_input_units(self, layer_id) -> int:
        """The number of input units of this layer. For convolutional
        layers this will be the number of all units in all channels.

        Parameters
        ----------
        layer_id:
            Identifier of a layer in this loaded_network.

        Raises
        ------
        ValueError:
            If the layer_id fails to identify a layer in this loaded_network.
        """
        input_shape = self.get_layer_input_shape(layer_id)
        # FIXME[hack/todo]: we should specify how to provide the input shape
        # (with or without batch index)
        if input_shape[0] is None:
            input_shape = input_shape[1:]
        return functools.reduce(operator.mul, input_shape, 1)

    def get_layer_output_units(self, layer_id) -> int:
        """The number of output units of this layer. For convolutional
        layers this will be the number of all units in all channels.

        Parameters
        ----------
        layer_id:
            Identifier of a layer in this loaded_network.

        Raises
        ------
        ValueError:
            If the layer_id fails to identify a layer in this loaded_network.
        """
        output_shape = self.get_layer_output_shape(layer_id)
        # FIXME[hack/todo]: we should specify how to provide the input shape
        # (with or without batch index)
        if output_shape[0] is None:
            output_shape = output_shape[1:]
        return functools.reduce(operator.mul, output_shape)

    def get_layer_input_channels(self, layer_id) -> int:
        """The number of input channels for a cross-correlation/convolution
        operation.

        Parameters
        ----------
        layer_id:
            Identifier of a convolutional layer in this loaded_network.

        Raises
        ------
        ValueError:
            If the layer_id fails to identify a convolutional layer.
        """
        self._check_layer_is_convolutional(layer_id)
        layer_shape = self._get_layer_weights_shape(layer_id)
        return layer_shape[-2]

    def get_layer_output_channels(self, layer_id) -> int:
        """The number of output channels for a cross-correlation/convolution
        operation.

        Parameters
        ----------
        layer_id:
            Identifier of a convolutional layer in this loaded_network.

        Raises
        ------
        ValueError:
            If the layer_id fails to identify a convolutional layer.
        """
        self._check_layer_is_convolutional(layer_id)
        layer_shape = self._get_layer_weights_shape(layer_id)
        return layer_shape[-1]

    #
    # Convolutional layers
    #

    def get_layer_kernel_size(self, layer_id) -> int:
        """The size of the kernel in a cross-correlation/convolution
        operation. This is just the spatial extension and does not
        include the number of channels.

        Parameters
        ----------
        layer_id:
            Identifier of a convolutional layer in this loaded_network.

        Raises
        ------
        ValueError:
            If the layer_id fails to identify a convolutional layer.
        """
        self._check_layer_is_convolutional(layer_id)
        layer_shape = self._get_layer_weights_shape(layer_id)
        return layer_shape[:-2]

    def get_layer_stride(self, layer_id) -> (int, int):
        """The stride for the cross-correlation/convolution operation.

        Parameters
        ----------
        layer_id:
            Identifier of a convolutional layer in this loaded_network.

        Raises
        ------
        ValueError:
            If the layer_id fails to identify a convolutional layer.
        """
        self._check_layer_is_convolutional(layer_id)
        raise NotImplementedError

    def get_layer_dilation(self, layer_id) -> (int, int):
        """The dilation for the cross-correlation/convolution operation, i.e,
        the horizontal/vertical offset between adjacent filter
        rows/columns.

        Parameters
        ----------
        layer_id:
            Identifier of a convolutional layer in this loaded_network.

        Raises
        ------
        ValueError:
            If the layer_id fails to identify a convolutional layer.
        """
        self._check_layer_is_convolutional(layer_id)
        raise NotImplementedError

    def get_layer_padding(self, layer_id) -> (int, int):
        """The padding for the cross-correlation/convolution operation, i.e,
        the number of rows/columns (on both sides) by which the input
        is extended (padded with zeros) before the operation is
        applied.

        Parameters
        ----------
        layer_id:
            Identifier of a convolutional layer in this loaded_network.

        Raises
        ------
        ValueError:
            If the layer_id fails to identify a convolutional layer.
        """
        self._check_layer_is_convolutional(layer_id)
        raise NotImplementedError

    def get_layer_output_padding(self, layer_id) -> (int, int):
        """The output padding for the cross-correlation/convolution operation.

        Parameters
        ----------
        layer_id:
            Identifier of a convolutional layer in this loaded_network.

        Raises
        ------
        ValueError:
            If the layer_id fails to identify a convolutional layer.
        """
        self._check_layer_is_convolutional(layer_id)
        raise NotImplementedError

    def layer_is_convolutional(self, layer_id) -> bool:
        """Check if the given layer is a convolutional layer. If so,
        additional information can be obtained by the methods
        get_layer_kernel_size, get_layer_channels, get_layer_stride,
        get_layer_padding, and get_layer_dilation.

        Parameters
        ----------
        layer:
            Identifier of a layer in this loaded_network.

        Returns
        -------
        bool
            True for convolutional layers, else False.
        """
        layer_shape = self._get_layer_weights_shape(layer_id)
        return layer_shape is not None and len(layer_shape) > 2

    def _check_layer_is_convolutional(self, layer_id) -> None:
        """Ensure that the given layer is convolutional.

        Parameters
        ----------
        layer:
            Identifier of a layer in this loaded_network.

        Raises
        ------
        ValueError
            If the given layer is not convolutional.
        """
        if not self.layer_is_convolutional(layer_id):
            raise ValueError('Not a convolutional layer: {}'.format(layer_id))

    #
    # Pre- and postprocessing
    #

    def preprocess(self, inputs: Any) -> Any:
        return inputs

    def postprocess(self, outputs: Any) -> Any:
        return outputs

    def internal_to_numpy(self, array: np.ndarray,
                          unbatch: bool = False) -> np.ndarray:
        """Convert data from internal format into a numpy array.
        Should be overwritten by subclases that do not use
        numpy arrays for internal representation.

        Arguments
        ---------
        array:
            The array in internal representation.
        unbatch: bool
            A flag indicating if the data should be unbatched.
        """
        if unbatch:
            if array.shape[0] != 1:
                raise ValueError("Cannot unbatch array "
                                 f"with batch size {array.shape[0]}")
            array = array[0]
        return array

    #
    # Information
    #

    def summary(self) -> None:
        print(f"Network[{type(self)}] with {len(self)} layers:")
        for index, layer in enumerate(self):
            print(f"({index:3}) {layer.get_id():20}: "
                  f"{layer.input_shape} -> {layer.output_shape}")


class ImageNetwork(ImageExtension, base=Network):
    """A network for image processiong. Such a network provides
    additional methods to support passing images as arguments.
    """
    @property
    def input_size(self) -> Tuple[int, int]:
        return self.get_input_shape()

    # FIXME[hack]: this changes the semantics of the function:
    # the base class expects inputs: np.ndarray, while we expect an Imagelike.
    # We should improve the interface (either a new method or a somethign
    # more flexible)
    def get_activations(self, inputs: Imagelike,
                        layer_ids: Any = None,
                        channel_axis: str = 'channels_last'
                        ) -> Union[np.ndarray, List[np.ndarray]]:

        LOG.debug("ImageNetwork.getActivations(%s, %s)",
                  inputs.shape, layer_ids)
        internal = self.image_to_internal(inputs)
        is_list = isinstance(layer_ids, list)

        # Check whether the layer_ids are actually a list.
        layer_ids, is_list = self._force_list(layer_ids)

        # Transform the input_sample appropriate for the loaded_network.
        LOG.debug("ImageNetwork: internal=%s", internal.shape)
        activations = self._get_activations(internal, layer_ids)

        # Transform the output to stick to the canocial interface.
        activations = [self._transform_outputs(activation, channel_axis,
                                               False, False)
                       for activation in activations]
        LOG.debug("ImageNetwork: activations=%s", activations[0].shape)
        # If it was just asked for the activations of a single layer,
        # return just an array.
        if not is_list:
            activations = activations[0]
        return activations

    #
    # Implementation of the Tool interface (not used yet)
    #

    def _preprocess(self, image: Imagelike, *args, **kwargs) -> Data:
        data = super(self, image=None, *args, **kwargs)
        data.add_attribute('image', self.image_to_internal(image))
        return data


class Classifier(SoftClassifier, Network):
    """A :py:class:`Network` to be used as classifier.

    """

    @property
    def logit_layer(self) -> 'Layer':
        """The layer providing the class "logits".
        This is usually the prefinal layer that is passed
        in the softmax output layer.
        """
        return self.output_layer_id()  # FIXME[todo]

    @property
    def score_layer(self) -> 'Layer':
        """The layer computing the class scores ("probabilities").
        This is usually the output layer of the :py:class:`Classifier`.
        """
        return self.output_layer_id()

    def _prepare(self, **kwargs) -> None:
        """Prepare this :py:class:`Classifier`.

        Raises
        ------
        ValueError
            If the :py:class:`ClassScheme` does not fit to this
            :py:class:`Classifier`.
        """
        super()._prepare()

        # check if the output layer fits to the classification scheme
        output_shape = self.get_output_shape(include_batch=False)
        if len(output_shape) > 1:
            raise ValueError(f"Network '{self}' seems not to be Classifier "
                             f"(output shape is {output_shape})")
        elif output_shape[0] != len(self._scheme):
            raise ValueError(f"Network '{self}' does not fit the the "
                             f"classification scheme: {output_shape} output "
                             f"units vs. {len(self._scheme)} classes")

    #
    # Implementation of the SoftClassifier API
    #

    def class_scores(self, data: Datalike) -> np.ndarray:
        """Implementation of the :py:class:`SoftClassifier` interface.
        """
        data_internal = self.image_to_internal(data)
        scores_internal = self._class_scores(data_internal)
        scores = self.internal_to_numpy(scores_internal, unbatch=True)
        return scores

    def _class_scores(self, inputs: np.ndarray) -> np.ndarray:
        # FIXME[todo]: at least in tensorflow it seems possible to feed
        # a list of inputs!
        # output = sess.run(network_output_tensor,
        #                   feed_dict={network_input_tensor:inputs})
        # However, we do not allow this yet!
        # print(f"network.Classifier.class_scores: inputs={type(inputs)}")
        inputs = np.asarray(inputs)
        scores = self.get_activations(inputs, self.score_layer)
        return scores


class Autoencoder(Network, method='network_changed'):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._encoder = None
        self._decoder = None

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder


class VariationalAutoencoder(Autoencoder):

    def sampleCode(self, n=1):
        pass

    def sampleData(self, n=1):
        pass

    def sampleCodeFor(self, input, n=1):
        pass

    def sampleDataFor(self, input, n=1):
        pass


class NetworkTool(Tool):

    _result: Tuple[str] = ('outputs')
    _internal_arguments: Tuple[str] = ('inputs_')
    _internal_result: Tuple[str] = ('outputs_')

    @property
    def network(self) -> Network:
        return self._network

    @network.setter
    def network(self, network: Network) -> None:
        self._network = Network

    def _preprocess(self, inputs, *args, **kwargs) -> Data:
        data = super()._preprocess(*args, **kwargs)
        if inputs is not None:
            data.add_attribute('_inputs', self._network.preprocess(inputs))
        return data

    def _process(self, inputs: Any) -> Any:
        """Default operation: propagate data through the network.
        """
        output_layer = self._network.output_layer_id()
        return self._network.get_activations(inputs, output_layer)

    def _postprocess(self, data: Data, what: str) -> None:
        if what == 'outputs':
            data.add_attribute(what, self._network.postprocess(data.outputs_))
        elif not hasattr(data, what):
            raise ValueError(f"Unknown property '{what}' for tool {self}")

#
# FIXME[old]: util functions
#


def remove_batch_dimension(shape: tuple) -> tuple:
    """Set the batch dimension to None as it does not matter for the
    Network interface."""
    shape = list(shape)
    shape[0] = None
    shape = tuple(shape)
    return shape


def transpose_channel_axis(array_or_shape: Union[np.ndarray, tuple],
                           input_format: str = None, output_format: str = None
                           ) -> Union[np.ndarray, tuple]:
    """Convert channel first to channel last format or vice versa.

    Parameters
    ----------
    array_or_shape
        The array or shape tuple to be converted. Needs to be at least rank 3.

    input_format
        Either 'channels_first' or 'channels_last'. If not given, opposite
        of `output_format` is used.

    output_format
        Either 'channels_first' or 'channels_last'. If not given, opposite
        of `input_format` is used.

    Returns
    -------
    The converted numpy array.

    """
    is_tuple = False
    # Check inputs.
    if isinstance(array_or_shape, np.ndarray):
        if array_or_shape.ndim < 3:
            # raise ValueError('Tensor needs to be at least of rank 3
            # but is of rank {}.'.format(array_or_shape.ndim))
            # Non-image arrays don't have to be converted.
            return array_or_shape
    elif isinstance(array_or_shape, tuple):
        # Convert to list for assignment later, but set a flag to
        # remember it was a tuple.
        array_or_shape = list(array_or_shape)
        is_tuple = True
        if len(array_or_shape) < 3:
            # raise ValueError('Shape needs to be at least of rank 3
            # but is of rank {}.'.format(len(array_or_shape)))
            return array_or_shape
    else:
        raise TypeError("Input must be either tuple or ndarray "
                        f"but is {type(array_or_shape)}")
    # Do the conversion based on the arguments.
    if input_format == output_format:
        # No conversion needed to same input and output format.
        return array_or_shape
    elif ((input_format == 'channels_first' and
           output_format == 'channels_last') or
          (input_format == 'channels_first' and output_format is None) or
          (input_format is None and output_format == 'channels_last')):
        if isinstance(array_or_shape, np.ndarray):
            return np.moveaxis(array_or_shape, 1, -1)
        elif is_tuple:
            num_channels = array_or_shape[1]
            del array_or_shape[1]
            array_or_shape.append(num_channels)
            array_or_shape[-1] = num_channels
            return tuple(array_or_shape)
    elif ((input_format == 'channels_last'
           and output_format == 'channels_first') or
          (input_format == 'channels_last' and output_format is None) or
          (input_format is None and output_format == 'channels_first')):
        if isinstance(array_or_shape, np.ndarray):
            return np.moveaxis(array_or_shape, -1, 1)
        elif is_tuple:
            num_channels = array_or_shape[-1]
            del array_or_shape[-1]
            array_or_shape[1].insert(1, num_channels)
            return tuple(array_or_shape)
    else:
        raise ValueError("Data format must be either 'channels_last' "
                         "or 'channels_first,' but is "
                         f"input_format='{input_format}' and "
                         f"output_format='{output_format}'")
