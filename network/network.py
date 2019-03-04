from typing import Tuple, Any, Union, List

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import functools
import operator
import numpy as np

from frozendict import FrozenOrderedDict

import importlib

from .util import convert_data_format
from util import Identifiable

from base.observer import BusyObservable, busy

# FIXME[design]: we should decide on some points:
#
#  * provide some information on the loaded_network
#     - version of the framework
#     - general info: number of parameters / memory usage
#     - GPU usage
#
#  * provide some information on the layer
#


class Network(Identifiable, BusyObservable, method='network_changed',
              changes=['state_changed', 'weights_changed']):
    """Abstract Network interface for all frameworks.

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

    We have decided to use a batch first, channel last ordering, that
    is ``NHWC``, i.e, ``(batch, height, width, channel)`` as this
    seems to be the natural ordering for RGB images. However, there
    may also be arguments against this ordering. Rüdiger has mentioned
    that cuDNN `requires channel first data
    <https://caffe2.ai/docs/tutorial-image-pre-processing.html#null__caffe-prefers-chw-order>`

    .. note::
        It may be more useful to be able to specify the desired order,
        either globally for a Network, in each method that gets or
        returns array data.


    Attributes
    ----------
    layer_dict: Dict[str,Layer]
   
    _output_labels: List[str]
        The output labels for this network. Only meaningful
        if this Network is a classifier.
    """

    
    @classmethod
    def framework_available(cls):
        """Check if the underlying framework is available (installed
        on this machine).
        """
        return false

    @classmethod
    def import_framework(cls):
        """Import the framework (python modules) required by this network.
        """
        pass # to be implemented by subclasses
    

    # FIXME[todo]: add additional parameters, e.g. cpu / gpu
    @classmethod
    def load(cls, module_name, *args, **kwargs):
        logger.info(f"Loading module: {module_name}")
        module = importlib.import_module(module_name)
        network_class = getattr(module, "Network")
        network_class.import_framework()
        instance = network_class(*args, **kwargs)
        return instance

    # ------------ Public interface ----------------

    def __init__(self, id=None, **kwargs):
        """

        Parameters
        ----------
        **kwargs
            data_format: {'channels_last', 'channels_first'}
                The place of the color channels in the tensors.
        """
        # Prohibited instantiation of base class.
        if self.__class__ == Network:
            raise NotImplementedError('Abstract base class Network '
                                      'cannot be used directly.')

        super().__init__(id)

        # Every loaded_network should know which data format it is using.
        # Default is channels_last.
        data_format = kwargs.get('data_format', 'channels_last')
        self._data_format = data_format

        # Create the layer representation.
        self.layer_dict = self._create_layer_dict()

        self._output_labels = None


    def __getitem__(self, item):
        """Provide access to the layers by number. Access by id is provided
        via :py:attr:`layer_dict`."""
        return tuple(self.layer_dict.values())[item]

    @busy
    def get_activations(self, layer_ids: Any,
                        input_samples: np.ndarray,
                        data_format: str='channels_last') -> Union[np.ndarray, List[np.ndarray]]:
        """Gives activations values of the loaded_network/model
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
        data_format: {'channels_last', 'channels_first'}
            The format in which the data is provided. Either "channels_first" or "channels_last".

        Returns
        -------
        Array of shape (input_samples, image_height, image_width, feature_maps).

        """
        # Check whether the layer_ids are actually a list.
        layer_ids, is_list = self._force_list(layer_ids)
        # Transform the input_sample appropriate for the loaded_network.
        input_samples = self._transform_input(input_samples, data_format)
        activations = self._compute_activations(layer_ids, input_samples)
        # Transform the output to stick to the canocial interface.
        activations = [self._transform_outputs(activation, data_format) for activation in activations]
        # If it was just asked for the activations of a single layer, return just an array.
        if not is_list:
            activations = activations[0]
        return activations

    def get_net_input(self, layer_ids: Any,
                      input_samples: np.ndarray,
                      data_format: str='channels_last') -> Union[np.ndarray, List[np.ndarray]]:
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
        Array of shape (input_samples, image_height, image_width, feature_maps).

        """
        # Check whether the layer_ids are actually a list.
        layer_ids, is_list = self._force_list(layer_ids)
        # Transform the input_sample appropriate for the loaded_network.
        input_samples = self._transform_input(input_samples, data_format)
        activations = self._compute_net_input(layer_ids, input_samples)
        # Transform the output to stick to the canocial interface.
        activations = [self._transform_outputs(activation, data_format) for activation in activations]
        # If it was just asked for the activations of a single layer, return just an array.
        if not is_list:
            activations = activations[0]
        return activations


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

    def _compute_activations(self, layer_ids: list,
                             input_samples: np.ndarray) -> List[np.ndarray]:
        """To be implemented by subclasses.
        Computes a list of activations from a list of layer ids.
        """
        raise NotImplementedError

    def _compute_net_input(self, layer_ids: list,
                           input_samples: np.ndarray) -> np.ndarray:
        """To be implemented by subclasses.
        Computes a list of net inputs from a list of layer ids."""
        raise NotImplementedError

    def _create_layer_dict(self) -> FrozenOrderedDict:
        """Create the mapping from layer ids to layer objects.

        Returns
        -------

        """
        raise NotImplementedError

    # ---------------------- Private helper functions -------------------------

    def _transform_input(self, inputs: np.ndarray,
                         data_format: str) -> np.ndarray:
        """Fills up the ranks of the input, e.g. if no batch size was
        specified and converts the input to the data format of the model.

        Parameters
        ----------
        inputs
            The inputs fed to the network.
        data_format: {'channels_last', 'channels_first'}
            The data format of inputs.

        Returns
        -------
        The transformed input.
        """

        inputs = self._fill_up_ranks(inputs)
        inputs = convert_data_format(inputs, input_format=data_format,
                                     output_format=self._data_format)

        return inputs


    def _transform_outputs(self, outputs: np.ndarray,
                           data_format: str) -> np.ndarray:
        """Convert output values into a desired data format.
        """
        return convert_data_format(outputs, input_format=self._data_format,
                                   output_format=data_format)

    def _fill_up_ranks(self, inputs: np.ndarray) -> np.ndarray:
        """Fill up the ranks of the input tensor in case no batch or
        color dimension is provided.

        Parameters
        ----------
        inputs
            The inputs fed to the network.
        data_format: {'channels_last', 'channels_first'}
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

    @staticmethod
    def _force_list(maybe_list: Union[List, Any]) -> Tuple[list, bool]:
        """Turn something into a list, if it is none.
        
        Returns
        -------
        maybe_list: list
            The input, turned into a list if necessary.
        is_list: bool
            A flag indicating whether the input was a list or not

        """
        is_list = True
        if not isinstance(maybe_list, list):
            is_list = False
            maybe_list = [maybe_list]
        return maybe_list, is_list

    ###------------------------- methods from Ulf ----------------------

    def _canonical_input_shape(self, input_shape : tuple) -> tuple:
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
        elif len(input_shape.shape) == 3:
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
        return network_input_shape[-1] if len(network_input_shape)>2 else 0

    ### -------------------- methods for accessing layer attributes ---------

    def input_layer_id(self):
        first_layer_id = next(iter(self.layer_dict.keys()))
        return first_layer_id

    def output_layer_id(self):
        for last_layer_id in iter(self.layer_dict.keys()): pass
        return last_layer_id

    def get_input_layer(self):
        return self.layer_dict[self.input_layer_id()]

    def get_output_layer(self):
        return self.layer_dict[self.output_layer_id()]
    
    def get_input_shape(self, include_batch = True) -> tuple:
        """Get the shape of the input data for the network.
        """
        shape = self.get_layer_input_shape(self.input_layer_id())
        return shape if include_batch else shape[1:]


    def get_output_shape(self, include_batch = True) -> tuple:
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


###------------------------- special methods for convolutional layers -----------------

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


    def get_layer_padding(self, layer_id) -> (int,int):
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


    def get_layer_output_padding(self, layer_id) -> (int,int):
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


    def set_output_labels(self, labels: List[str]) -> None:
        """Provide labels for the output layer. This is useful for 
        networks that are used as classifier. It allows to report
        results in form of human readable labels instead of just
        the indices of units within the output layer.

        Parameters
        ----------
        labels:
            A list providing the labels.

        Raises
        ------
        ValueError
            If the length of the list does not match the number of
            neurons in the output layer.
        """
        output_shape = self.get_output_shape(include_batch = False)
        if len(output_shape) > 1 or output_shape[0] != len(labels):
            raise ValueError("Labels do not fit to the output layer")
        self._output_labels = labels

    def get_label_for_class(self, index: int) -> str:
        """Get a class label for the given class index.

        Parameters
        ----------
        index:
            The class index, i.e. the index of the unit in
            the classification (=output) layer.

        Returns
        -------
        label:
            The label for the given class.
        """
        return (str(index) if self._output_labels is None
                else self._output_labels[index])

    def classify_top_n(self, input_samples: np.ndarray, n = 5):
        """Output the top-n classes for given input.
        """

        # FIXME[todo]: at least in tensorflow it seems possible to feed a list of inputs!
        #output = sess.run(network_output_tensor, feed_dict = {network_input_tensor:input_samples})
        # However, we do not allow this yet!
        input_samples = np.asarray(input_samples)
        output = self.get_activations(self.output_layer_id(), input_samples)

        for input_im_ind in range(output.shape[0]):
            inds = np.argsort(output)[input_im_ind,:]
            print("Image", input_im_ind)
            for i in range(n):
                print("  {}: {} ({})".format(i,self._output_labels[inds[-1-i]],
                                             output[input_im_ind, inds[-1-i]]))

from base.controller import Controller

class NetworkController(Controller):
    pass
