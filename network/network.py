import operator
import functools
import numpy as np
from typing import Any, Union

# FIXME[design]: we should decide on some points:
#
#  * provide some information on the network
#     - version of the framework
#     - general info: number of parameters / memory usage 
#     - GPU usage
#
#  * provide some information on the layer

class BaseNetwork:
    """Abstract network interface for all frameworks.

    The network API will allow for to order the dimensions in data
    arrays in a way independent of the underlying network
    implementation. There seem to be different orderings applied in
    different frameworks:
    
      * tensorflow default is to use channel last ("NHWC")
        (can be changed to channel first: data_format="NCHW")
        https://www.tensorflow.org/api_docs/python/tf/nn/conv2d

      * pytorch only supports channel first (N,C,H,W)
        http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d

      * pycaffe: ?
    
      * theano: ?
    
    We have decided to use a batch first, channel last ordering, that
    is "NHWC", i.e, (batch, height, width, channel) as seems to be the
    natural ordering for RGB images. However, there may also arguments
    against this ordering. Ruediger has mentioned that cuDNN requires
    channel first data [refernce?]

    FIXME[todo]: it may be more useful to be able to specify the
    desired order, either globally for a network, at in each method
    that gets or returns array data.

    """

    def __init__(self):
        raise NotImplementedError


    @property
    def layer_ids(self) -> list:
        """
        Get list of layer ids.
        Returns
        -------

        """

        raise NotImplementedError

    
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
        raise NotImplementedError


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
        raise NotImplementedError


    def layer_is_convolutional(self, layer_id) -> bool:
        """Check if the given layer is a convolutional layer. If so,
        additional information can be obtained by the methods
        get_layer_kernel_size, get_layer_channels, get_layer_stride,
        get_layer_padding, and get_layer_dilation.

        Parameters
        ----------
        layer:
            Identifier of a layer in this network.

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
            Identifier of a layer in this network.

        Raises
        ------
        ValueError
            If the given layer is not convolutional.
        """
        if not self.layer_is_convolutional(layer_id):
            raise ValueError('Not a convolutional layer: {}'.format(layer_id))


    def get_activations(self, layer_ids: Any, input_samples: np.ndarray) -> tuple:
        """Gives activations values of the network/model
        for a given layername and an input (input_sample).
        
        Parameters
        ----------
        layer_ids
            The layers the activations should be fetched for. Single
            layer_id or list of layer_ids.
        input_samples       
             For multi-channel, two-dimensional data, we expect the
             input data to be given in with channel last, that is
             (N,H,W,C). For plain data of dimensionality D we expact
             batch first (N,D).

        Returns
        -------
        Array of shape (input_samples, image_height, image_width, feature_maps).

        """
        ## Checking whether the layer_ids are actually a list.
        if not isinstance(layer_ids, list):
            layer_ids = [layer_ids]
        return layer_ids, self._canonical_input_data(input_samples)


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
        raise NotImplementedError
    
    def _get_layer_weights_shape(self, layer_id) -> tuple:
        weights = self.get_layer_weights(layer_id)
        if weights is None:
            return None
        # FIXME[hack/old]: old implementation returns list: (weights,bias)
        if isinstance(weights,list):
            if len(weights) == 0: # e.g. dropout layer
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
        raise NotImplementedError

    
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
        return (self.get_layer_weights(self,layer_id).size
                + self.get_layer_biases(self,layer_id).size)


    def get_layer_input_units(self, layer_id) -> int:
        """The number of input units of this layer. For convolutional
        layers this will be the number of all units in all channels.
        
        Parameters
        ----------
        layer_id:
            Identifier of a layer in this network.

        Raises
        ------
        ValueError:
            If the layer_id fails to identify a layer in this network.
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
            Identifier of a layer in this network.

        Raises
        ------
        ValueError:
            If the layer_id fails to identify a layer in this network.
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
            Identifier of a convolutional layer in this network.

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
            Identifier of a convolutional layer in this network.

        Raises
        ------
        ValueError:
            If the layer_id fails to identify a convolutional layer.
        """
        self._check_layer_is_convolutional(layer_id)
        layer_shape = self._get_layer_weights_shape(layer_id)
        return layer_shape[-1]


    def get_layer_kernel_size(self, layer_id) -> int:
        """The size of the kernel in a cross-correlation/convolution
        operation. This is just the spatial extension and does not
        include the number of channels.

        Parameters
        ----------
        layer_id:
            Identifier of a convolutional layer in this network.

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
            Identifier of a convolutional layer in this network.

        Raises
        ------
        ValueError:
            If the layer_id fails to identify a convolutional layer.
        """
        self._check_layer_is_convolutional(layer_id)
        raise NotImplementedError


    def _check_get_activations_args(self, layer_ids, input_samples: np.ndarray):
        """"""
        # Checking whether the layer_ids are actually a list.
        if not isinstance(layer_ids, list):
            layer_ids = [layer_ids]
        # Checking whether input samples was provided with all for channels.
        if len(input_samples.shape) == 2:
            # Only width and height means we are dealing with one grayscale image.
            input_samples = input_samples[np.newaxis, :, :, np.newaxis]
        elif len(input_samples.shape) == 3:
            # We have to decide whether the batch or the channel dimension is missing.
            # Ask the network what shape it expects. Since either the last three dimensions
            # (in case channel was provided) or second and third dimension (in case batch) was
            # provided have to match.
            first_layer_id = self.layer_ids[0]
            network_input_shape = self.get_layer_input_shape(first_layer_id)
            print('network input shape: ', network_input_shape)
            if self._is_channel_provided(input_samples.shape, network_input_shape):
                input_samples = input_samples[np.newaxis, ...]
            elif self._is_batch_provided(input_samples.shape, network_input_shape):
                input_samples = input_samples[..., np.newaxis]
            else:
                raise ValueError('Non matching input dimensions.')
        elif len(input_samples.shape) > 4:
            raise ValueError('Too many input dimensions. Should be maximally 4 instead of {}.'.format(
                len(input_samples.shape)
            ))

        def _is_channel_provided(self, input_sample_shape: tuple, network_input_shape: tuple) -> bool:
            """Check if a given shape includes a channel dimension.
            The channel dimension is assumed to be at the first axis.
            """
            return input_sample_shape == network_input_shape[1:]


    def get_layer_padding(self, layer_id) -> (int,int):
        """The padding for the cross-correlation/convolution operation, i.e,
        the number of rows/columns (on both sides) by which the input
        is extended (padded with zeros) before the operation is
        applied.

        Parameters
        ----------
        layer_id:
            Identifier of a convolutional layer in this network.

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
            Identifier of a convolutional layer in this network.

        Raises
        ------
        ValueError:
            If the layer_id fails to identify a convolutional layer.
        """
        self._check_layer_is_convolutional(layer_id)
        raise NotImplementedError



    def _convert_data_format(self, array_or_shape: Union[np.ndarray, tuple],
                                input_format: str=None, output_format: str=None) -> Union[np.ndarray, tuple]:
        """Convert channel first to channel last format or vice versa.

        Parameters
        ----------
        array_or_shape
            The array or shape tuple to be converted. Needs to be at least rank 3.
        input_format
            Either 'channels_first' or 'channels_last'. If not given, opposite of `output_format` is used.
        output_format
            Either 'channels_first' or 'channels_last'. If not given, opposite of `input_format` is used.

        Returns
        -------
        The converted numpy array.

        """
        is_tuple = False
        # Check inputs.
        if isinstance(array_or_shape, np.ndarray):
            print('found array')
            if array_or_shape.ndim < 3:
                raise ValueError('Tensor needs to be at least of rank 3 but is of rank {}.'.format(array_or_shape.ndim))
        elif isinstance(array_or_shape, tuple):
            # Convert to list for assignment later, but set a flag to remember it was a tuple.
            array_or_shape = list(array_or_shape)
            is_tuple = True
            if len(array_or_shape) < 3:
                raise ValueError('Shape needs to be at least of rank 3 but is of rank {}.'.format(len(array_or_shape)))
        else:
            raise TypeError('Input must be either tuple or ndarray but is {}'.format(type(array_or_shape)))
        # Do the conversion based on the arguments.
        if ((input_format == 'channels_first' and output_format == 'channels_last') or
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
        elif ((input_format == 'channels_last' and output_format == 'channels_first') or
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
            raise ValueError('Format must be eiher "channels_last" or "channels_first".')

    def get_layer_dilation(self, layer_id) -> (int, int):
        """The dilation for the cross-correlation/convolution operation, i.e,
        the horizontal/vertical offset between adjacent filter
        rows/columns.

        Parameters
        ----------
        layer_id:
            Identifier of a convolutional layer in this network.

        Raises
        ------
        ValueError:
            If the layer_id fails to identify a convolutional layer.
        """
        self._check_layer_is_convolutional(layer_id)
        raise NotImplementedError


    def get_layer_info(self, layername):
        """FIXME[todo]: we still have to decide on some info API

        Parameters
        ----------
        layername

        Returns
        -------
        """
        raise NotImplementedError


    def _canonical_input_data(self, input_samples: np.ndarray):
        """Check that given input data has the correct shape for feeding it to
        the network and adapt it otherwise.

        Returns
        -------
        ndarray:
            A version of input_samples having the expected form.
        """
            
        network_input_shape = self.get_layer_input_shape(self.layer_ids[0])

        ## Checking whether input samples was provided with all for
        ## channels.
        if len(input_samples.shape) == 2:
            ## Only width and height means we are dealing with one
            ## grayscale image.
            input_samples = input_samples[np.newaxis, :, :, np.newaxis]
        elif len(input_samples.shape) == 3:
            ## We have to decide whether the batch or the channel
            ## dimension is missing.  Ask the network what shape it
            ## expects. Since either the last three dimensions (in
            ## case channel was provided) or second and third
            ## dimension (in case batch) was provided have to match.
            if self._is_channel_provided(input_samples.shape,
                                         network_input_shape):
                input_samples = input_samples[np.newaxis, ...]
            elif self._is_batch_provided(input_samples.shape,
                                         network_input_shape):
                input_samples = input_samples[..., np.newaxis]

        ## sanity check
        if input_samples.shape[1:] != network_input_shape:
            raise ValueError('Incorrect input shape: expect {}, got {}'.
                             format(network_input_shape, input_samples.shape))

        ## Check the data type of the input data and make sure it
        ## fits to the network.
        # FIXME[hack/todo]: make this more flexible - check network
        # for suitable input types, make the conversion more general
        # than just np.uint8, ...
        if input_samples.dtype == np.uint8:
            input_samples = input_samples.astype(np.float32)/256
        elif input_samples.dtype != np.float32:
            raise ValueError('Incorrect input dtype: expect {}, got {}'.
                             format(np.float32, input_samples))

        return input_samples


    def _canonical_input_shape(self, input_shape : tuple) -> tuple:
        """Transform an input shape into the canonical form. For
        convolutional layers, this is channel last ordering (N,H,W,C).
        For flat input of dimension D it is (D,N).

        This method is intended to be used to determine the actual
        layer shapes for networks, that do not provide this
        information.
        
        Arguments
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
            ## from the network input.
            input_shape = (1, *input_shape, network_input_channels)
        elif len(input_samples.shape) == 3:
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


    def _is_channel_provided(self, input_sample_shape: tuple,
                             network_input_shape: tuple) -> bool:
        """Check if a given shape includes a channel dimension.
        The channel dimension is assumed to be at the first axis.
        """
        return input_sample_shape == network_input_shape[1:]


    def _is_batch_provided(self, input_sample_shape: tuple,
                           network_input_shape: tuple) -> bool:
        """Check if a given shape includes a batch dimension.
        The batch dimension is assumed to be at the first axis.
        """
        return input_sample_shape[1:3] == network_input_shape[1:3]


    def _get_number_of_input_channels(self) -> int:
        """Get the number of input channels for this network.
        This is the number of channels each input given to the network
        should have.  Usually this coincides with the number of
        channels in the first layer of the network.

        The standard implementation just extracts this from the input
        shape. However, in some networks the input shape is not
        available upon initialization and those networks should
        reimplement this method to provide the channel number,
        which should always be known.

        Returns
        -------
        int
            The number of input channels or 0 if the network does not
            have input channels.
        """
        network_input_shape = self.get_layer_input_shape(self.layer_ids[0])
        return network_input_shape[-1] if len(network_input_shape)>2 else 0


