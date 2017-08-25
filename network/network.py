
import numpy as np
from typing import Any, Union

# FIXME[design]: we should decide on some points:
#
#  * how to provide layer size (just "tuple") is not enough
#    (e.g. convolution as (channel, width, height) or
#    (width, height, channel), or (batch, height, channel), ...
#    this affects 
#     -> get_layer_input_shape(self, layer_id) -> tuple:
#     -> get_layer_output_shape(self, layer_id)
#    maybe multiple methods: number_of_channels(), get_filter_size()
#
#  * how to provide weights?
#    - why list? maybe tuple (pair = (weights, bias))
#    - what are the axis of the array? (order!)
#    - maybe optional parameter for convolutional layers to just
#      fetch individual channel
#    -> get_layer_weights(self, layer_id) -> List[np.ndarray]:
#
#  * how to provide activation?
#     - for one or multiple inputs?
#       maybe flexible, depending on input ...
#     - for one or multiple layers?
#       maybe flexible: layer_id as list/tuple (multiple) or non-list (single)?
#     - which axis?
#     - which datatype?
#     -> get_activations(selfs, layer_ids: list, input_samples: np.ndarray) -> list:
#
#  * provide some information on the network
#     - version of the framework
#     - general info: number of parameters / memory usage 
#     - GPU usage
#
#  * provide some information on the layer

class BaseNetwork:
    """Abstract network interface for all frameworks."""

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

        """
        raise NotImplementedError

    def get_activations(self, layer_ids: Any, input_samples: np.ndarray) -> tuple:
        """Gives activations values of the network/model
        for a given layername and an input (inputsample).
        Parameters
        ----------
        layer_ids
            The layers the activations should be fetched for. Single layer_id or list of layer_ids.
        input_samples
            Array of samples the activations should be computed for. Dimensions should be (N, H, W, C).

        Returns
        -------
        Array of shape (input_samples, image_height, image_width, feature_maps).

        """
        return self._check_get_activations_args(layer_ids, input_samples)

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
        raise NotImplementedError

    def get_layer_info(self, layername):
        """

        Parameters
        ----------
        layername

        Returns
        -------

        """
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

        return layer_ids, input_samples

    def _is_channel_provided(self, input_sample_shape: tuple, network_input_shape: tuple) -> bool:
        return input_sample_shape == network_input_shape[1:]

    def _is_batch_provided(self, input_sample_shape: tuple, network_input_shape: tuple) -> bool:
        return input_sample_shape[1:3] == network_input_shape[1:3]

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






















