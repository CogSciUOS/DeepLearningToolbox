import numpy as np
from typing import Any

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
            if self._is_channel_provided(input_samples.shape, network_input_shape):
                input_samples = input_samples[np.newaxis, ...]
            elif self._is_batch_provided(input_samples.shape, network_input_shape):
                input_samples = input_samples[..., np.newaxis]
            else:
                raise ValueError('Non matching input dimensions.')
        elif len(input_samples.shape) > 4:
            raise ValueError('Too many input dimensions. Should be maximally four instead of {}.'.format(
                len(input_samples.shape)
            ))

        return layer_ids, input_samples


    def _is_channel_provided(self, input_sample_shape: tuple, network_input_shape: tuple) -> bool:
        return input_sample_shape == network_input_shape[1:]

    def _is_batch_provided(self, input_sample_shape: tuple, network_input_shape: tuple) -> bool:
        return input_sample_shape[1:3] == network_input_shape[1:3]























