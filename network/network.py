import numpy as np

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

    def get_activations(selfs, layer_ids: list, input_samples: np.ndarray) -> list:
        """Gives activations values of the network/model
        for a given layername and an input (inputsample).
        Parameters
        ----------
        layer_ids: The layers the activations should be fetched for.
        input_samples: Array of samples the activations should be computed for.

        Returns
        -------
        Array of shape (input_samples, image_height, image_width, feature_maps).

        """
        raise NotImplementedError

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


