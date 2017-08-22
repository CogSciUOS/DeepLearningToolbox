
import numpy as np

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


