from network import BaseNetwork
import caffe
import numpy as np

class CaffeNetwork(BaseNetwork):

    def __init__(self, model_def: str, model_weights: str):
        """
        Load Caffe model.
        Parameters
        ----------
        model_def
            Path to the .prototxt model definition file.
        model_weights
            Path the .caffemodel weights file.
        """
        super().__init__()
        self._caffenet = caffe.Net(model_def,
                                   model_weights,
                                   caffe.TEST)

    @property
    def layer_ids(self) -> list:
        """
        Get list of layer ids.
        Returns
        -------

        """

        return tuple(self._caffenet.layer_dict.keys())

    def get_layer_input_shape(self, layer_id) -> tuple:
        """
        Give the shape of the input of the given layer.

        Parameters
        ----------
        layer_id

        Returns
        -------
        """
        previous_layer_index = list(self._caffenet.blobs.keys()).index(layer_id) - 1
        return list(self._caffenet.blobs.values())[previous_layer_index].data.shape

    def get_layer_output_shape(self, layer_id) -> tuple:
        """
        Give the shape of the output of the given layer.

        Parameters
        ----------
        layer_id

        Returns
        -------

        """
        # net.blobs stores the activations directly.
        return self._caffenet.blobs[layer_id].shape

    def get_activations(self, layer_ids: list, input_samples: np.ndarray) -> list:
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
        return [self._caffenet.blobs[layer_id].data for layer_id in layer_ids]

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
        return self._caffenet.params[layer_id][0].data

