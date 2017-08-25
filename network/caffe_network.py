from network import BaseNetwork
import caffe
import numpy as np
from typing import Union, List

class CaffeNetwork(BaseNetwork):

    # Layer types that have to be converted.
    LAYER_TYPES_TO_CONVERT = {
        'Convolution',
        'Input'
    }

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

        return list(self._caffenet.layer_dict.keys())

    def get_layer_input_shape(self, layer_id) -> tuple:
        """
        Give the shape of the input of the given layer.

        Parameters
        ----------
        layer_id

        Returns
        -------
        """
        previous_layer_idx = list(self._caffenet.blobs.keys()).index(layer_id) - 1
        # Input to the first layer is, is the output of the layer, since in
        # Caffe the first layer will always be an Input/Data layer.
        if previous_layer_idx == -1:
            previous_layer_idx = 0
        caffe_input_shape = list(self._caffenet.blobs.values())[previous_layer_idx].data.shape
        # Set batch dimension to None as it does not matter for other frameworks.
        caffe_input_shape = self._remove_batch_dimension(caffe_input_shape)
        input_shape = self._convert_caffe_data_format(layer_id, caffe_input_shape)
        return input_shape

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
        caffe_output_shape = self._caffenet.blobs[layer_id].shape
        caffe_output_shape = self._remove_batch_dimension(caffe_output_shape)
        return self._convert_caffe_data_format(layer_id, caffe_output_shape)

    def get_activations(self, layer_ids, input_samples: np.ndarray) -> List[np.ndarray]:
        """Gives activations values of the network/model
        for a given layername and an input (inputsample).
        Parameters
        ----------
        layer_ids: The layers the activations should be fetched for.
        input_samples: Array of samples the activations should be computed for.

        Returns
        -------
        Array of shape (input_samples, image_height, image_width, feature_maps), i.e. NHWC.

        """
        layer_ids, input_samples = self._check_get_activations_args(layer_ids, input_samples)

        # Reshape the input layer to match the batch size.
        # Assuming the first layer is the input layer.
        input_blob = next(iter(self._caffenet.blobs.values()))
        old_input_shape = input_blob.data.shape
        new_input_shape = list(old_input_shape)
        # Reshape the network. Change only the batch size.
        # The batch size is otherwise fixed in the model definition.
        new_input_shape[0] = input_samples.shape[0]
        input_blob.reshape(*new_input_shape)
        self._caffenet.reshape()
        # Feed the input into the network and forward it.
        input_blob.data[...] = input_samples
        self._caffenet.forward()

        activations = [self._convert_caffe_data_format(
                            layer_id, self._caffenet.blobs[layer_id].data
                      )
                       for layer_id in layer_ids]
        return activations

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
        caffe_weights = self._caffenet.params[layer_id][0].data
        return self._convert_caffe_data_format(layer_id, caffe_weights)

    def _check_get_activations_args(self, layer_ids, input_samples: np.ndarray):
        layer_ids, input_samples = super()._check_get_activations_args(layer_ids, input_samples)
        # Swap axes from NHWC -> NCHW, since caffe requires NCHW format.
        print('check in caffe', input_samples.shape, type(input_samples))
        input_samples = self._convert_data_format(input_samples, output_format='channels_first')
        return layer_ids, input_samples

    def _remove_batch_dimension(self, shape: tuple) -> tuple:
        """Set the batch dimension to None as it does not matter for the Network interface."""
        shape = list(shape)
        shape[0] = None
        shape = tuple(shape)
        return shape

    def _convert_caffe_data_format(self, layer_id,
                                   array_or_shape: Union[np.ndarray, tuple]) -> Union[np.ndarray, tuple]:
        if self._must_convert(layer_id):
            return self._convert_data_format(
                array_or_shape,
                output_format='channels_last'
            )
        else:
            return array_or_shape

    def _must_convert(self, layer_id) -> bool:
        """Check whether the layer is of a type such that the data format hast to be changed."""
        return self._caffenet.layer_dict[layer_id].type in self.LAYER_TYPES_TO_CONVERT
