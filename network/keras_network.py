

from network.network import BaseNetwork
import sys
import keras
from keras.models import Model, load_model
import numpy as np
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.layers import Conv2D, MaxPooling2D

class KerasNetwork(BaseNetwork):

    def __init__(self, modelfile_path: str):
        """
        Load Keras model.
        Parameters
        ----------
        modelfile_path
            Path to the .h5 model file.
        """
        print("KerasNetwork: Using keras version: {}".format(keras.__version__))
        self._model = load_model(modelfile_path)

    def get_layer_id_list(self):
        """
        Get list of layer ids.
        Returns
        -------

        """
        return [layer_spec['config']['name'] for layer_spec in self._model.get_config()]

    def get_layer_input_shape(self, layer_id) -> tuple:
        """
        Give the shape of the input of the given layer.

        Parameters
        ----------
        layer_id

        Returns
        -------
        """
        return self._model.get_layer(layer_id).get_input_shape_at(0)

    def get_layer_output_shape(self, layer_id) -> tuple:
        """
        Give the shape of the output of the given layer.

        Parameters
        ----------
        layer_id

        Returns
        -------

        """
        return self._model.get_layer(layer_id).get_output_shape_at(0)

    def get_activations(self, layer_id, input_samples: np.ndarray) -> np.ndarray:
        """
        Gives activations values of the network/model
        for a given layername and an input (inputsample).
        Parameters
        ----------
        layer_id
        inputsample

        Returns
        -------

        """
        # Build up a new model from the first layer up to the layer we care about.
        input_samples = input_samples[:, :, :, np.newaxis]
        intermediate_layer_model = Model(inputs=self._model.input,
                                         outputs=self._model.get_layer(layer_id).output)
        intermediate_output = intermediate_layer_model.predict(input_samples)
        return intermediate_output

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
        idx = 0
        for layer_spec in self._model.get_config():
            if layer_spec['config']['name'] == layer_id:
                return self._model.get_weights()[idx]
            else:
                # Check whether the layer has weights.
                if layer_spec['class_name'] in {'Dense', 'Conv2D'}:
                    # Since a layer would have weights and biases, increment by four.
                    idx += 2
        raise ValueError('Layer id not in model.')


















