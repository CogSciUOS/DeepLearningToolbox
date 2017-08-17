from network import BaseNetwork
import keras
from keras.models import Model, load_model
import numpy as np
from typing import List

class KerasNetwork(BaseNetwork):
    """Abstract base class for the keras networks for the specific backends.

    Implements only the functionality that can be efficiently implemented in pure
    Keras.
    """
    def __init__(self, model_file: str):
        """
        Load Keras model.
        Parameters
        ----------
        modelfile_path
            Path to the .h5 model file.
        """
        # Set learning phase to train in case setting to test would affect gradient computation.
        keras.backend.set_learning_phase(1)
        self._model = load_model(model_file)
        self._layer_ids = self._compute_layer_ids()

    @property
    def layer_ids(self) -> list:
        """
        Get list of layer ids.
        Returns
        -------

        """
        return self._layer_ids

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

    def get_layer_weights(self, layer_id) -> List[np.ndarray]:
        """
        Returns weights INCOMING to the
        layer of the model
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
        return self._model.get_layer(layer_id).get_weights()

    def _compute_layer_ids(self):
        """

        Returns
        -------

        """
        # Apparently config can be sometimes a dict or simply a list of `layer_specs`.
        config = self._model.get_config()
        if isinstance(config, dict):
            layer_specs = config['layers']
        elif isinstance(config, list):
            layer_specs = config
        else:
            raise ValueError('Config is neither a dict nor a list.')
        return [layer_spec['config']['name'] for layer_spec in layer_specs]


















