import os
import numpy as np
import tensorflow as tf
from network.network import BaseNetwork

class TensorFlowNetwork(BaseNetwork):
    """Abstract network interface for all frameworks."""

    def __init__(self, model_file: str):
        self._sess = sess=tf.Session()
        saver = tf.train.import_meta_graph(os.path.join(model_file, '.meta'))
        saver.restore(sess, model_file)



    def get_layer_id_list(self):
        """
        Get list of layer ids.
        Returns
        -------

        """



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


