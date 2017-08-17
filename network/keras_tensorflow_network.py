from network import KerasNetwork
import keras
import numpy as np

class KerasTensorFlowNetwork(KerasNetwork):

    def __init__(self, model_file: str):
        """
        Load Keras model.
        Parameters
        ----------
        modelfile_path
            Path to the .h5 model file.
        """
        super().__init__(model_file)
        self._sess = keras.backend.get_session()



    def get_activations(self, layer_ids: list, input_samples: np.ndarray) -> list:
        """Gives activations values of the network/model
        for a given layername and an input (inputsample).
        Parameters
        ----------
        layer_ids: The layers the activations should be fetched for.
        input_samples: Array of samples the activations should be computed for.

        Returns
        -------

        """
        activation_tensors  = [self._model.get_layer(layer_id).output for layer_id in layer_ids]
        network_input_tensor = self._sess.graph.get_operations()[0].values()
        return self._sess.run(fetches=activation_tensors, feed_dict={network_input_tensor: input_samples})

