# standard imports
from __future__ import absolute_import  # support for Python 2 code
from collections import OrderedDict  # new in version 2.7
import sys
import os
import logging

# third party imports
import numpy as np
from dltb.thirdparty.keras import keras
from dltb.thirdparty.tensorflow import v1 as tf

# toolbox imports
from .network import Network as BaseNetwork
from .keras import Network as KerasNetwork
from .tensorflow import Network as TensorflowNetwork

# logging
LOG = logging.getLogger(__name__)


# FIXME[concept]: this was part (and may become part) of the
# KerasTensorFlowLayerMixin class.
# A KerasNetwork with tensorflow backend may dynamically extend
# its layers to support this methods ...
class KerasTensorFlowLayerMixin:

    @property
    def input(self):
        return self._keras_layer_objs[0].input

    @property
    def output(self):
        return self._keras_layer_objs[-1].output


class Network(KerasNetwork, TensorflowNetwork):
    """This :py:class:`Network` combines the
    :py:class:`KerasNetwork` and the :py:class:`TensorflowNetwork`.
    The basic ideas are the following:

    * From the :py:class:`KerasNetwork` it inherits the construction
      of the layers from the structure of the Keras Model.

    * From the :py:class:`TensorflowNetwork` it inherits graph
      and session management.
    """

    def __init__(self, **kwargs):
        """
        Load Keras model.
        Parameters
        ----------
        modelfile_path
            Path to the .h5 model file.
        """
        print("**keras_tensorflow.Network:", kwargs)
        super().__init__(**kwargs)
        self._snapshot = None  # FIXME[old]: what is this? -> documentation or remove

    def _prepare_graph(self):
        if self._graph is None:
            self._graph = tf.Graph()
        else:
            print("network.keras: INFO: keras graph was already set")

    def _prepare_session(self):
        if self._session is None:
            #self._session = keras.backend.get_session()
            LOG.info("Keras-Tensorflow: prepare session with new session.")
            self._session = tf.Session(graph=self._graph)

            # FIXME[warning]:
            #   The name tf.keras.backend.set_session is deprecated.
            #   Please use tf.compat.v1.keras.backend.set_session instead.
            # (keras.__name__ is 'tensorflow.python.keras.api._v1.keras')
            # keras.backend.set_session(self._session)
            import tensorflow.compat.v1.keras.backend
            tensorflow.compat.v1.keras.backend.set_session(self._session)
        else:
            LOG.info("Keras-Tensorflow: prepare session "
                     "using existing session.")
            keras.backend.set_session(self._session)
        # FIXME[hack]: create a cleaner concept of keras/tensorflow preparation
        # super()._prepare_session()


    def _compute_activations(self, layer_ids: list, input_samples: np.ndarray):
        """Gives activations values of the loaded_network/model
        for a given layername and an input (inputsample).
        
        Parameters
        ----------
        layer_ids: The layers the activations should be fetched for.
        input_samples: Array of samples the activations should be computed for.

        Returns
        -------

        """
        activation_tensors = \
            [self.layer_dict[layer_id].output for layer_id in layer_ids]
        return self._feed_input(activation_tensors, input_samples)

    def _compute_net_input(self, layer_ids: list, input_samples: np.ndarray):
        ops = self._sess.graph.get_operations()
        net_input_tensors = []
        # To get the net input of a layer we take the second to last
        # operation as the net input.  This operation corresponds
        # usually to the addition of the bias.
        for layer_id in layer_ids:
            output_op = self.layer_dict[layer_id].output.op
            # Assumes the input to the activation is the net input.
            net_input_tensors.append(output_op.inputs[0])
        return self._feed_input(net_input_tensors, input_samples)

    def _feed_input(self, fetches: list, input_samples: np.ndarray):
        network_input_tensor = self._model.layers[0].input
        return self._sess.run(fetches=fetches,
                              feed_dict={network_input_tensor: input_samples})


    # FIXME[hack]: The method 
    #   network.keras_tensorflow.Network.get_input_shape()
    # is resolved to
    #   network.tensorflow.Network.get_input_shape()
    # which tries to access 
    #   self.get_input_tensor()
    # and so
    #   self._input_placeholder.outputs
    # but self._input_placeholder is not set.
    # Hence we will resolve to the original implementation
    #  network.network.get_input_shape()
    def _get_input_shape(self) -> tuple:
        """Get the shape of the input data for the network.
        """
        return BaseNetwork._get_input_shape(self)
    
    #
    # FIXME[old]: this was network.keras.KerasModel
    # 

    def reset(self):
        """Reset the model parameters (weights) to the last snapshot.
        """
        if self._snapshot is not None:
            self._keras_model.set_weights(self._snapshot)
            self._sess.run(tf.global_variables_initializer())

    @property
    def graph(self):
        return self._graph

    @property
    def session(self):
        return self._session


# FIXME[old]
from .keras import ApplicationsNetwork
