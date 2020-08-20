# standard imports
from __future__ import absolute_import  # support for Python 2 code
from collections import OrderedDict  # new in version 2.7
import sys
import os
import logging
import importlib

# third party imports
import numpy as np
from dltb.thirdparty.keras import keras
from dltb.thirdparty.tensorflow import v1 as tf

# toolbox imports 
from .network import Network as BaseNetwork
from .keras import Network as KerasNetwork, Classifier
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


class ApplicationsNetwork(Network, Classifier):
    """One of the predefined and pretrained networks from the
    :py:mod:`keras.applications`.
    """
    KERAS_APPLICATIONS = 'tensorflow.keras.applications'

    def __init__(self, model: str, module: str = None, **kwargs) -> None:
        super().__init__(scheme='ImageNet', **kwargs)

        # evaluate the module argument
        if module is None:
            self._module, self._module_name = None, None
        elif isinstance(module, ModuleType):
            self._module, self._module_name = module, module.__name__
        elif isinstance(module, str):
            self._module = None
            self._module_name = (module if '.' in module else
                                 '.'.join((self.KERAS_APPLICATIONS, module)))
        else:
            raise TypeError("Argument 'module' should by module "
                            "or module name")

        # evaluate the model argument
        module_name = None
        self._model, self._model_name, self._model_function = None, None, None
        if isinstance(model, str):
            # from tensorflow.keras.applications import ResNet50
            # from tensorflow.keras.applications.resnet50 import ResNet50
            if "." in model:
                module_name, self._model_name = module.rsplit('.')
            else:
                self._model_name = model
                if self._module_name is None:
                    self._module_name = ".".join((self.KERAS_APPLICATIONS,
                                                  model.lower()))
        elif isinstance(model, FunctionType):
            # model.__module__ = 'tensorflow.python.keras.applications'
            # model.__name__ = 'wrapper'
            if model.__name__ == 'wrapper':
                self._model_name = model.__closure__[0].cell_contents.__name__
            else:
                self._model_name = model.__name
            self._model_function = model
        elif model is None:
            raise ValueError("No Model provided for ApplicationsNetwork")
        else:
            raise TypeError("Instantiation of ApplicationsNetwork with "
                            "bad model type ({type(model)}).")

        # Consistency Check
        if (module_name and self._module_name and
            module_name != self._module_name):
            # Inconsistent module names
            raise ValueError("Inconsistent module name: "
                             "'{self._module_name}' should be '{module_name}'")

    def _prepared(self) -> bool:
        """Check if this :py:class:`ApplicationsNetwork` has been prepared.

        """
        return self._model is not None and super()._prepared()

    def _prepare(self) -> None:
        if self._module is None:
            self._module = importlib.import_module(self._module_name)
            
        self.preprocess_input = getattr(self._module, 'preprocess_input')
        self.decode_predictions = getattr(self._module, 'decode_predictions')

        if self._model is None:
            if self._model_function is None:
                self._model_function = getattr(self._module, self._model_name)

            # Initialize the model (will download data if not yet present)
            self._model = self._model_function(weights='imagenet')
            # FIXME[warning]: (tensorflow.__version__ = '1.15.0')
            # From tensorflow_core/python/ops/resource_variable_ops.py:1630:
            #   calling BaseResourceVariable.__init__ (from
            #   tensorflow.python.ops.resource_variable_ops) with constraint
            #   is deprecated and will be removed in a future version.
            # Instructions for updating:
            # If using Keras pass *_constraint arguments to layers.
       
        super()._prepare()


    #
    # Preprocessing input
    #

    # Some models use images with values ranging from 0 to 1. Others
    # from -1 to +1. Others use the "caffe" style, that is not
    # normalized, but is centered.

    # From the source code, Resnet is using the caffe style.
    #

    # You don't need to worry about the internal details of
    # preprocess_input. But ideally, you should load images with the
    # keras functions for that (so you guarantee that the images you
    # load are compatible with preprocess_input).

    # Example: https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet/preprocess_input

    # i = tf.keras.layers.Input([None, None, 3], dtype = tf.uint8)
    # x = tf.cast(i, tf.float32)
    # x = tf.keras.applications.mobilenet.preprocess_input(x)
    # core = tf.keras.applications.MobileNet()
    # x = core(x)
    # model = tf.keras.Model(inputs=[i], outputs=[x])
    #
    # image = tf.image.decode_png(tf.io.read_file('file.png'))
    # result = model(image)

    # https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet/preprocess_input
    #
    # The images are converted from RGB to BGR, then each color
    # channel is zero-centered with respect to the ImageNet dataset,
    # without scaling.
