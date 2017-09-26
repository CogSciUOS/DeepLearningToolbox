from __future__ import absolute_import
from typing import List

import numpy as np
from collections import OrderedDict
from frozendict import FrozenOrderedDict

import keras
from keras.models import Model, load_model


from . import Network as BaseNetwork


class Network(BaseNetwork):
    """Abstract base class for the keras networks for the specific backends.

    Implements only the functionality that can be efficiently implemented in pure
    Keras.
    """
    def __init__(self, **kwargs):
        """
        Load Keras model.
        Parameters
        ----------
        model_file
            Path to the .h5 model file.
        """
        # Set learning phase to train in case setting to test would affect gradient computation.
        # TODO: Set it to test, to eliminate dropout, check for gradient computation later.
        keras.backend.set_learning_phase(0)
        if 'model_file' in kwargs.keys():
            self._model = load_model(kwargs.pop('model_file'))
        elif 'model' in kwargs.keys():
            self._model = kwargs['model']
        # Check the data format.
        kwargs['data_format'] = keras.backend.image_data_format()
        super().__init__(**kwargs)

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


















