from __future__ import absolute_import

import keras

from . import Network
from .keras_tensorflow import Network as KerasTensorFlowNetwork


def build_keras_network(model_file: str) -> Network:
    """
    Wrapper for the more low-level KerasNetwork classes. Figures out the backend and
    returns an istance of the appropriate class.

    Parameters
    ----------
    model_file
        Path to the .h5 model file.
    Returns
    -------

    """
    current_backend = keras.backend.backend()
    if current_backend == 'tensorflow':
        return KerasTensorFlowNetwork(model_file)
    elif current_backend == 'theano':
        raise NotImplementedError('Theano not supported yet.')
    elif current_backend == 'CNTK':
        raise NotImplementedError('CNTK not supported yet.')
    else:
        raise ValueError('You are using the backend {}, which is not a valid Keras backend'.format(current_backend))
