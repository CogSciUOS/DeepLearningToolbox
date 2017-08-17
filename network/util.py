import keras
from network import BaseNetwork, KerasTensorFlowNetwork

def build_keras_network(model_file: str) -> BaseNetwork:
    """
    Wrapper for the more low-level KerasNetwork classes. Figures out the backend and
    returns an istance of the appropriate class.

    Parameters
    ----------
    model_file

    Returns
    -------

    """
    current_backend = keras.backend.backend()
    if current_backend == 'tensorflow':
        return KerasTensorFlowNetwork(model_file)
    elif current_backend == 'theano':
        raise NotImplementedError
    elif current_backend == 'CNTK':
        raise NotImplementedError
    else:
        raise ValueError('You are using the backend {}, which is not a valid Keras backend'.format(current_backend))
