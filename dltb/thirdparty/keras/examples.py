from dltb.network import Network

def keras(backend: str, cpu: bool,
          model_file: str='models/example_keras_mnist_model.h5') -> Network:
    # actually: KerasNetwork
    '''
    Visualise a Keras-based network

    Parameters
    ----------
    backend     :   str
                    Name of the Keras backend
                    (currently only "tensorflow" or "theano")
    cpu         :   bool
                    Whether to use only cpu, not gpu
    model_file       :   str
                    Filename where the model_file is located (in hdf5)

    Returns
    -------
    network.network.Network
        The concrete network instance to visualise

    Raises
    ------
    RuntimeError
        In case of unknown backend

    '''
    if backend == 'tensorflow':
        return Network.load('network.keras_tensorflow',
                            model_file=model_file)
    elif backend == 'theano':
        return Network.load('network.keras_theano',
                            model_file=model_file)
    else:
        raise RuntimeError('Unknown backend {backend}')
