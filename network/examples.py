# from network.keras import Network as KerasNetwork
# from network.torch import Network as TorchNetwork

from .network import Network

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


def torch(cpu: bool, model_file: str, net_class: str, parameter_file: str,
          input_shape: tuple) -> Network:  # actually: TorchNetwork
    '''
    Visualise a Torch-based network

    .. error:: Torch network currently does not work.

    Parameters
    ----------
    cpu : bool
        Whether to use only cpu, not gpu

    model_file : str
        Filename where the model is defined (a Python file with a
        :py:class:`torch.nn.Module` sublcass)

    net_class : str
        Name of the model_file class (see ``model_file``)

    parameter_file : str
        Name of the file storing the model weights (pickled torch weights)

    input_shape : tuple
        Shape of the input images

    Returns
    -------
    network: TorchNetwork
        The concrete network instance to visualize.

    '''
    # FIXME[todo]: Fix errors when running torch network
    return Network.load('network.torch',
                        model_file, parameter_file, net_class=net_class,
                        input_shape=input_shape, use_cuda=not cpu)


