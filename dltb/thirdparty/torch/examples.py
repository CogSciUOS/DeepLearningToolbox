from dltb.network import Network

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
