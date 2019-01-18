from typing import Iterator

from . import Network

def load_alexnet() -> Network:
    if 'ALEXNET_MODEL' in os.environ:
        model_path = os.getenv('ALEXNET_MODEL', '.')
    else:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  'models', 'example_tf_alexnet')
    checkpoint = os.path.join(model_path, 'bvlc_alexnet.ckpt')
    if not os.path.isfile(checkpoint + '.meta'):
        raise ValueError('AlexNet checkpoint files do not exist. You can generate them by downloading "bvlc_alexnet.npy" and then running models/example_tf_alexnet/alexnet.py.')

    from .tensorflow import Model as TensorflowNetwork
    network = TensorflowNetwork(checkpoint=checkpoint)
    
    from datasources.imagenet_classes import class_names
    network.set_output_labels(class_names)
    return network


def load_lucid(name: str) -> Network:
    """Load the Lucid model with the given name.

    Returns
    -------
    model: LucidModel
        A reference to the LucidModel.
    """

    from .lucid import Network as LucidNetwork
    network = LucidNetwork(name=name)
    return network

def lucid_names() -> Iterator[str]:
    """Provide an iterator vor the available Lucid model names.

    Returns
    -------
    names: Iterator[str]
        An iterartor for the model names.
    """
    import lucid.modelzoo.nets_factory as nets
    return nets.models_map.keys()
