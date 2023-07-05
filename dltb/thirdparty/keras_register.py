"""Register torch implementations.

This module should only be imported if `keras` (or a `tensorflow` with
keras) is installed.
"""

# standard imports
import sys
import os
import logging
import importlib

# toolbox imports
from dltb.config import config
from dltb.network import Network
from dltb.util.importer import importable, add_preimport_hook
from . import THIRDPARTY

if not importable('keras') and not importable('tensorflow'):
    raise ImportError("Registering keras definitions failed "
                      "as neither 'keras' nor 'tensorflow' are importable.")

# logging
KERAS = THIRDPARTY + '.keras'
LOG = logging.getLogger(KERAS)


config.add_property('keras_prefer_tensorflow', default=True,
                    description="If 'True', the 'tensorflow.keras' "
                    "implementation wil be preferred over and replace "
                    "the old 'keras' module.")


def patch_keras_import(fullname: str, path, target=None) -> None:
    # pylint: disable=unused-argument
    """Patch the keras import process to use the TensorFlow implementation
    of Keras (`tensorflow.keras`) instead of the standard Keras.IO
    implementation (`keras`).

    """
    if not config.keras_prefer_tensorflow and importable('keras'):
        # The only way to configure the keras backend appears to be
        # via environment variable. We thus inject one for this
        # process. Keras must be loaded after this is done
        # os.environ['KERAS_BACKEND'] = 'theano'
        os.environ['KERAS_BACKEND'] = 'tensorflow'

        # Importing keras unconditionally outputs a message
        # "Using [...] backend." to sys.stderr (in keras/__init__.py).
        # There seems to be no sane way to avoid this.

        return  # nothing else to do ...

    keras = None
    if 'tensorflow.keras' in sys.modules:
        keras = sys.modules['tensorflow.keras']
    elif 'tensorflow' in sys.modules:
        keras = sys.modules['tensorflow'].keras
    else:
        module_spec = importlib.util.find_spec('tensorflow.keras')
        if module_spec is not None:
            # Load the module from module_spec.
            # Remark: This actually seems not be necessary,
            # as for some reason find_spec() already puts
            # the module in sys.modules.
            keras = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(keras)

    if keras is not None:
        LOG.info("Mapping 'keras' -> 'tensorflow.keras'")
        sys.modules['keras'] = keras
    else:
        LOG.info("Not mapping 'keras' -> 'tensorflow.keras'")


add_preimport_hook('keras', patch_keras_import)


# Adapt autoencoder to MNIST dimensions
_ORIGINAL_DIM = 28 * 28
# _intermediate_dim = 512
# _latent_dim = 2
Network.register_instance('mnist-vae', 'models.example_keras_vae_mnist',
                          'KerasAutoencoder', original_dim=_ORIGINAL_DIM)

Network.register_instance('mnist-keras-tf', 'models.example_keras_advex_mnist',
                          'KerasMnistClassifier')

# FIXME[todo/old]
#Network.register_instance('mnist-keras', 'models.example_keras_advex_mnist', '?')

Network.register_instance('resnet-keras', KERAS + 'network',
                          'ApplicationsNetwork', model='ResNet50')
