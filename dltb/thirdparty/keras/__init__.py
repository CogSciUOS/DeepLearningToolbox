"""Providing keras functionality.

Importing this module (`dltb.thirdparty.keras`): This module is
usually automatically imported after the `keras` module has been
imported (as post-import dependency).  For this to work,
`dltb.thirdparty` has to be imported before importing `keras`. It is
not possible to import this module (`dltb.thirdparty.keras`) directly.

There exist Two different keras implementations:
* keras.io
* tensorflow.keras

Usually it seems preferable to use `tensorflow.keras` when available.
There is some code in `dltb.thirdparty` that patches the import
machinery to automatically import `tensorflow.keras` when `keras` is
imported.  For this to work, it is essential, that `dltb.thirdparty`
is imported before importing `keras`.

Also keras.io my use different backends:
* `theano`
* `tensorflow`

As theano is officially declared dead, usually the tensorflow backend
should be used.  This module will ensure that this is actually the
case.

"""

# standard imports
import os
import sys
import logging

if 'keras' not in sys.modules:
    raise ImportError("The module '{__name__}' must not be imported directly. "
                      "It is automatically imported following 'import keras'")

# thirdparty imports
import keras
# we are assuming that nowadays TensorFlow is the only backend
import tensorflow as tf

# toolbox imports
from ... import thirdparty
from ...config import config
from ...datasource import Datasource

# logging
LOG = logging.getLogger(__name__)

HAVE_KERAS_IO = sys.modules['keras'].__name__ == 'keras'
HAVE_TENSORFLOW_KERAS = sys.modules['keras'].__name__ == 'tensorflow.keras'

if HAVE_TENSORFLOW_KERAS:
    # from . import tensorflow
    # import tensorflow.keras as keras
    # import tensorflow.compat.v1.keras as keras
    # from tensorflow.compat.v1.keras import backend
    # sys.modules['keras.backend'] = backend
    pass

if HAVE_KERAS_IO:

    LOG.info(f"Backend: {keras.backend.backend()}")
    assert keras.backend.backend() == "tensorflow", \
        ("Keras should use the tensorflow backend, "
         f"not {keras.backend.backend()}")

    # some sanity checks
    # (C1) keras should not be (much) newer than tensorflow.
    #      Some ideas a given on
    #        https://docs.floydhub.com/guides/environments/
    #      In the following configurations I experienced problems:
    #      tensorflow 1.3.0, keras 2.2.4
    #      - keras uses an 'axis' argument when calling the
    #        tf.nn.softmax() function which is not supported:
    #        "softmax() got an unexpected keyword argument 'axis'"
    from packaging import version
    keras_version = version.parse(keras.__version__)
    if keras_version >= version.parse("2.2.4"):
        tf_min_version = "1.11.0"
    elif keras_version >= version.parse("2.2.0"):
        tf_min_version = "1.10.0"
    elif keras_version >= version.parse("2.2.0"):
        tf_min_version = "1.9.0"
    elif keras_version >= version.parse("2.1.6"):
        tf_min_version = "1.5.0"
    elif keras_version >= version.parse("2.0.8"):
        tf_min_version = "1.4.0"
    elif keras_version >= version.parse("2.0.6"):
        tf_min_version = "1.0.0"
    else:
        raise ImportError("Your keras is too old."
                          f"We require at least keras 2.0.6, "
                          f"but you have keras {keras.__version__}.")

    if version.parse(tf.__version__) < version.parse(tf_min_version):
        raise ImportError("Your tensorflow is too old for your keras."
                          f"keras {keras.__version__} requires "
                          f"at least tensorflow {tf_min_version}, "
                          f"but you have tensorflow {tf.__version__}.")


if False:  # FIXME[concept]: we nee some genral concept what to do here ...
    # image_dim_ordering:
    #   'tf' - "tensorflow":
    #   'th' - "theano":
    keras.backend.set_image_dim_ordering('tf')
    LOG.info(f"image_dim_ordering: {keras.backend.image_dim_ordering()}")
    LOG.info(f"image_data_format: {keras.backend.image_data_format()}")

    if config.use_cpu:
        # unless we do this, TF still checks and finds gpus (not
        # sure if it actually uses them)
        #
        # UPDATE: TF now still loads CUDA, there seems to be no
        # way around this
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

        # FIXME[todo]: the following setting causes a TensorFlow
        # error: failed call to cuInit: CUDA_ERROR_NO_DEVICE
        # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        LOG.info("Running in CPU-only mode.")
        from multiprocessing import cpu_count
        num_cpus = cpu_count()
        config = tf.ConfigProto(intra_op_parallelism_threads=num_cpus,
                                inter_op_parallelism_threads=num_cpus,
                                allow_soft_placement=True,
                                device_count={'CPU': num_cpus, 'GPU': 0})
        session = tf.Session(config=config)
        keras.backend.set_session(session)


if False:
    ...
    # Experimental code, not tested yet!
    
    # FIXME[todo]: configure the keras download directory
    #
    # According to the documentation (https://keras.io/getting-started/faq/)
    # the default download location for all Keras files is $HOME/.keras/
    #
    # In fact this seems be hardcoded (or at least was, check
    # keras/utils/data_utils.py)
    #
    # Setting the KERAS_HOME environment variable should allow to change
    # the location the keras directory from $HOME/.keras/ to something else.
    # However, it seems not possible to change individual subdirectories
    # like `datasets` or `models`.
    #
    # URLs for model weights can be found in the source code for the respective
    # models, e.g., keras/applications/resnet50.py shows
    # WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/' \
    #    'releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    # WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/'\
    #    'deep-learning-models/releases/download/v0.2/' \
    #    'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # These can be downloaded to any location and then loaded from there (how?)


def available() -> bool:
    """Check if the underlying framework is available (installed
    on this machine).
    """
    return HAVE_TENSORFLOW_KERAS or HAVE_KERAS_IO


def info() -> None:
    print(f"HAVE_TENSORFLOW_KERAS: {HAVE_TENSORFLOW_KERAS}")
    print(f"HAVE_KERAS_IO: {HAVE_KERAS_IO}")
    print(f"Keras version: {keras.__version__}")


    
Datasource.register_instance('mnist-train', __name__ + '.datasource.keras',
                             'KerasDatasource', name='mnist',
                             section='train')
Datasource.register_instance('mnist-test', __name__ + '.datasource.keras',
                             'KerasDatasource', name='mnist',
                             section='test')
Datasource.register_instance('cifar10-train', __name__ + '.datasource.keras',
                             'KerasDatasource', name='cifar10',
                             section='train')
