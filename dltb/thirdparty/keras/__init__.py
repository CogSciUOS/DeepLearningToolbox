"""Providing keras functionality.

Two different keras implementations are available:
* keras.io
* tensorflow.keras

Also keras my use different backends
* `theano`
* `tensorflow`
"""

# standard imports
import os
import sys
import importlib
import logging

# toolbox imports
from ...datasource import Datasource

# logging
LOG = logging.getLogger(__name__)

HAVE_KERAS_IO = \
    importlib.util.find_spec("keras") is not None
HAVE_TENSORFLOW_KERAS = \
    importlib.util.find_spec("tensorflow.keras") is not None

if HAVE_TENSORFLOW_KERAS:
    from . import tensorflow
    # import tensorflow.keras as keras
    import tensorflow.compat.v1.keras as keras
    sys.modules['keras'] = keras
    #from tensorflow.compat.v1.keras import backend
    #sys.modules['keras.backend'] = backend
elif HAVE_KERAS_IO:
    # The only way to configure the keras backend appears to be
    # via environment variable. We thus inject one for this
    # process. Keras must be loaded after this is done
    #os.environ['KERAS_BACKEND'] = 'theano'
    os.environ['KERAS_BACKEND'] = 'tensorflow'

    # Importing keras unconditionally outputs a message "Using [...] backend."
    # to sys.stderr (in keras/__init__.py). There seems to be
    # no sane way to avoid this.
    import keras
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

if False: # FIXME[concept]: we nee some genral concept what to do here ...
    if cpu:
        # unless we do this, TF still checks and finds gpus (not
        # sure if it actually uses them)
        #
        # UPDATE: TF now still loads CUDA, there seems to be no
        # way around this
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

        # FIXME[todo]: the following setting causes a TensorFlow
        # error: failed call to cuInit: CUDA_ERROR_NO_DEVICE
        #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        LOG.info("Running in CPU-only mode.")
        from multiprocessing import cpu_count
        num_cpus = cpu_count()
        config = tf.ConfigProto(intra_op_parallelism_threads=num_cpus,
                                inter_op_parallelism_threads=num_cpus,
                                allow_soft_placement=True,
                                device_count={'CPU': num_cpus, 'GPU': 0})
        session = tf.Session(config=config)

    # image_dim_ordering:
    #   'tf' - "tensorflow":
    #   'th' - "theano":
    keras.backend.set_image_dim_ordering('tf')
    LOG.info(f"image_dim_ordering: {keras.backend.image_dim_ordering()}")
    LOG.info(f"image_data_format: {keras.backend.image_data_format()}")

    if cpu:
        keras.backend.set_session(session)


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
