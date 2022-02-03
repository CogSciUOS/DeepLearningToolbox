"""Configuration of Tensorflow 2.x.

This package contains code to be run after the tensorflow 2.x package
has been loaded.  This code should setup tensorflow based on the Deep
Learning Toolbox configuration.  It should only do TF2 specific parts
of the setup.  General setup (TF1 + TF2) should be done in the
`_postimport` module.

There should be no need to explicitly import this package. It should
be loaded automatically once after a tensorflow (version 2.x) package
has been imported.

"""

# thirdparty imports
import tensorflow as tf  # assumed to be tensorflow 2.x

# local imports
from . import LOG


#
# CPU/GPU info
#


# In TensorFlow 2.x, ConfigProto is deprecated, but a function called
# set_memory_growth is provided by tensorflow.config.experimental.
physical_devices = tf.config.experimental.\
    list_physical_devices('GPU')  # pylint: disable=no-member
for gpu in physical_devices:
    # (at least on beam) I get the following RuntimeError:
    #   "Physical devices cannot be modified after being initialized"
    # tf.config.experimental.\
    #     set_memory_growth(gpu, True)  # pylint: disable=no-member
    config = tf.config.experimental.\
        VirtualDeviceConfiguration(memory_limit=1024)  # 1024/2048/3072
    LOG.info("v2: setting config for GPU %s: config=%s", gpu, config)
    tf.config.experimental.\
        set_virtual_device_configuration(gpu, [config])
