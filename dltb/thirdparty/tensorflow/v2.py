"""Import TensorFlow ensuring that the TensorFlow 2.x API is active.

This module should only be imported if the TensorFlow installation is
known to be TF2.  In case that the TensorFlow installation is TF1,
importing this module will raise an `ImportError`.


Intended use:

```
from dltb.thirdparty.tensorflow.v1 import tf
```

Alternative use:

```
import dltb.thirdparty.tensorflow.v1
import tensorflow as tf
```

"""

# local imports
from . import LOG, get_tensorflow_version, set_tensorflow_version

# the following should be run before Tensorflow is imported
tensorflow_version = get_tensorflow_version()
if tensorflow_version is None:
    LOG.info("v2: Setting DLTB tensorflow_version to 'v2'")
    set_tensorflow_version('v2')
elif tensorflow_version != 'v2':
    raise ImportError("Trying to import TensorFlow v2 API, while "
                      f"tensorflow_version is set to {tensorflow_version}")


# standard imports
import importlib

# thirdparty imports
from packaging import version
import tensorflow as tf
from tensorflow.version import VERSION


if version.parse(VERSION) < version.parse("2.0.0"):
    # we have version 1.x -> we can not proceed here!
    LOG.error("v2: Trying to use TensorFlow 2.x API with a "
              "TensorFlow %s API", VERSION)
    raise ImportError("Failed to import TensorFlow 2.x API as installed"
                      f"version is only tensorflow {VERSION}")

# Adapt tensorflow using TF2 specific code
importlib.import_module('._v2', __package__)

#
# Convenience functions
#

class Utils:
    """TensorFlow utilities.
    """

    def batch_norm(inputs, is_training, name: str):
        """Create a batch normalization layer.
        """
        return tf.keras.layers.BatchNormalization(inputs,
                                                  decay=0.9,
                                                  is_training=is_training,
                                                  reuse=tf.AUTO_REUSE,
                                                  scope=name)


    # Initializers
    # ------------
    #
    # In TensorFlow 2.x, initializers can be found in tf.initializers. They
    # can be used as follows:
    #
    #   initializer = tf.initializers.GlorotUniform()
    #   var = tf.Variable(init(shape=shape))
    #
    # To write code that works with TensorFlow 1.x and TensorFlow 2.x, use
    # code like the following:
    #
    #    from dltb.thirdparty.tensorflow import GlorotUniform
    #    initializer = GlorotUniform
    #    var = tf.Variable(initializer(shape))

    GlorotUniform = tf.initializers.GlorotUniform
    # Glorot uniform and Xavier uniform are two different names of the
    # same initialization type.
    XavierUniform = GlorotUniform
