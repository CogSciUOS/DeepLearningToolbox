"""Provide tensorflow functionality.

The module provides the different TensorFlow version (if available) as:
* `tensorflow_v1`
* `tensorflow_v2`

Intended Usage
--------------

This module (`dltb.thirdparty.tensorflow`) should be imported before
the actual `tensorflow` module is imported.  It sets up hooks and
provides functions to configure TensorFlow.  This module will not
import TensorFlow


The function `set_tensorflow_version()` can be used to choose if
TensorFlow should be used with TensorFlow 1.x API (`'v1'`) or with the
TensorFlow 2.x API (`'v2'') API.  If no explicit version is set, the
version of the installed TensorFlow package will be used.


```
from dltb.thirdparty.tensorflow import set_tensorflow_version
set_tensorflow_version('v1')
import tensorflow as tf
```

A (more compact) alternative to import a specific version of
tensorflow is to import the submodule v1 or v2 respectively.

```
from dltb.thirdparty.tensorflow.v2 import tf
```


Implementation
--------------

There are several files involved in realizing the import mechanism:

* `__init__.py`: setup the pre- and post-import hooks
   This file (that is the module `dltb.thirdparty.tensorflow`)
   should be imported early enough (that is before importing `tensorflow`)
   to work.

* `_preimport.py`: code to be executed before tensorflow is imported.
   This module should not be imported explicitly but will used by
   the internal import hooks.

* `_postimport.py`: code to be executed after tensorflow is imported.
   This module should not be imported explicitly but will used by
   the internal import hooks.

* `v1.py`: the public interface to import tensorflow in version 1 mode.

* `v2.py`: the public interface to import tensorflow in version 1 mode.


FIXME[todo]: the alternative could be to realize a pre-import hook,
that imports this file before importing tensorflow.

"""

# standard imports
from typing import Optional
import os
import sys
import logging
import importlib

# toolbox imports
from dltb.base.implementation import Implementable

# logging
LOG = logging.getLogger(__name__)
TF_LOGGER = logging.getLogger('tensorflow')

#handler = logging.StreamHandler(sys.stderr)
#handler.setLevel(logging.DEBUG)
#LOG.setLevel(logging.DEBUG)
#LOG.addHandler(handler)

LOG.info("init: Importing Deep Learning Toolbox 'tensorflow' module.")

if 'tensorflow' in sys.modules:
    LOG.warning("Tensorflow has been imported before "
                "`dltb.thirdparty.tensorflow'. "
                "Import hooks will have no effect.")

#
# tensorflow_version
#

_TENSORFLOW_VERSION = None  # 'v1' or 'v2'

def get_tensorflow_version() -> Optional[str]:
    """Desired version of the TensorFlow API to use.
    """
    return _TENSORFLOW_VERSION


def set_tensorflow_version(version: Optional[str]) -> None:
    """Desired version of the TensorFlow API to use.
    """
    global _TENSORFLOW_VERSION
    if 'tensorflow' in sys.modules:
        LOG.warning("Module 'tensorflow' is already loaded. "
                    "Setting the API version may have no effect.")

    _TENSORFLOW_VERSION = version


def tensorflow_version_available(version: str) -> bool:
    """Check if the given tensorflow version is available.
    This means that tensorflow is installed and no other
    version has been activated yet.
    """
    if _TENSORFLOW_VERSION is not None:
        # tensorflow was already imported
        return _TENSORFLOW_VERSION == version

    tensorflow_spec = importlib.util.find_spec('tensorflow')
    if tensorflow_spec is None:
        # tensorflow is not installed
        return False

    if version == 'v1':
        return True

    return os.path.isdir(tensorflow_spec.submodule_search_locations[0] +
                         '/_api/' + version)


#
# tensorflow_verbosity
#

def set_tensorflow_verbosity() -> None:
    """Set desired verbosity of the TensorFlow API.
    """
    if 'tensorflow' in sys.modules:
        LOG.warning("Module 'tensorflow' is already loaded. "
                    "Setting verbosity now may have no effect.")


#if tf_contrib is not None:
#    tf_contrib._warning = None

# Tensorflow 1.x:
# tf.logging.set_verbosity(tf.logging.WARNING)
# Tensorflow 2.x:
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARNING)



#
# tensorflow_devices
#

def set_tensorflow_devices(gpu: bool = True) -> None:
    """Set desired TensorFlow devices.
    """
    if 'tensorflow' in sys.modules:
        LOG.warning("Module 'tensorflow' is already loaded. "
                    "Setting devices may have no effect.")

    if not gpu:
        # Different options to select the device:
        #
        #  * Using CUDA_VISIBLE_DEVICES environment variable.
        #  * with tf.device(...):  # '/gpu:2'
        #  * config = tf.ConfigProto(device_count = {'GPU': 1})
        #    sess = tf.Session(config=config).
        #
        # TensorFlow 2.x:
        #   physical_devices = tf.config.list_physical_devices('GPU')
        #   
        #   # tf.config.set_visible_devices(devices, device_type=None)
        #   #  device_type: 'GPU' or 'CPU'
        #   tf.config.set_visible_devices(physical_devices[1:], 'GPU')
        #
        #   logical_devices = tf.config.list_logical_devices('GPU')
        #
        #
        # inspect available GPU memory:
        #  nvidia-smi --query-gpu=memory.free --format=csv
        #
        #
        # GPU/CUDA info
        # -------------
        #
        # import tensorflow as tf
        # if tf.test.gpu_device_name():
        #    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        # else:
        #    print("Please install GPU version of TF")
        # print(tf.list_devices())
        # print(tf.test.is_built_with_cuda())


        # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # first two GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # do not use GPU
        # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # TF_FORCE_GPU_ALLOW_GROWTH:
        # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

Implementable.register_module_alias(__name__, 'tensorflow')
Implementable.register_module_alias(__name__, 'tf')
Implementable.register_implementation('dltb.tool.autoencoder.Autoencoder',
                                      __name__ + '.ae.Autoencoder')
