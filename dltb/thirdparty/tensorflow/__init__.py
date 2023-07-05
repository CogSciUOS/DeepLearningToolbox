"""Provide tensorflow functionality.

The module provides the different TensorFlow version (if available) as:
* `tensorflow_v1`
* `tensorflow_v2`

This module can either be imported directly, or it will be automatically
imported as a `tensorflow` postimport hook (assuming that
:py:mod:`dltb.thirdparty.tensorflow_register` has been imported).

The goal of this module is to adapt TensorFlow to the desired behavior:
* Switch on the TensorFlow 1.x compatibility mode if desired.
* Introduce compatibility functions, like layers and initializers


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
from warnings import simplefilter

# thirdparty imports
from packaging import version
import tensorflow
from tensorflow.version import VERSION
# For Tensorflow 1.14+ the following should work
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation

from tensorflow.python.client import device_lib

# toolbox imports
from dltb.config import config
from dltb.base.package import Package
from dltb.util.importer import import_module


# logging
LOG = logging.getLogger(__name__)
TF_LOGGER = logging.getLogger('tensorflow')

#handler = logging.StreamHandler(sys.stderr)
#handler.setLevel(logging.DEBUG)
#LOG.setLevel(logging.DEBUG)
#LOG.addHandler(handler)

LOG.info("init: Importing Deep Learning Toolbox 'tensorflow' module.")

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


tensorflow_version = get_tensorflow_version()
LOG.info("postimport: have imported TensorFlow %s (desired: %s)",
         VERSION, tensorflow_version)


#
# Import version specific file
#
if tensorflow_version is None:
    tensorflow_version = \
        'v1' if version.parse(VERSION) < version.parse("2.0.0") else 'v2'

if tensorflow_version is not None:
    LOG.info("postimport: importing %s.%s", __package__, tensorflow_version)
    utils = import_module('.' + tensorflow_version, __package__)
    # Utils = utils.Utils


#
# setting variables v1, v2, tf_contrib
#

tf_contrib = getattr(tensorflow, 'contrib', None)

if hasattr(tensorflow, 'compat'):
    v1 = getattr(tensorflow.compat, 'v1', None)
    v2 = getattr(tensorflow.compat, 'v2', None)
    if tf_contrib is not None and not hasattr(v1, 'contrib'):
        v1.contrib = tf_contrib
else:
    v1 = tensorflow
    v2 = None  # pylint: disable=invalid-name


# Compatibility problems
# ======================
#
# Numpy
# -----
#
# Some Tensorflow versions have problems with newer versions of numpy:
#  \tensorflow\python\framework\dtypes.py:516: FutureWarning:
#      Passing (type, 1) or '1type' as a synonym of type is deprecated;
#      in a future version of numpy, it will be understood as
#      (type, (1,)) / '(1,)type'.
# The options are to either upgrade TensorFlow or downgrade Numpy:
# - TensorFlow 1.10.0 and Numpy ?
#    -> upgrade TensorFlow to 1.10.1
#   => tensorflow 1.10.0 has requirement numpy<=1.14.5,>=1.13.3
# - TensorFlow 1.14.0 and Numpy 1.19.2
#    -> downgrade Numpy to 1.16.4


#
# tensorflow verbosity: try to silence TensorFlow a bit
#

deprecation._PER_MODULE_WARNING_LIMIT = 0
deprecation._PRINT_DEPRECATION_WARNINGS = False

# reduce the number of warnings (does not only affect TensorFlow)
simplefilter(action='ignore', category=FutureWarning)


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
# TensorBoard
#

# older TensorBoard versions (e.g. tensorboard 2.4.0) cause numpy
# deprecation warnings when used with numpy >= 1.20 (as tensorboard uses
# the old np.object/np.bool instead of the builtin object/bool types).



class TensorflowPackage(Package):
    """An extended :py:class:`Package` for providing specific TensorFlow
    information.

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(key='tensorflow', **kwargs)

    def initialize_info(self) -> None:
        """Register torch specific information.
        """
        super().initialize_info()

        #  There is an undocumented method called
        #  device_lib.list_local_devices() that enables you to list
        #  the devices available in the local process (As an
        #  undocumented method, this is subject to backwards
        #  incompatible changes.)
        for idx, dev in enumerate(device_lib.list_local_devices()):
            self.add_info(f'device{idx}',
                          f"Device: {dev.name} ({dev.device_type})",
                          title=f"Tensorflow device {idx}")

        # Note that (at least up to TensorFlow 1.4), calling
        # device_lib.list_local_devices() will run some initialization
        # code that, by default, will allocate all of the GPU memory
        # on all of the devices (GitHub issue). To avoid this, first
        # create a session with an explicitly small
        # per_process_gpu_fraction, or allow_growth=True, to prevent
        # all of the memory being allocated. See this question for
        # more details

        # https://stackoverflow.com/questions/38009682/how-to-tell-if-tensorflow-is-using-gpu-acceleration-from-inside-python-shell

        

TensorflowPackage()
