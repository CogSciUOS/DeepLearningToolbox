"""Tensorflow post-import module.  This module should be imported
(directly) after the tensorflow library was imported.

The goal of this module is to adapt TensorFlow to the desired behavior:
* Switch on the TensorFlow 1.x compatibility mode if desired.
* Introduce compatibility functions, like layers and initializers
"""

# standard imports
import sys
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

# toolbox imports
from dltb.config import config
from . import LOG, TF_LOGGER, get_tensorflow_version


dltb_tensorflow = sys.modules['dltb.thirdparty.tensorflow']
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
    utils = importlib.import_module('.' + tensorflow_version, __package__)
    # dltb_tensorflow.Utils = utils.Utils


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


#
# Try to silence TensorFlow a bit
#

deprecation._PER_MODULE_WARNING_LIMIT = 0
deprecation._PRINT_DEPRECATION_WARNINGS = False

# reduce the number of warnings (does not only affect TensorFlow)
simplefilter(action='ignore', category=FutureWarning)




# Compatibility problems
# -----------------------
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
