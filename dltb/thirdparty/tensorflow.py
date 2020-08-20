"""Provide tensorflow functionality.

The module provides the different TensorFlow version (if available) as:
* `tensorflow_v1`
* `tensorflow_v2`
"""

import os

# TF_CPP_MIN_LOG_LEVEL: Control the amount of TensorFlow log
# message displayed on the console.
#  0 = INFO
#  1 = WARNING
#  2 = ERROR
#  3 = FATAL
#  4 = NUM_SEVERITIES
# Defaults to 0, so all logs are shown.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# TF_CPP_MIN_VLOG_LEVEL brings in extra debugging information
# and actually works the other way round: its default value is
# 0 and as it increases, more debugging messages are logged
# in.
# Remark: VLOG messages are actually always logged at the INFO
# log level. It means that in any case, you need a
# TF_CPP_MIN_LOG_LEVEL of 0 to see any VLOG message.
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import tensorflow

if hasattr(tensorflow, 'compat'):
    v1 = getattr(tensorflow.compat, 'v1', None)
    v2 = getattr(tensorflow.compat, 'v2', None)
else:
    v1 = tensorflow
    v2 = None

tf = v1

# clear the namespace
del os, tensorflow


class Utils:

    @staticmethod
    def clip_0_1(tensor: tf.Tensor):
        return tf.clip_by_value(tensor, clip_value_min=0.0, clip_value_max=1.0)

    @staticmethod
    def download(url: str, file: str) -> None:
        """Download a video.
        """
        # FIXME[todo]: read the docs: what is the return value?
        path = tf.keras.utils.get_file(file, url)
