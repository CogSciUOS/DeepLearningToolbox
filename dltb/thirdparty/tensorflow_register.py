"""Register tensorflow implementations.

This module should only be imported if the `tensorflow` package is installed.
"""
# standard imports
import os
import sys
import logging

# toolbox imports
from dltb.config import config
from dltb.base.implementation import Implementable
from dltb.network import Network
from dltb.util.importer import importable
from dltb.util.importer import add_preimport_hook, add_postimport_depency
from . import THIRDPARTY

# logging
TENSORFLOW = THIRDPARTY + '.tensorflow'
LOG = logging.getLogger(TENSORFLOW)

if not importable('tensorflow'):
    raise ImportError("Registering TensorFlow definitions failed "
                      "as module 'tensorflow' is not importable.")

#
# tensorflow_devices
#

config.add_property('tensorflow_use_gpu', default=lambda c: c.use_gpu,
                    description="Use GPU(s) for Tensorflow if available.")


def patch_tensorflow_import(fullname, path, target=None):
    # pylint: disable=unused-argument
    """TensorFlow preimport.  This code is to be executed before
    the `tensorflow` library is imported.

    This will adapt GPU settings and output behaviour based on
    `config` settings.
    """

    LOG.info("Preparing TensorFlow import")

    #
    # Set desired TensorFlow devices.
    #
    if 'tensorflow' in sys.modules:
        LOG.warning("Module 'tensorflow' is already loaded. "
                    "Setting devices may have no effect.")

    if config.tensorflow_use_gpu:
        LOG.info("Use GPUs for TensorFlow if available"
                 "(not setting CUDA_VISIBLE_DEVICES='')")
        # Using CUDA_VISIBLE_DEVICES environment variable indicates which
        # CUDA devices should be accessible to TensorFlow
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # first two GPUs
        # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # TF_FORCE_GPU_ALLOW_GROWTH:
        # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        # CUDA_DEVICE_ORDER:
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        # TF_FORCE_GPU_ALLOW_GROWTH:
        # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
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
    else:
        # use CPU even if GPU is available
        LOG.info("Disabling GPUs for TensorFlow "
                 "(set CUDA_VISIBLE_DEVICES='')")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # do not use GPU


        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # do not use GPU

    #
    # Adapt TensorFlow verbosity
    #

    # TF_CPP_MIN_LOG_LEVEL: Control the amount of TensorFlow log
    # message displayed on the console.
    #  0 = INFO
    #  1 = WARNING
    #  2 = ERROR
    #  3 = FATAL
    #  4 = NUM_SEVERITIES
    # Defaults to 0, so all logs are shown.
    LOG.info("Reduce amount of TensorFlow messages "
             "(set TF_CPP_MIN_LOG_LEVEL=1, i.e. only WARNING or more severe)")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    # TF_CPP_MIN_VLOG_LEVEL brings in extra debugging information
    # and actually works the other way round: its default value is
    # 0 and as it increases, more debugging messages are logged
    # in.
    # Remark: VLOG messages are actually always logged at the INFO
    # log level. It means that in any case, you need a
    # TF_CPP_MIN_LOG_LEVEL of 0 to see any VLOG message.
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


add_preimport_hook('tensorflow', patch_tensorflow_import)
add_postimport_depency('tensorflow', ('.tensorflow', THIRDPARTY))


Implementable.register_module_alias(TENSORFLOW, 'tensorflow')
Implementable.register_module_alias(TENSORFLOW, 'tf')
Implementable.register_implementation('dltb.tool.autoencoder.Autoencoder',
                                      TENSORFLOW + '.ae.Autoencoder')

Network.register_instance('alexnet-tf', TENSORFLOW + '.network', 'Alexnet')
# FIXME[old]:                     extend=Classifier, scheme='ImageNet')
