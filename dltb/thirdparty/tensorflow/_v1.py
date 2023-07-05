"""Configuration of Tensorflow 1.x.

This package contains code to be run after the tensorflow 1.x package
has been loaded.  This code should setup tensorflow based on the Deep
Learning Toolbox configuration.  It should only do TF1 specific parts
of the setup.  General setup (TF1 + TF2) should be done in the
`_postimport` module.

There should be no need to explicitly import this package. It should
be loaded automatically once after a tensorflow (version 1.x) package
has been imported.
"""

# standard imports
import logging

# thirdparty imports
import tensorflow as tf  # assumed to be tensorflow 1.x

# toolbox imports
from dltb.config import config
from . import LOG, TF_LOGGER

#
# CPU/GPU info
#
if not config.tensorflow_use_gpu:
    tf.config.set_visible_devices([], 'GPU')

# FIXME[todo]: make config observable
# def config_changed(config, what: str) -> None:
#     if what == 'use_gpu' or what == 'tensorflow_use_gpu':
#        tensorflow_use_gpu(config.tensorflow_use_gpu)
#     if what == 'tensorflow_version':
#        tensorflow_set_version(config.tensorflow_version)


# TensorFlow has a quite aggressive default GPU memory allocation policy,
# grabbing a large amount of GPU memory upon first initialization.
# This may lead to CUDA_ERROR_OUT_OF_MEMORY errors like the following:
#
#   | I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0
#   | I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y
#   | I tensorflow/core/common_runtime/gpu/gpu_device.cc:975]
#   |    Creating TensorFlow device (/gpu:0) ->
#   |    (device: 0, name: GeForce GT 750M, pci bus id: 0000:01:00.0)
#   | E tensorflow/stream_executor/cuda/cuda_driver.cc:1002]
#   |    failed to allocate 67.48M (70754304 bytes) from device:
#   |    CUDA_ERROR_OUT_OF_MEMORY
#   | Training...
#   | E tensorflow/stream_executor/cuda/cuda_dnn.cc:397]
#   |    could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
#   | E tensorflow/stream_executor/cuda/cuda_dnn.cc:364]
#   |    could not destroy cudnn handle: CUDNN_STATUS_BAD_PARAM
#   | F tensorflow/core/kernels/conv_ops.cc:605]
#   |    Check failed: stream->parent()->GetConvolveAlgorithms(&algorithms)
#
# FIXME[todo/check]:
# Some people report this problems only occur in the context of Jupyter,
# suggesting that probably, the kernel memory footprint is limited. This
# would imply, that the following code is only required in such contexts.
#

# FIXME[todo]: check if GPU is available

# TensorFlow 1.x offers the ConfigProto method, that allows to control
# several aspects of the computing devices.

tf_config = tf.ConfigProto()
# pylint: disable=no-member
#
# allow_soft_placement: bool = False
# isolate_session_state: bool = False
# log_device_placement: bool = False
# share_cluster_devices_in_session: bool = False
# inter_op_parallelism_threads: int = 0
# intra_op_parallelism_threads: int = 0
# operation_timeout_in_ms: int = 0
# placement_period: int = 0
# use_per_session_threads: bool = False

# cluster_def: ClusterDef
# device_count: ScalarMapContainer
# device_filters: RepeatedScalarContainer
# experimental: Experimental
# graph_options: GraphOptions
# rpc_options: RPCOptions
# session_inter_op_thread_pool: RepeatedCompositeContainer

# gpu_options: tensorflow.GPUOptions
# ----------------------------------
#
# gpu_options = tensorflow.GPUOptions(per_process_gpu_memory_fraction=0.7)
# tf_config = tensorflow.ConfigProto(gpu_options=gpu_options,
#                                 allow_soft_placement = True)

# allocator_type:
# allow_growth: bool = False
tf_config.gpu_options.allow_growth = False  # True / [False]

# per_process_gpu_memory_fraction: float = 0.0
#     the memory fraction available to load GPU resources to create
#     cudnn handle.
#     Use as small fraction as could fit in your memory (you can start
#     with 0.3 or even smaller, then increase until you get the error,
#     that's your limit.)
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.7  # [0.0] to 1.0

# force_gpu_compatible: bool = False
# deferred_deletion_bytes: int = 0
# polling_active_delay_usecs: int = 0
# polling_inactive_delay_msecs: int = 0
# visible_device_list: str
# experimental: Experimental

# Observations with StyleGAN:
#  allow_growth    / per_process_gpu_memory_fraction
#  False / True    / 0.0 (Default)                     Error 1
#  False (Default) / 0.1                               Error 2/Warning 2
#  False (Default) / 0.2                               OK/Warning 2
#  False (Default) / 0.3                               OK/Warning 2
#  False (Default) / 0.4                               OK/Warning 2
#  False / True    / 0.5                               OK/Warning 2
#  False (Default) / 0.6                               OK/Warning 2
#  False (Default) / 0.7                               OK
#  False (Default) / 0.8                               Error 1
#  False (Default) / 0.9                               Error 1
#  False (Default) / 1.0                               Error 1
# Error 1:
#   Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
# Warning 2:
#   Allocator (GPU_0_bfc) ran out of memory trying to allocate
#      292,30MiB with freed_by_count=0.
#      The caller indicates that this is not a failure, but may mean that
#      there could be performance gains if more memory were available.
# Error 2:
#   Resource exhausted: OOM when allocating tensor with
#      shape[1,16,1024,1024] and type float on
#      /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc

# FIXME[bug]: tensorflow emits an incorrect warning
# (on the tensorflow logger):
#    The name tf.keras.backend.set_session is deprecated.
#    Please use tf.compat.v1.keras.backend.set_session instead.
# We are using tf.compat.v1.keras.backend.set_session here!
# This also happens when importing dltb.thirdparty.tensorflow
# in IPython. However, it does not happen if executing the code
# from this file in the IPython interpreter!
# https://github.com/tensorflow/tensorflow/issues/33182
old_level = TF_LOGGER.level
TF_LOGGER.setLevel(logging.ERROR)
tf.keras.backend.set_session(tf.Session(config=tf_config))
TF_LOGGER.setLevel(old_level)
# tf.keras.backend.set_session(tf.Session(config=tf_config))
# tensorflow.compat.v1.keras.backend.set_session(tf.Session(config=tf_config))
LOG.info("v1: setting keras config=%s", tf_config)
