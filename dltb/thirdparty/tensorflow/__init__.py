"""Provide tensorflow functionality.

The module provides the different TensorFlow version (if available) as:
* `tensorflow_v1`
* `tensorflow_v2`
"""

import os
import logging


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
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


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

# TF_FORCE_GPU_ALLOW_GROWTH:
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


import tensorflow  # pylint: disable=wrong-import-position

# logging
LOG = logging.getLogger(__name__)
TF_LOGGER = logging.getLogger('tensorflow')

if hasattr(tensorflow, 'compat'):
    v1 = getattr(tensorflow.compat, 'v1', None)
    v2 = getattr(tensorflow.compat, 'v2', None)
else:
    v1 = tensorflow
    v2 = None  # pylint: disable=invalid-name

tf = v1

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
if v1 is not None:  # TensorFlow 1.x
    # FIXME[todo]: check if GPU is available

    # TensorFlow 1.x offers the ConfigProto method, that allows to control
    # several aspects of the computing devices.

    config = v1.ConfigProto()
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
    # config = tensorflow.ConfigProto(gpu_options=gpu_options,
    #                                 allow_soft_placement = True)

    # allocator_type:
    # allow_growth: bool = False
    config.gpu_options.allow_growth = False  # True / [False]

    # per_process_gpu_memory_fraction: float = 0.0
    #     the memory fraction available to load GPU resources to create
    #     cudnn handle.
    #     Use as small fraction as could fit in your memory (you can start
    #     with 0.3 or even smaller, then increase until you get the error,
    #     that's your limit.)
    config.gpu_options.per_process_gpu_memory_fraction = 0.7  # [0.0] to 1.0

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
    v1.keras.backend.set_session(v1.Session(config=config))
    TF_LOGGER.setLevel(old_level)
    # v1.keras.backend.set_session(v1.Session(config=config))
    # tensorflow.compat.v1.keras.backend.set_session(v1.Session(config=config))
    LOG.info("v1: setting keras config=%s", config)


# In TensorFlow 2.x, ConfigProto is deprecated, but a function called
# set_memory_growth is provided by tensorflow.config.experimental.
if v2 is not None:  # TensorFlow 2.x
    physical_devices = v2.config.experimental.\
        list_physical_devices('GPU')  # pylint: disable=no-member
    for gpu in physical_devices:
        v2.config.experimental.\
            set_memory_growth(gpu, True)  # pylint: disable=no-member
        config = v2.config.experimental.\
            VirtualDeviceConfiguration(memory_limit=1024)  # 1024/2048/3072
        LOG.info("v2: setting config for GPU %s: config=%s", gpu, config)
        v2.config.experimental.\
            set_virtual_device_configuration(gpu, [config])


# clear the namespace
del os, tensorflow


class Utils:
    """Collection of utility functions provided by TensorFlow.
    """

    @staticmethod
    def clip_0_1(tensor: tf.Tensor):
        """Clip values of Tensor to be in the range [0,1].
        """
        return tf.clip_by_value(tensor, clip_value_min=0.0, clip_value_max=1.0)

    @staticmethod
    def download(url: str, file: str) -> None:
        """Download a video.
        """
        # FIXME[todo]: read the docs: what is the return value?
        _path = tf.keras.utils.get_file(file, url)


class CUDA:
    """Interface to access CUDA functionality.
    """

    @staticmethod
    def is_avalable() -> bool:
        """Check if CUDA support is available.
        """
        return True  # FIXME[todo]

    @staticmethod
    def number_of_gpus() -> int:
        """Determine how many GPUs are available.
        """
        return 1  # FIXME[todo]


def output_tensorflow_info():

    print(f"TensorFlow version {tf.__version__}")

    session = tf.get_default_session()
    print(f"  - default session: {session is not None}")

    tf_config = tf.ConfigProto()
    # print(dir(tf_config))
    print(f"  - allow_soft_placement: {tf_config.allow_soft_placement}")
    print(f"  - device_count: {tf_config.device_count}")
    print(f"  - experimental: {tf_config.experimental}")
    print(f"  - gpu_options: {tf_config.gpu_options}")
    # print(dir(tf_config.gpu_options))
    gpu_options = tf_config.gpu_options
    print(f"    -> allocator_type: {gpu_options.allocator_type}")
    print("    -> deferred_deletion_bytes: "
          f"{gpu_options.deferred_deletion_bytes}")
    print(f"    -> experimental: {gpu_options.experimental}")
    print(f"    -> force_gpu_compatible: {gpu_options.force_gpu_compatible}")

    # In some cases it is desirable for the process to only allocate a
    # subset of the available memory, or to only grow the memory usage
    # as it is needed by the process. TensorFlow provides two
    # configuration options on the session to control this. The first
    # is the allow_growth option, which attempts to allocate only as
    # much GPU memory based on runtime allocations, it starts out
    # allocating very little memory, and as sessions get run and more
    # GPU memory is needed, we extend the GPU memory region needed by
    # the TensorFlow process.
    
    # 1) Allow growth: (more flexible)
    #
    #    config.gpu_options.allow_growth = True
    #
    print(f"    -> allow_growth: {tf_config.gpu_options.allow_growth}")

    # 2) Allocate fixed memory:
    #
    # The second method is per_process_gpu_memory_fraction option,
    # which determines the fraction of the overall amount of memory
    # that each visible GPU should be allocated. Assume that you have
    # 12GB of GPU memory a value of
    # per_process_gpu_memory_fraction=0.333 will allocate ~4GB
    #
    #    config.gpu_options.allow_growth = False
    #    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    #
    # The per_process_gpu_memory_fraction acts as a hard upper bound
    # on the amount of GPU memory that will be used by the process on
    # each GPU on the same machine. Currently, this fraction is
    # applied uniformly to all of the GPUs on the same machine; there
    # is no way to set this on a per-GPU basis.
    #
    # Note: The actual GPU memory allocated by a process can be a bit
    # above what is specified by per_process_gpu_memory_fraction.
    # I believe this is due to cudnn and cublas context initialization.
    # That is only relevant if you are executing kernels that use those
    # libs though.
    #
    # Note 2: No release of memory needed, it can even worsen memory
    # fragmentation when done
    print("    -> per_process_gpu_memory_fraction: "
          f"{gpu_options.per_process_gpu_memory_fraction}")
    print("    -> polling_active_delay_usecs: "
          f"{gpu_options.polling_active_delay_usecs}")
    print("    -> polling_inactive_delay_msecs: "
          f"{gpu_options.per_process_gpu_memory_fraction}")
    print("    -> visible_device_list: "
          f"{gpu_options.visible_device_list}")


# FIXME[todo]: one may include code that runs tensorflow models in
# the following try-except block:
#
#
# from tensorflow.python.framework.errors_impl import OpError
#
# try:
#        # In Tensorflow 2.0
#        #physical_devices = tf.config.experimental.list_physical_devices('GPU')
#        #config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
#
# except OpError as error:
#        # The message can be quite long!
#        message = error.message
#        message = message.split(".", 1)[0]
#        print(f"TensorFlow error ({type(error).__name__}): {message}")
#        print(f"Error code: {error.error_code}")
#        print(f"Arguments: {error.args}")
#        # print(f"node_def: {error.node_def}")
#        # print(f"op: {error.op}")
#        print(f"cause: {error.__cause__}")
#        # The following Exception is raised:
#        #   tensorflow.python.framework.errors_impl.UnknownError
#        #   -> tensorflow.python.framework.errors_impl.OpError
#        #      -> Exception
#        #         -> BaseException
#        #
#        # On the console the following error message from TensorFlow:
#        #   E tensorflow/stream_executor/cuda/cuda_dnn.cc:329]
#        #        Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
#        #   E tensorflow/stream_executor/cuda/cuda_dnn.cc:329]
#        #        Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
#        #
#        # The error seems to be raised by
#        #  generate_images
#        #   -> github/stylegan/dnnlib/tflib/network.py,  line 418,
#        #      out_gpu = net_gpu.get_output_for(*in_gpu, return_as_list=True,
#        #                                       **dynamic_kwargs)
#        #   -> github/stylegan/dnnlib/tflib/network.py, line 222
#        #      out_expr = self._build_func(*final_inputs, **build_kwargs)
#        #   -> ...
#        #   -> tensorflow/python/framework/ops.py", line 3327,
#        #      in _create_op_internal
#        #
#        # (0) Unknown: Failed to get convolution algorithm.
#        #     This is probably because cuDNN failed to initialize,
#        #     so try looking to see if a warning log message was printed above.
#        #      [[node Gs/_Run/Gs/G_synthesis/4x4/Conv/Conv2D
#        #        (defined at <string>:159) ]]
#        #      [[Gs/_Run/saturate_cast/_1421]]
#        # tb = sys.exc_info()[2]
#        # raise RuntimeError("TensorFlow error").with_traceback(tb)
