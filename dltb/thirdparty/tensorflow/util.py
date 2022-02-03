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
