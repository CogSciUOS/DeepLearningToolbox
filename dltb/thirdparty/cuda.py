# FIXME[todo]: up to now this is just a collection of code snippets
# that deal with CUDA related issues. We still have to distill some
# meaningful API from this ...


class Cuda:
    # We use numba to detect the version of the CUDA toolkit. If numba
    # is not available, you should set this manually as
    # cudatoolkit_version
    try:
        from numba import cuda
        print(f"CUDA is available: {cuda.is_available()}")

        toolkit_version = cuda.runtime.get_version()
    except ImportError:
        if 'cudatoolkit_version' not in globals():
            print("Set Cuda.toolkit_version=(major, minor) manually or "
                  "install numba for automatic detection.", file=sys.stderr)
    finally:
        assert len(toolkit_version) == 2

    toolkit_version_number = toolkit_version[0]*10 + toolkit_version[1]
    toolkit_version_string = '.'.join(map(str, toolkit_version))
    print(f"CUDA Toolkit version: {'.'.join(map(str, toolkit_version))}")


if False:  # nnabla related code

    # We have to know which `cudatookit` version we are using, as nnabla
    # depends on that version.

    # nnabla_ext_name = "nnabla-ext-cuda{Cuda.toolkit_version_number}"
    # !pip install nnabla-ext-cuda100
    # !pip install nnabla-ext-cuda101
    import nnabla_ext.cuda

    print(f"NNAbla CUDA Toolkit Version: {nnabla_ext.cuda.__cuda_version__}")
    print(f"NNAbla CuDNN Version: {nnabla_ext.cuda.__cudnn_version__}")
    assert nnabla_ext.cuda.__cuda_version__ == Cuda.toolkit_version_string, \
        "Invalid nnabla version"


if False:
    import tensorflow
    print(f"TensorFlow version: {tensorflow.__version__}")

    print(f"TensorFlow build with CUDA: {tensorflow.test.is_built_with_cuda()}")
    # print(f"TensorFlow GPU available: {tensorflow.test.is_gpu_available()}")  # deprecated
    print(f"TensorFlow device: {tensorflow.test.gpu_device_name()}")
    print(f"TensorFlow devices: {tensorflow.config.list_physical_devices('GPU')}")
    # from tensorflow.python.client import device_lib
    # print(f"TensorFlow devices: {device_lib.list_local_devices()}")

    # We need TensorFlow Version 1.x
    if version.parse(tensorflow.__version__) < version.parse("2.0.0"):
        tf = tensorflow
    elif sys.modules['tensorflow'].__name__ != 'tensorflow_core.compat.v1':
        import tensorflow.compat.v1 as tf
        tf.compat.v1.disable_v2_behavior()
        # the following line will make subsequent imports of tensorflow
        # use the v1 interface
        sys.modules['tensorflow'] = tf
