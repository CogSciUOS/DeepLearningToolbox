

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

def get_strategy(xla=0, fp16=0, no_cuda=0):
    """Determines the strategy under which the network is trained.
  
    From
    [1] https://github.com/huggingface/transformers/
        blob/8eb7f26d5d9ce42eb88be6f0150b22a41d76a93d/
        src/transformers/training_args_tf.py

    Returns
    -------
    strategy:
        The strategy object (of type
        `tensorflow.python.distribute.distribute_lib.StrategyBase`).
    """
    print("TensorFlow: setting up strategy")

    if xla:
        tf.config.optimizer.set_jit(True)

    gpus = tf.config.list_physical_devices("GPU")
    # Set to float16 at first
    if fp16:
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
        tf.keras.mixed_precision.experimental.set_policy(policy)

    if no_cuda:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    else:
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        except ValueError:
            tpu = None
  
        if tpu:
            # Set to bfloat16 in case of TPU
            if fp16:
                policy = tf.keras.mixed_precision.experimental.\
                    Policy("mixed_bfloat16")
                tf.keras.mixed_precision.experimental.set_policy(policy)
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)

            strategy = tf.distribute.experimental.TPUStrategy(tpu)

        elif len(gpus) == 0:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        elif len(gpus) == 1:
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        elif len(gpus) > 1:
            # If you only want to use a specific subset of GPUs
            # use `CUDA_VISIBLE_DEVICES=0`
            strategy = tf.distribute.MirroredStrategy()
        else:
            raise ValueError("Cannot find the proper strategy! Please"
                             " check your environment properties.")

    print(f"Using strategy: {strategy}")
    return strategy


# FIXME[hack]: provides some networks for testing

def create_resnet50() -> tf.keras.Model:
    """Create ResNet50
    """
    resnet_base = tf.keras.applications.ResNet50(
        input_shape=(32,32,3),
        weights='imagenet',
        pooling='avg',
        include_top=False)
    output = tf.keras.layers.Dense(10, activation="softmax")\
             (resnet_base.output)
    model = tf.keras.Model(inputs=[resnet_base.input], outputs=[output])
    return model

def create_resnet101() -> tf.keras.Model:
    """Create ResNet101
    """
    resnet_base = tf.keras.applications.ResNet101(
        input_shape=(32,32,3),
        weights='imagenet',
        pooling='avg',
        include_top=False)
    # Downloading data from 
    # https://storage.googleapis.com/tensorflow/keras-applications/resnet/
    # resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5

    output = tf.keras.layers.Dense(10, activation="softmax")\
             (resnet_base.output)
    model = tf.keras.Model(inputs=[resnet_base.input], outputs=[output])
    return model

def train_model_on_cifar10(model_creator):
    """Train the ResNet50 model on the cifar10 dataset.

    see:
    https://www.tensorflow.org/guide/keras/train_and_evaluate
    """
    strategy = get_strategy()
    with strategy.scope():
        model = model_creator()
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # scale the data

    print("Train the model")
    model.fit(x_train, y_train, epochs=10, batch_size=256)

    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=128)
    print("test loss, test acc:", results)

    return model

def evaluate_model_on_cifar10(model):

    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # scale the data

    results_train = model.evaluate(x_train, y_train, batch_size=128)
    print("train loss, test acc:", results_train)

    results_test = model.evaluate(x_test, y_test, batch_size=128)
    print("test loss, test acc:", results_test)


# Certain versions of TensorFlow do not like to pickle Keras models
# (and actually it is not recommended to do so!).  However, if you
# want to do so nevertheless (and some existing code you find online
# is actually doing that) a hotfix [1] can be applied.
#
# [1]
#  https://github.com/tensorflow/tensorflow/issues/34697#issuecomment-627193883
import pickle

from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

# Hotfix function
def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__

# Run the function
# make_keras_picklable()


def save_model_as_pickle(model, pickle_name):
    # pickle the model
    with open(pickle_name, 'wb') as pickle_file:
        pickle.dump(model, pickle_file)  


def load_pickled_model(pickle_name):
    with open(pickle_name, 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    return model


# Test 1: create, train, and save models:
#
#  from dltb.thirdparty.tensorflow import create_resnet50, create_resnet101
#  from dltb.thirdparty.tensorflow import train_model_on_cifar10
#  from dltb.thirdparty.tensorflow import save_model_as_pickle
#  from dltb.thirdparty.tensorflow import make_keras_picklable
#  make_keras_picklable()
#
#  resnet50 = train_model_on_cifar10(create_resnet50)
#  save_model_as_pickle(resnet50, 'resnet50.pkl')
#
#  resnet101 = train_model_on_cifar10(create_resnet101)
#  save_model_as_pickle(resnet101, 'resnet101.pkl')
#
# Test 2: load and evaluate the models:
#
#  from dltb.thirdparty.tensorflow import load_pickled_model  # , unpack
#
#  resnet50 = load_pickled_model('resnet50.pkl')
#  resnet101 = load_pickled_model('resnet101.pkl')
