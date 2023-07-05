"""Import TensorFlow ensuring that the TensorFlow 1.x API is active.


Intended use:

```
from dltb.thirdparty.tensorflow.v1 import tf
```

Alternative use:

```
import dltb.thirdparty.tensorflow.v1
import tensorflow as tf
```
"""

# local imports
# pylint: disable=wrong-import-order
from . import LOG, get_tensorflow_version, set_tensorflow_version

# the following should be run before Tensorflow is imported
# pylint: disable=wrong-import-position
tensorflow_version = get_tensorflow_version()
if True or tensorflow_version is None:  # FIXME[hack]
    LOG.info("v1: Setting DLTB tensorflow_version to 'v1'")
    set_tensorflow_version('v1')
elif tensorflow_version != 'v1':
    raise ImportError("Trying to import TensorFlow v1 API, while "
                      f"tensorflow_version is set to {tensorflow_version}")

# standard imports
from typing import Optional, Any
import sys
import importlib

# thirdparty imports
from packaging import version
import tensorflow
from tensorflow.version import VERSION  # plyint: disable=import-error

# toolbox imports
from dltb.base.prepare import Preparable
from dltb.tool.train import Trainable


__all__ = ['tensorflow', 'tf', 'tf_contrib', 'Utils']


if version.parse(VERSION) < version.parse("2.0.0"):
    # we have version 1.x -> nothing to do
    LOG.info("v1: Using TensorFlow %s with the TensorFlow %s API",
             VERSION, tensorflow_version)
    # pylint: disable=no-member
    tf_layers = tensorflow.layers
    tf_contrib = tensorflow.contrib
else:
    # we have version 2.x -> adapt to use TensorFlow 1.x API.
    LOG.info("v1: Adapting TensorFlow %s to be used with the "
             "TensorFlow v1 API", VERSION)
    importlib.import_module('._v2_as_v1', __package__)
    # pylint: disable=invalid-name
    tf_layers = tensorflow.keras.layers
    tf_contrib = None

# Adapt tensorflow using TF1 specific code
importlib.import_module('._v1', __package__)

# set the tf variable (_v2_as_v1 may have changed sys.modules['tensorflow'])
tf = sys.modules['tensorflow']


#
# Convenience functions
#

class Utils:
    """Tensorflow utilities.
    """

    # Initializers
    # ------------
    #
    # Compatibility notice: old TensorFlow 1.x code used the following
    # coding scheme:
    #
    #    initializer = tf.contrib.layers.xavier_initializer()
    #    var = tf.get_variable("var_name", shape=shape,
    #                          initializer=initializer)
    #
    # Notice that both, the use of tf.contrib and the use of tf.get_variable
    # have been deprecated.  In TensorFlow 2.0, initializers can  be found in
    # tf.initializers. The use of tf.get_variable can also be avoided by
    # constructing the variable explicitly using tf.Variable().  So to
    # make the code more portable, one could use:
    #
    #    from dltb.thirdparty.tensorflow import XavierUniform
    #    initializer = XavierUniform
    #    var = tf.Variable(initializer(shape))
    #

    # Initialization:
    # * Glorot uniform and Xavier uniform are two different names of the
    #   same initialization type.
    if tf_contrib is not None:
        XavierUniform = tf_contrib.layers.xavier_initializer
    else:
        XavierUniform = tf.compat.v2.initializers.GlorotUniform
    # Glorot uniform and Xavier uniform are two different names of the
    # same initialization type.
    GlorotUniform = XavierUniform

    # Tensorflow 1.x:
    #
    #   initializer = tf.contrib.layers.xavier_initializer()
    #   var = tf.get_variable("var_name", shape=shape, initializer=initializer)
    #
    # oder (ohne tf.get_variable))
    #
    #   initializer = tf.contrib.layers.xavier_initializer()
    #   var = tf.Variable(initializer(shape))
    #
    # initializer = tf_contrib.layers.variance_scaling_initializer(factor=1.0)
    #

    # Tensorflow 2.x:
    #
    #  initializer = tf.initializers.GlorotUniform()
    #  var = tf.Variable(init(shape=shape))
    #
    # initializer = tf.initializers.GlorotUniform()
    # initializer = tf.initializers.VarianceScaling(scale=1.0)

    #initializer = tf_contrib.layers.xavier_initializer()
    initializer = XavierUniform()

    @classmethod
    def conv(cls, inputs, filters, name: str):
        """Create a 2D convolutional layer.
        """
        net = tf.layers.conv2d(inputs=inputs,
                               filters=filters,
                               kernel_size=[3, 3],
                               strides=(1, 1),
                               padding="SAME",
                               kernel_initializer=cls.initializer,
                               name=name,
                               reuse=tf.AUTO_REUSE)
        return net

    @staticmethod
    def maxpool(inputs, name: str):
        """Create a max-pooling layer.
        """
        net = tf.nn.max_pool(value=inputs,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding="SAME", name=name)
        return net

    # Tensorflow 1.x:
    #
    #   tf.contrib.layers.batch_norm(_input, center=True, scale=scale,
    #         epsilon=batch_norm_eps, decay=batch_norm_decay,
    #         is_training=is_train, reuse=True, updates_collections=None,
    #         scope=scope, fused=False
    #
    # Tensorflow 2.x: (TF1: decay -> TF2: momentum)
    #
    #   tf.keras.layers.BatchNormalization(...)
    #
    #
    @staticmethod
    def batch_norm(inputs, is_training, name: str):
        """Create a batch normalization layer.
        """
        net = tf_contrib.layers.batch_norm(inputs,
                                           decay=0.9,
                                           is_training=is_training,
                                           reuse=tf.AUTO_REUSE,
                                           scope=name)
        return net

    @staticmethod
    def leaky(inputs):
        """Create a leaky ReLU.
        """
        return tf.nn.leaky_relu(inputs)

    @staticmethod
    def relu(inputs):
        """Create a rectified linear unit (ReLU).
        """
        return tf.nn.relu(inputs)

    @staticmethod
    def drop_out(inputs, keep_prob):
        """Create a dropout layer.
        """
        if version.parse(VERSION) < version.parse("2.0.0"):
            return tf.nn.dropout(inputs, keep_prob)
        return tf.nn.dropout(inputs, rate=1-keep_prob)

    @classmethod
    def dense(cls, inputs, units, name: str):
        """Create a dense layer.
        """

        # UserWarning: `tf.layers.dense` is deprecated and will be
        # removed in a future version. Please use
        # `tf.keras.layers.Dense` instead.
        if version.parse(VERSION) < version.parse("2.0.0"):
            net = tf.layers.dense(inputs=inputs,
                                  units=units,
                                  reuse=tf.AUTO_REUSE,
                                  name=name,
                                  kernel_initializer=cls.initializer)
        else:
            net = tf.keras.layers.\
                Dense(units=units, # reuse=tf.AUTO_REUSE,
                      name=name, kernel_initializer=cls.initializer)(inputs)
        return net

    @classmethod
    def dense_layer(cls, inputs, units, name: str, keep_prob):
        """Create a dense layer.  The layer consists of a dense connection
        followed by a ReLU activation and a dropout layer.

        """
        return cls.drop_out(cls.relu(cls.dense(inputs, units, name)),
                            keep_prob)


class TensorflowBase(Preparable, Trainable):
    """Base class for TensorFlow 1.x models.
    """

    def __init__(self, graph: Optional[tf.Graph] = None,
                 session: Optional[tf.Session] = None,
                 **kwargs) -> None:
        """Initialized the `TensorflowBase`.
        """
        super().__init__(self, **kwargs)
        self._tf_graph = graph
        self._tf_session = session

        self._tf_saver = None
        self._tf_global_step = None
        self._tf_prepared_for_training = False

    def _prepared(self) -> bool:
        """Check if the `TensorflowBase` is prepared.
        """
        return self._tf_graph is not None and super()._prepared()

    def _prepare(self) -> None:
        """Prepare the `TensorflowBase`.
        """
        super()._prepare()
        if self._tf_graph is None:
            self._tf_graph = tf.Graph()
        with self._tf_graph.as_default():
            self._tf_prepare()

            # sess.run(tf.initialize_all_variables())
            # initialize_all_variables (from tensorflow.python.ops.variables)
            # is deprecated and will be removed after 2017-03-02.
            # Instructions for updating:
            # Use `tf.global_variables_initializer` instead.
            self._tf_session = tf.Session()
            self._tf_session.run(tf.global_variables_initializer())

    def _unprepare(self) -> None:
        """Unprepare the `TensorflowBase`.
        """
        if self._tf_session is not None:
            self._tf_session.close()
            del self._tf_session
            self._tf_session = None

        if self._tf_graph is not None:
            # with self._tf_graph.as_default():
            #    tf.reset_default_graph()
            del self._tf_graph
            self._tf_graph = None
        super()._unprepare()

    def prepare_training(self, restore: bool = False) -> None:
        """Prepare the training process.
        """
        with self._tf_graph.as_default():
            # Initialize the session
            if self._tf_session is None:
                self._tf_session = tf.Session()

            # and prepare the training (define optimizers, etc.)
            if not self._tf_prepared_for_training:
                self._tf_prepare_training()

                # make sure all training variables are initialized
                self._tf_session.run(tf.global_variables_initializer())

    def restore_from_checkpoint(self) -> None:
        """Restore state from a checkpoint.
        Restored information should include training_step (epoch/batch)
        """
        if self._tf_saver is not None:
            self._tf_restore_from_checkpoint(self._tf_saver)

    def get_hyperparamter(self, name: str) -> Any:
        """Get hyperparamters for thie `Trainable`.
        The `TensorflowBase` can provide the `global_step`.
        """
        if name == 'global_step':
            if self._tf_session is None:
                raise RuntimeError("No session.")
            return self._tf_session.run(self._tf_global_step)
        return super().get_hyperparamter(name)

    def _tf_prepare(self) -> None:
        """Prepare the TensorFlow Graph for this :py:class:`TensorflowBase`.
        """
        self._tf_global_step = tf.Variable(0, trainable=False)

        # create a Saver
        self._tf_saver = tf.train.Saver(name='aa',
                                        filename="my_test_model")

    def _tf_prepare_training(self) -> None:
        """Prepare this :py:class:`TensorflowBase` for training.
        """
        self._tf_prepared_for_training = True

    def _tf_store_checkpoint(self, saver: tf.train.Saver) -> None:
        saver.save(self._tf_session, 'checkpoints/my_test_model',
                   global_step=self._tf_global_step,
                   write_meta_graph=False)
        LOG.info("Saved checkpoint to '%s'", saver.last_checkpoints)

    def _tf_restore_from_checkpoint(self, saver: tf.train.Saver) -> None:
        """Initialize and restore variables from checkpoint path (if
        available)

        """
        latest_checkpoint = None if saver is None else \
            tf.train.latest_checkpoint('checkpoints')

        if latest_checkpoint is not None:
            print(f"\n##\n## Restoring graph from {latest_checkpoint}\n##\n")
            saver.restore(self._tf_session, latest_checkpoint)

            global_step = self._tf_session.run(self._tf_global_step)
            print(f"Global step: {global_step}")
        else:
            if saver is not None:
                saver.save(self._tf_session, 'checkpoints/my_test_model')
