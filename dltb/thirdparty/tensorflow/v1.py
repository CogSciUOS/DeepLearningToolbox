"""Tensorflow version 1 access module.

from dltb.thirdparty.tensorflow.v1 import tensorflow as tf
"""
import sys
import logging

import tensorflow  # pylint: disable=wrong-import-position
from . import TF_LOGGER

def disable_v2_behavior():
    """Disables TensorFlow 2.x behaviors.

    This function can be called at the beginning of the program
    (before `Tensors`, `Graphs` or other structures have been created,
    and before devices have been initialized. It switches all global
    behaviors that are different between TensorFlow 1.x and 2.x to
    behave as intended for 1.x.

    User can call this function to disable 2.x behavior during complex
    migrations.
    """
    from tensorflow.python import tf2
    from tensorflow.python.data.experimental.ops import counter
    from tensorflow.python.data.experimental.ops import interleave_ops
    from tensorflow.python.data.experimental.ops import random_ops
    from tensorflow.python.data.experimental.ops import readers as exp_readers
    from tensorflow.python.data.ops import dataset_ops
    from tensorflow.python.data.ops import readers
    from tensorflow.python.eager import monitoring
    from tensorflow.python.framework import ops
    from tensorflow.python.framework import tensor_shape
    from tensorflow.python.ops import control_flow_v2_toggles
    from tensorflow.python.ops import variable_scope

    # _v2_behavior_usage_gauge.get_cell("disable").set(True)
    tf2.disable()
    ops.disable_eager_execution()
    tensor_shape.disable_v2_tensorshape()  # Also switched by tf2

    # FIXME[bug]:
    #   Warning: disable_resource_variables (from
    #   tensorflow.python.ops.variable_scope) is deprecated and will be
    #   removed in a future version.
    #   Instructions for updating:
    #     non-resource variables are not supported in the long term
    #
    # The function tf.compat.v1.disable_resource_variables() is
    # depreciated instead you can mention use_resource=False in
    # tf.get_variable() which will be forced to true when eager excecution
    # is enabled by default in Tensorflow 2.x.
    # variable_scope.disable_resource_variables()

    ops.disable_tensor_equality()
    # Disables TensorArrayV2 and control flow V2.
    control_flow_v2_toggles.disable_control_flow_v2()
    # Make sure internal uses of tf.data symbols map to V1 versions.
    dataset_ops.Dataset = dataset_ops.DatasetV1
    readers.FixedLengthRecordDataset = readers.FixedLengthRecordDatasetV1
    readers.TFRecordDataset = readers.TFRecordDatasetV1
    readers.TextLineDataset = readers.TextLineDatasetV1
    counter.Counter = counter.CounterV1
    interleave_ops.choose_from_datasets = \
        interleave_ops.choose_from_datasets_v1
    interleave_ops.sample_from_datasets = \
        interleave_ops.sample_from_datasets_v1
    random_ops.RandomDataset = random_ops.RandomDatasetV1
    exp_readers.CsvDataset = exp_readers.CsvDatasetV1
    exp_readers.SqlDataset = exp_readers.SqlDatasetV1
    exp_readers.make_batched_features_dataset = (
        exp_readers.make_batched_features_dataset_v1)
    exp_readers.make_csv_dataset = exp_readers.make_csv_dataset_v1


if hasattr(tensorflow, 'compat'):
    v1 = getattr(tensorflow.compat, 'v1', None)
    v2 = getattr(tensorflow.compat, 'v2', None)
    # v1.disable_v2_behavior()
    old_level = TF_LOGGER.level
    TF_LOGGER.setLevel(logging.ERROR)
    disable_v2_behavior()
    TF_LOGGER.setLevel(old_level)

else:
    v1 = tensorflow
    v2 = None  # pylint: disable=invalid-name

tensorflow = v1


# FIXME[hack]: this following line will make subsequent imports of
# tensorflow use the v1 interface
sys.modules['tensorflow'] = tensorflow


del v1, v2
