# standard imports
from unittest import TestCase, skip


# thirdparty imports
import numpy as np
from dltb.thirdparty.tensorflow.v1 import tf
#import tensorflow as tf
import keras
from keras.datasets import mnist

# toolbox imports
from dltb import config
from dltb.network import NetworkParsingError
from dltb.thirdparty.keras.tensorflow import Network as KerasTensorFlowNetwork
import dltb.thirdparty.keras.tensorflow as keras_tensorflow_layers

@skip("Old API, access to private member _model which is not set.")
class TestKerasTensorFlowNetwork(TestCase):

    @classmethod
    def setUpClass(cls):
        model_file = config.model_directory / 'example_keras_mnist_model.h5'
        cls.loaded_network = KerasTensorFlowNetwork(model_file=model_file)
        cls.data = mnist.load_data()[1][0]

    @classmethod
    def tearDownClass(cls):
        # Destroy possible old tensorflow graphs.
        keras.backend.clear_session()
        tf.reset_default_graph()

    def test_get_net_input(self):
        input_image = self.data[0:1, :, :, np.newaxis]
        net_input = self.loaded_network.get_net_input('dense_2', input_image)
        net_input_op = self.loaded_network._sess.graph.get_operation_by_name('dense_2/BiasAdd')
        input_tensor = self.loaded_network._sess.graph.get_operations()[0].values()
        net_input_tf = self.loaded_network._sess.run(fetches=net_input_op.outputs[0], feed_dict={input_tensor: input_image})
        self.assertTrue(
            np.allclose(
                net_input_tf,
                net_input
            )
        )

    def test_get_activations(self):
        input_image = self.data[0:1, :, :, np.newaxis]
        activations = \
            self.loaded_network.get_activations(input_image, 'dense_2')
        prediction = self.loaded_network._model.predict(input_image)
        # Increase absolute tolerance a little to make in work.
        self.assertTrue(
            np.allclose(prediction, activations, atol=1e-6)
        )

    #
    # Test layer properties from layer dict.
    #

    def test_layer_dict(self):
        # Check the names.
        self.assertEqual(list(self.loaded_network.layer_dict.keys()),
                         ['conv2d_1',
                          'max_pooling2d_1',
                          'conv2d_2',
                          'dropout_1',
                          'flatten_1',
                          'dense_1',
                          'dropout_2',
                          'dense_2'])
        # Check that the right types where selected.
        self.assertTrue(isinstance(self.loaded_network.layer_dict['conv2d_1'], keras_tensorflow_layers.KerasTensorFlowConv2D))
        self.assertTrue(isinstance(self.loaded_network.layer_dict['max_pooling2d_1'], keras_tensorflow_layers.KerasTensorFlowMaxPooling2D))
        self.assertTrue(isinstance(self.loaded_network.layer_dict['conv2d_2'], keras_tensorflow_layers.KerasTensorFlowConv2D))
        self.assertTrue(isinstance(self.loaded_network.layer_dict['dropout_1'], keras_tensorflow_layers.KerasTensorFlowDropout))
        self.assertTrue(isinstance(self.loaded_network.layer_dict['flatten_1'], keras_tensorflow_layers.KerasTensorFlowFlatten))
        self.assertTrue(isinstance(self.loaded_network.layer_dict['dense_1'], keras_tensorflow_layers.KerasTensorFlowDense))
        self.assertTrue(isinstance(self.loaded_network.layer_dict['dropout_2'], keras_tensorflow_layers.KerasTensorFlowDropout))
        self.assertTrue(isinstance(self.loaded_network.layer_dict['dense_2'], keras_tensorflow_layers.KerasTensorFlowDense))

    # Testing the layer properties.
    def test_input_shape(self):
        self.assertEqual((None, 13, 13, 32), self.loaded_network.layer_dict['conv2d_2'].input_shape)

    def test_output_shape(self):
        self.assertEqual((None, 11, 11, 32), self.loaded_network.layer_dict['conv2d_2'].output_shape)

    def test_num_parameters(self):
        self.assertEqual(9248, self.loaded_network.layer_dict['conv2d_2'].num_parameters)

    def test_weights(self):
        self.assertTrue(
            (self.loaded_network._model.get_layer('conv2d_2').get_weights()[0] ==
             self.loaded_network.layer_dict['conv2d_2'].weights).all()
        )

    def test_bias(self):
        # layer.get_weights() gives a list containing weights and bias.
        self.assertTrue(
            (self.loaded_network._model.get_layer('conv2d_2').get_weights()[1] ==
             self.loaded_network.layer_dict['conv2d_2'].bias).all()
        )

    def test_strides(self):
        self.assertEqual(self.loaded_network._model.get_layer('conv2d_2').strides, self.loaded_network.layer_dict['conv2d_2'].strides)
        self.assertEqual(self.loaded_network._model.get_layer('max_pooling2d_1').strides, self.loaded_network.layer_dict['max_pooling2d_1'].strides)
        with self.assertRaises(AttributeError):
            self.loaded_network.layer_dict['dense_1'].strides

    def test_padding(self):
        self.assertEqual(self.loaded_network._model.get_layer('conv2d_2').strides, self.loaded_network.layer_dict['conv2d_2'].strides)
        self.assertEqual(self.loaded_network._model.get_layer('max_pooling2d_1').strides, self.loaded_network.layer_dict['max_pooling2d_1'].strides)
        with self.assertRaises(AttributeError):
            self.loaded_network.layer_dict['dense_1'].padding


    def test_kernel_size(self):
        self.assertEqual(self.loaded_network._model.get_layer('conv2d_2').kernel_size, self.loaded_network.layer_dict['conv2d_2'].kernel_size)
        with self.assertRaises(AttributeError):
            self.loaded_network.layer_dict['dense_1'].kernel_size
        with self.assertRaises(AttributeError):
            self.loaded_network.layer_dict['max_pooling2d_1'].kernel_size

    def test_filters(self):
        self.assertEqual(32,
                         self.loaded_network.layer_dict['conv2d_2'].filters)
        with self.assertRaises(AttributeError):
            self.loaded_network.layer_dict['dense_1'].filters
        with self.assertRaises(AttributeError):
            self.loaded_network.layer_dict['max_pooling2d_1'].filters

    def test_pool_size(self):
        self.assertEqual(self.loaded_network._model.get_layer('max_pooling2d_1').pool_size, self.loaded_network.layer_dict['max_pooling2d_1'].pool_size)
        with self.assertRaises(AttributeError):
            self.loaded_network.layer_dict['dense_1'].pool_size
        with self.assertRaises(AttributeError):
            self.loaded_network.layer_dict['conv2d_2'].pool_size

    #
    # Testing wrappers around layer properties.
    #

    def test_get_layer_input_shape(self):
        self.assertEqual((None, 13, 13, 32),
                         self.loaded_network.get_layer_input_shape('conv2d_2'))

    def test_get_layer_output_shape(self):
        self.assertEqual((None, 11, 11, 32),
                         self.loaded_network.get_layer_output_shape('conv2d_2'))


    def test_get_layer_weights(self):
        # _model.get_weights() gives a list with the weights of each
        # layer. "conv2d_2 is the third layer, hence at index 2.
        self.assertTrue(
            (self.loaded_network._model.get_weights()[2] ==
             self.loaded_network.get_layer_weights('conv2d_2')).all()
        )


@skip("Old API.")
class TestKerasTensorFlowParser(TestCase):
    """Test model parsing."""

    def tearDown(self):
        # Destroy possible old tensorflow graphs.
        keras.backend.clear_session()
        tf.reset_default_graph()

    def test_separate_activations(self):
        """Create a model with separate activation functions,
        to check if they will get merged.
        """
        model = keras.models.Sequential()
        model.add(keras.layers.Convolution2D(32, (3, 3),
                                             input_shape=(28, 28, 1),
                                             activation='relu'))
        model.add(keras.layers.Dense(1024))
        model.add(keras.layers.Activation('sigmoid'))

        network = KerasTensorFlowNetwork(model=model)
        self.assertEqual(['conv2d_1',
                          'dense_1'], list(network.layer_dict.keys()))

        self.assertEqual(network.layer_dict['dense_1'].input,
                         model.get_layer('dense_1').input)
        self.assertEqual(network.layer_dict['dense_1'].output,
                         model.get_layer('activation_1').output)

    def test_missing_activation(self):
        """Create a model with missing activation function to check that that
        raises an error."""
        model = keras.models.Sequential()
        model.add(keras.layers.Convolution2D(32, (3, 3),
                                             input_shape=(28, 28, 1)))
        model.add(keras.layers.Dense(1024))
        model.add(keras.layers.Activation('sigmoid'))
        with self.assertRaises(NetworkParsingError):
            KerasTensorFlowNetwork(model=model)

    def test_double_activation_function(self):
        """Create a model with two consectutive activation functions to check
        that that raises an error.

        """
        model = keras.models.Sequential()
        model.add(keras.layers.Convolution2D(32, (3, 3),
                                             input_shape=(28, 28, 1),
                                             activation='relu'))
        model.add(keras.layers.Dense(1024))
        model.add(keras.layers.Activation('sigmoid'))
        model.add(keras.layers.Activation('relu'))
        with self.assertRaises(NetworkParsingError):
            KerasTensorFlowNetwork(model=model)
