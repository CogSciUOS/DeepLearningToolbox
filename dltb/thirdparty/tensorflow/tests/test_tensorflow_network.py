"""Test suite for the TensorFlow `Network` implementation.
"""

# standard imports
from unittest import TestCase, skipUnless
import os

# third-party imports
import numpy as np

# FIXME[old]: The following lines allow the test to be run from within
# the test directory (and provide the MODELS_DIRECTORY):
# if __package__: from . import MODELS_DIRECTORY
# else: from __init__ import MODELS_DIRECTORY

# toolbox imports
from dltb.config import config
from dltb.thirdparty.tensorflow import tensorflow_version_available
#import network.tensorflow

# FIXME[hack]: we need tf, but we want to make sure that it is only
# loaded after the import mechanism was patched by the deep learning
# toolbox
# pylint: disable=wrong-import-order
#import tensorflow as tf

@skipUnless(tensorflow_version_available('v1'),
            "TensorFlow version 1 is not available.")
class TestTensorFlowNetwork(TestCase):
    """Test suite for a Tensorflow `Network`.
    """
    dltb_tensorflow_network = None
    
    @classmethod
    def setUpClass(cls):
        """Load a small MNIST classifier for testing the interface.
        """
        import network.tensorflow
        cls.dltb_tensorflow_network = network.tensorflow
        TensorFlowNetwork = network.tensorflow.Network

        checkpoints = os.path.join(config.model_directory,
                                   'example_tf_mnist_model',
                                   'tf_mnist_model.ckpt')
        cls.loaded_network = TensorFlowNetwork(checkpoint=checkpoints)

        from keras.datasets import mnist
        cls.data = mnist.load_data()[1][0].astype('float32')
        cls.data = cls.data / cls.data.max()

    @classmethod
    def tearDownClass(cls):
        # FIXME[old]: AttributeError: 'Network' object has no attribute '_sess'
        # cls.loaded_network._sess.close()
        tf.reset_default_graph()

    def test_get_net_input(self):
        """Test the net input probagated to a layer.
        """

        # choose an example image and feed it through the network
        input_image = self.data[0:1, :, :, np.newaxis]
        net_input = self.loaded_network.get_net_input('conv2d_1', input_image)

        self.assertEqual(net_input.shape, (1, 26, 26, 32))

        # pylint: disable=bad-whitespace
        expected = np.array([
            [0.12313882,  0.07273569,  0.05022647, -0.00863712, -0.01102792],
            [0.1309115,   0.14641027,  0.18909475,  0.19199452,  0.16788514],
            [0.09613925,  0.1401405,   0.14798687,  0.1795305,   0.20586434],
            [0.04382331,  0.0247027,   0.02338777, -0.00067293,  0.02700226],
            [0.02401066, -0.00127091, -0.01244084,  0.01109774,  0.00698234]
        ], dtype='float32')
        self.assertTrue(np.allclose(expected, net_input[0, 5:10, 5:10, 0]))

    def test_get_activations(self):
        """Compute activation (layer output) values.
        """
        input_image = self.data[0:1, :, :, np.newaxis]
        activations = \
            self.loaded_network.get_activations(input_image, 'conv2d_1')
        self.assertEqual(activations.shape, (1, 26, 26, 32))

        # pylint: disable=bad-whitespace
        expected_activation = np.array([
            [ 0.12313882,  0.07273569,  0.05022647,  0.        ,  0.        ],
            [ 0.1309115 ,  0.14641027,  0.18909475,  0.19199452,  0.16788514],
            [ 0.09613925,  0.1401405 ,  0.14798687,  0.1795305 ,  0.20586434],
            [ 0.04382331,  0.0247027 ,  0.02338777,  0.        ,  0.02700226],
            [ 0.02401066,  0.        ,  0.        ,  0.01109774,  0.00698234]
        ],  dtype='float32')

        # Increase absolute tolerance a little to make in work.
        self.assertTrue(np.allclose(expected_activation,
                                    activations[0, 5:10, 5:10, 0], atol=1e-6))

    #
    # Test layer properties from layer dict.
    #

    def test_layer_dict(self):
        """Check that the layer dict obtained from traversing the
        network graph is correct.
        """
        expected_layers = [
            'conv2d_1',
            'max_pooling2d_1',
            'conv2d_2',
            'dropout_1',
            'flatten_1',
            'dense_1',
            'dropout_2',
            'dense_2'
        ]
        actual_layers = list(self.loaded_network.layer_dict.keys())

        # Check the names.
        self.assertEqual(actual_layers, expected_layers)
        # FIXME[bug]: currently the layer_dict only contains
        #  'conv2d_1'
        #  'max_pooling2d_1'
        #  'conv2d_2'
        #  'dense_1'

        # Check that the right types where selected.
        self.assertIsInstance(self.loaded_network.layer_dict['conv2d_1'],
                              self.dltb_tensorflow_network.Conv2D)
        self.assertIsInstance(self.loaded_network.layer_dict['max_pooling2d_1'],
                              self.dltb_tensorflow_network.MaxPooling2D)
        self.assertIsInstance(self.loaded_network.layer_dict['conv2d_2'],
                              self.dltb_tensorflow_network.Conv2D)
        self.assertIsInstance(self.loaded_network.layer_dict['dropout_1'],
                              self.dltb_tensorflow_network.Dropout)
        self.assertIsInstance(self.loaded_network.layer_dict['flatten_1'],
                              self.dltb_tensorflow_network.Flatten)
        self.assertIsInstance(self.loaded_network.layer_dict['dense_1'],
                              self.dltb_tensorflow_network.Dense)
        self.assertIsInstance(self.loaded_network.layer_dict['dropout_2'],
                              self.dltb_tensorflow_network.Dropout)
        self.assertIsInstance(self.loaded_network.layer_dict['dense_2'],
                              self.dltb_tensorflow_network.Dense)

    #
    # Testing the layer properties.
    #

    def test_input_shape(self):
        """Check the input shape of a specific layer.
        """
        layer = self.loaded_network.layer_dict['conv2d_2']
        self.assertEqual((None, 13, 13, 32), layer.input_shape)

    def test_output_shape(self):
        """Check the output shape of a specific layer.
        """
        layer = self.loaded_network.layer_dict['conv2d_2']
        self.assertEqual((None, 11, 11, 32), layer.output_shape)

    def test_num_parameters(self):
        """Check the number of paramters in a specific layer.
        """
        layer = self.loaded_network.layer_dict['conv2d_2']
        self.assertEqual(9248, layer.num_parameters)

    def test_weights(self):
        """Check the values of some selected weights.
        """
        layer = self.loaded_network.layer_dict['conv2d_1']
        # pylint: disable=bad-whitespace
        expected = np.array([
            [-0.08219472,  0.01501322,  0.03917561],
            [ 0.13132864,  0.04290215, -0.04941976],
            [-0.05186096, -0.03700988,  0.18301845]
        ], dtype='float32')

        self.assertTrue(np.allclose(expected, layer.weights[:, :, 0, 0]))

    def test_bias(self):
        """Test selected bias values.
        """
        # layer.get_weights() gives a list containing weights and bias.
        layer = self.loaded_network.layer_dict['conv2d_1']
        self.assertAlmostEqual(2.55220849e-03, layer.bias[0])

    def test_strides(self):
        """Test access to layers' `stride` attribute.
        """
        layer = self.loaded_network.layer_dict['conv2d_2']
        self.assertEqual((1, 1), layer.strides)

        layer = self.loaded_network.layer_dict['max_pooling2d_1']
        self.assertEqual((2, 2), layer.strides)

        layer = self.loaded_network.layer_dict['dense_1']
        with self.assertRaises(AttributeError):
            layer.strides  # pylint: disable=pointless-statement

    def test_padding(self):
        """Test access to layers' `padding` attribute.
        """
        layer = self.loaded_network.layer_dict['conv2d_2']
        self.assertEqual('VALID', layer.padding)
        # FIXME[bug]: instead of 'VALID' the functions returns (0, 0)

        layer = self.loaded_network.layer_dict['max_pooling2d_1']
        self.assertEqual('VALID', layer.padding)

        layer = self.loaded_network.layer_dict['dense_1']
        with self.assertRaises(AttributeError):
            layer.strides  # pylint: disable=pointless-statement

    def test_kernel_size(self):
        """Test access to layers' `kernel_size` attribute.
        """
        layer = self.loaded_network.layer_dict['conv2d_2']
        self.assertEqual((3, 3), layer.kernel_size)

        layer = self.loaded_network.layer_dict['dense_1']
        with self.assertRaises(AttributeError):
            layer.kernel_size  # pylint: disable=pointless-statement

        layer = self.loaded_network.layer_dict['max_pooling2d_1']
        with self.assertRaises(AttributeError):
            layer.kernel_size  # pylint: disable=pointless-statement

    def test_filters(self):
        """Test access to layers' `filters` attribute.
        """
        layer = self.loaded_network.layer_dict['conv2d_2']
        self.assertEqual(32, layer.filters)

        layer = self.loaded_network.layer_dict['dense_1']
        with self.assertRaises(AttributeError):
            layer.filters  # pylint: disable=pointless-statement

        layer = self.loaded_network.layer_dict['max_pooling2d_1']
        with self.assertRaises(AttributeError):
            layer.filters  # pylint: disable=pointless-statement

    def test_pool_size(self):
        """Test access to layers' `pool_size` attribute.
        """
        layer = self.loaded_network.layer_dict['max_pooling2d_1']
        self.assertEqual((2, 2), layer.pool_size)

        layer = self.loaded_network.layer_dict['dense_1']
        with self.assertRaises(AttributeError):
            layer.pool_size  # pylint: disable=pointless-statement

        layer = self.loaded_network.layer_dict['conv2d_2']
        with self.assertRaises(AttributeError):
            layer.pool_size  # pylint: disable=pointless-statement

    #
    # Testing wrappers around layer properties.
    #

    def test_get_layer_input_shape(self):
        """The `get_layer_input_shape` method.
        """
        shape = self.loaded_network.get_layer_input_shape('conv2d_2')
        self.assertEqual((None, 13, 13, 32), shape)

    def test_get_layer_output_shape(self):
        """The `get_layer_output_shape` method.
        """
        shape = self.loaded_network.get_layer_output_shape('conv2d_2')
        self.assertEqual((None, 11, 11, 32), shape)

    def test_get_layer_weights(self):
        """The `get_layer_weights` method.
        """
        weights = self.loaded_network.get_layer_weights('conv2d_1')
        reference = np.array([[-0.08219472, 0.01501322, 0.03917561],
                              [0.13132864, 0.04290215, -0.04941976],
                              [-0.05186096, -0.03700988, 0.18301845]],
                             dtype='float32')
        self.assertTrue(np.allclose(reference, weights[:, :, 0, 0]))
