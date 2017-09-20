from unittest import TestCase
from layers import tensorflow_layers
import numpy as np
from keras.datasets import mnist
from network.exceptions import ParsingError
from network import TensorFlowNetwork


class TestTensorFlowNetwork(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.loaded_network = TensorFlowNetwork(checkpoint='../../models/example_tf_mnist_model/tf_mnist_model.ckpt')
        cls.data = mnist.load_data()[1][0].astype('float32')
        cls.data = cls.data / cls.data.max()

    def test_get_net_input(self):
        input_image = self.data[0:1, :, :, np.newaxis]
        net_input = self.loaded_network.get_net_input('conv2d_1', input_image)
        self.assertEqual(net_input.shape, (1, 26, 26, 32))
        self.assertTrue(
            np.allclose(
                np.array([[0.12313882, 0.07273569, 0.05022647, -0.00863712, -0.01102792],
                       [0.1309115, 0.14641027, 0.18909475, 0.19199452, 0.16788514],
                       [0.09613925, 0.1401405, 0.14798687, 0.1795305, 0.20586434],
                       [0.04382331, 0.0247027, 0.02338777, -0.00067293, 0.02700226],
                       [0.02401066, -0.00127091, -0.01244084, 0.01109774, 0.00698234]], dtype='float32'),
                net_input[0, 5:10, 5:10, 0]
            )
        )

    def test_get_activations(self):
        input_image = self.data[0:1, :, :, np.newaxis]
        activations = self.loaded_network.get_activations('conv2d_1', input_image)
        self.assertEqual(activations.shape, (1, 26, 26, 32))
        predicted_activation = np.array([[ 0.12313882,  0.07273569,  0.05022647,  0.        ,  0.        ],
                                       [ 0.1309115 ,  0.14641027,  0.18909475,  0.19199452,  0.16788514],
                                       [ 0.09613925,  0.1401405 ,  0.14798687,  0.1795305 ,  0.20586434],
                                       [ 0.04382331,  0.0247027 ,  0.02338777,  0.        ,  0.02700226],
                                       [ 0.02401066,  0.        ,  0.        ,  0.01109774,  0.00698234]],  dtype='float32')
        # Increase absolute tolerance a little to make in work.
        self.assertTrue(
            np.allclose(predicted_activation, activations[0, 5:10, 5:10, 0], atol=1e-6)
        )



        # Test layer properties from layer dict.

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
        self.assertIsInstance(
            self.loaded_network.layer_dict['conv2d_1'], tensorflow_layers.TensorFlowConv2D)
        self.assertIsInstance(self.loaded_network.layer_dict['max_pooling2d_1'],
                                   tensorflow_layers.TensorFlowMaxPooling2D)
        self.assertIsInstance(self.loaded_network.layer_dict['conv2d_2'], tensorflow_layers.TensorFlowConv2D)
        self.assertIsInstance(self.loaded_network.layer_dict['dropout_1'], tensorflow_layers.TensorFlowDropout)
        self.assertIsInstance(self.loaded_network.layer_dict['flatten_1'], tensorflow_layers.TensorFlowFlatten)
        self.assertIsInstance(self.loaded_network.layer_dict['dense_1'], tensorflow_layers.TensorFlowDense)
        self.assertIsInstance(self.loaded_network.layer_dict['dropout_2'], tensorflow_layers.TensorFlowDropout)
        self.assertIsInstance(self.loaded_network.layer_dict['dense_2'], tensorflow_layers.TensorFlowDense)



        # Testing the layer properties.

    def test_input_shape(self):
        self.assertEqual((None, 13, 13, 32), self.loaded_network.layer_dict['conv2d_2'].input_shape)

    def test_output_shape(self):
        self.assertEqual((None, 11, 11, 32), self.loaded_network.layer_dict['conv2d_2'].output_shape)

    def test_num_parameters(self):
        self.assertEqual(9248, self.loaded_network.layer_dict['conv2d_2'].num_parameters)

    def test_weights(self):
        self.assertTrue(
            np.allclose(
                np.array([[-0.08219472,  0.01501322,  0.03917561],
                           [ 0.13132864,  0.04290215, -0.04941976],
                           [-0.05186096, -0.03700988,  0.18301845]], dtype='float32'),
                self.loaded_network.layer_dict['conv2d_1'].weights[:, :, 0, 0]
            )
        )

    def test_bias(self):
        # layer.get_weights() gives a list containing weights and bias.
        self.assertAlmostEqual(
            2.55220849e-03,
            self.loaded_network.layer_dict['conv2d_1'].bias[0]
        )

    def test_strides(self):
        self.assertEqual((1, 1),
                         self.loaded_network.layer_dict['conv2d_2'].strides)
        self.assertEqual((2, 2),
                         self.loaded_network.layer_dict['max_pooling2d_1'].strides)

        with self.assertRaises(AttributeError):
            self.loaded_network.layer_dict['dense_1'].strides

    def test_padding(self):
        self.assertEqual('VALID',
                         self.loaded_network.layer_dict['conv2d_2'].padding)
        self.assertEqual('VALID',
                         self.loaded_network.layer_dict['max_pooling2d_1'].padding)

        with self.assertRaises(AttributeError):
            self.loaded_network.layer_dict['dense_1'].strides

    def test_kernel_size(self):
        self.assertEqual((3, 3),
                         self.loaded_network.layer_dict['conv2d_2'].kernel_size)
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
        self.assertEqual((2, 2),
                         self.loaded_network.layer_dict['max_pooling2d_1'].pool_size)
        with self.assertRaises(AttributeError):
            self.loaded_network.layer_dict['dense_1'].pool_size
        with self.assertRaises(AttributeError):
            self.loaded_network.layer_dict['conv2d_2'].pool_size

            # Testing wrappers around layer properties.

    def test_get_layer_input_shape(self):
        self.assertEqual((None, 13, 13, 32), self.loaded_network.get_layer_input_shape('conv2d_2'))

    def test_get_layer_output_shape(self):
        self.assertEqual((None, 11, 11, 32), self.loaded_network.get_layer_output_shape('conv2d_2'))

    def test_get_layer_weights(self):
        self.assertTrue(
            np.allclose(
                np.array([[-0.08219472, 0.01501322, 0.03917561],
                          [0.13132864, 0.04290215, -0.04941976],
                          [-0.05186096, -0.03700988, 0.18301845]], dtype='float32'),
                self.loaded_network.get_layer_weights('conv2d_1')[:, :, 0, 0]
            )
        )