from unittest import TestCase
import os
os.environ['GLOG_minloglevel'] = '2' # Suppress verbose output from caffe.
from network import CaffeNetwork
import numpy as np
from keras.datasets import mnist
from layers import caffe_layers


class TestCaffeNetwork(TestCase):


    def setUp(self):
        model_def = '../../models/example_caffe_network_deploy.prototxt'
        model_weights = '../../models/mnist.caffemodel'
        self.loaded_network = CaffeNetwork(model_def=model_def, model_weights=model_weights)
        # Load the images from the test set and normalize.
        self.data = mnist.load_data()[1][0]
        self.data = self.data / self.data.max()

    # Test layer properties from layer dict.

    def test_layer_dict(self):
        print(list(self.loaded_network.layer_dict.keys()))
        self.assertEqual(
            list(self.loaded_network.layer_dict.keys()),
            ['conv2d_1',
            'max_pooling2d_1',
            'conv2d_2',
            'dropout_1',
            'flatten_1',
            'dense_1',
            'dropout_2',
            'dense_2']
        )

        # Check that the right types where selected.
        self.assertTrue(isinstance(self.loaded_network.layer_dict['conv2d_1'], caffe_layers.CaffeConv2D))
        self.assertTrue(isinstance(self.loaded_network.layer_dict['max_pooling2d_1'], caffe_layers.CaffeMaxPooling2D))
        self.assertTrue(isinstance(self.loaded_network.layer_dict['conv2d_2'], caffe_layers.CaffeConv2D))
        self.assertTrue(isinstance(self.loaded_network.layer_dict['dropout_1'], caffe_layers.CaffeDropout))
        self.assertTrue(isinstance(self.loaded_network.layer_dict['flatten_1'], caffe_layers.CaffeFlatten))
        self.assertTrue(isinstance(self.loaded_network.layer_dict['dense_1'], caffe_layers.CaffeDense))
        self.assertTrue(isinstance(self.loaded_network.layer_dict['dropout_2'], caffe_layers.CaffeDropout))
        self.assertTrue(isinstance(self.loaded_network.layer_dict['dense_2'], caffe_layers.CaffeDense))

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
                np.array([[0.00441102, 0.03252346, 0.03093702],
                       [-0.02963322, -0.01514516, 0.00517636],
                       [-0.04767472, -0.05382977, -0.00228736]], dtype='float32'),
                self.loaded_network.layer_dict['conv2d_2'].weights[0, 0]
            )
        )

    def test_bias(self):
        # layer.get_weights() gives a list containing weights and bias.
        self.assertAlmostEqual(
            -0.089870423,
            self.loaded_network.layer_dict['conv2d_2'].bias[0]
        )

    def test_strides(self):
        self.assertEqual((1, 1),
                         self.loaded_network.layer_dict['conv2d_2'].strides)
        self.assertEqual((2, 2),
                         self.loaded_network.layer_dict['max_pooling2d_1'].strides)
        with self.assertRaises(AttributeError):
            self.loaded_network.layer_dict['dense_1'].strides


    def test_kernel_size(self):
        self.assertEqual((3, 3),
                         self.loaded_network.layer_dict['conv2d_2'].kernel_size)
        with self.assertRaises(AttributeError):
            self.loaded_network.layer_dict['dense_1'].kernel_size
        with self.assertRaises(AttributeError):
            self.loaded_network.layer_dict['max_pooling2d_1'].kernel_size

    def test_pool_size(self):
        self.assertEqual((3, 3),
                         self.loaded_network.layer_dict['max_pooling2d_1'].pool_size)
        with self.assertRaises(AttributeError):
            self.loaded_network.layer_dict['dense_1'].pool_size
        with self.assertRaises(AttributeError):
            self.loaded_network.layer_dict['conv2d_2'].pool_size

    # Test loaded_network functions.

    def test_get_activations(self):
        input_image = self.data[0:1, :, :, np.newaxis]
        activations = self.loaded_network.get_activations('dense_2', input_image)
        prediction = np.array([  3.51925933e-09,   2.81613372e-10,   2.09109629e-07,
                                 1.37495732e-07,   3.44873262e-11,   7.33259398e-10,
                                 2.16445026e-13,   9.99998212e-01,   1.22597754e-09,
                                 1.40018460e-06], dtype='float32')
        # Increase absolute tolerance a little to make in work.
        self.assertTrue(
            np.allclose(prediction, activations, atol=1e-6)
        )

    def test_get_net_input(self):
        # TODO look at no inplace network for this.
        input_image = self.data[0:1, :, :, np.newaxis]
        activations = self.loaded_network.get_net_input('dense_2', input_image)
        prediction = np.array([-1.36028111, -3.88575125, 2.72432756, 2.30506134,
                               -5.98569441, -2.92878628, -11.05670547, 18.10473251,
                               -2.4147923, 4.62582827], dtype='float32')
        # Increase absolute tolerance a little to make in work.
        self.assertTrue(
            np.allclose(prediction, activations, atol=1e-6)
        )


    def test_get_layer_input_shape(self):
        self.assertEqual((None, 13, 13, 32), self.loaded_network.get_layer_input_shape('conv2d_2'))

    def test_get_layer_output_shape(self):
        self.assertEqual((None, 11, 11, 32), self.loaded_network.get_layer_output_shape('conv2d_2'))




    def test_get_layer_weights(self):
        weights = self.loaded_network.get_layer_weights('conv2d_2')[0, 0]
        self.assertTrue(
            np.allclose(
                weights,
                np.array([[0.00441102, 0.03252346, 0.03093702],
                          [-0.02963322, -0.01514516, 0.00517636],
                          [-0.04767472, -0.05382977, -0.00228736]], dtype='float32')
            )
        )

