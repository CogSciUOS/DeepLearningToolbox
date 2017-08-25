from unittest import TestCase
import os
os.environ['GLOG_minloglevel'] = '2' # Suppress verbose output from caffe.
from network import CaffeNetwork
import numpy as np
from keras.datasets import mnist


class TestCaffeNetwork(TestCase):


    def setUp(self):
        model_def = '../../models/example_caffe_network_deploy.prototxt'
        model_weights = '../../models/mnist.caffemodel'
        self.network = CaffeNetwork(model_def, model_weights)
        # Load the images from the test set and normalize.
        self.data = mnist.load_data()[1][0]
        self.data = self.data / self.data.max()

    def test_layer_ids(self):
        self.assertEqual(self.network.layer_ids,
                         ['data',
                          'conv2d_1',
                          'relu_1',
                          'max_pooling2d_1',
                          'conv2d_2',
                          'relu_2',
                          'dropout_1',
                          'dense_1',
                          'dropout_2',
                          'dense_2',
                          'prob'])

    def test_get_layer_input_shape(self):
        self.assertEqual((None, 13, 13, 32), self.network.get_layer_input_shape('conv2d_2'))

    def test_get_layer_output_shape(self):
        self.assertEqual((None, 11, 11, 32), self.network.get_layer_output_shape('conv2d_2'))

    def test_get_activations(self):
        input_image = self.data[0]
        activations = self.network.get_activations(['dense_2'], input_image)
        self.assertTrue(
            np.allclose(
                np.array([0.21200967, -0.49982196, 3.12783265, 3.91348648,
                       -3.45651197, -6.23287964, -10.75421906, 13.81074047,
                       -3.08114743, 1.74546885], dtype='float32'),
                activations[0]
            )
        )
        # Check whether the network is appropriately reshaped.
        self.assertEqual(activations[0].shape, (1, 10))


    def test_get_layer_weights(self):
        weights = self.network.get_layer_weights('dense_2')
        # Check the shape.
        self.assertEqual(weights.shape, (10, 64))
        self.assertTrue(
            np.allclose(
                weights[:, 0],
                np.array([-0.15826955, 0.06935215, -0.07810633, -0.106333, 0.17425275,
                          -0.11146098, -0.01561634, 0.12711005, -0.00633655, 0.15025583], dtype='float32')
            )
        )


    def test__check_activation_args(self):

        mock_input_sample = np.zeros((1, 1, 28, 28))

        layer_ids, input_samples = self.network._check_get_activations_args(['dense_1'], mock_input_sample)
        self.assertEqual(
            layer_ids,
            ['dense_1']
        )
        self.assertTrue(np.all(mock_input_sample == input_samples))

        layer_ids, input_samples = self.network._check_get_activations_args('dense_1', mock_input_sample)
        self.assertEqual(
            layer_ids,
            ['dense_1']
        )
        self.assertTrue(np.all(mock_input_sample == input_samples))

        layer_ids, input_samples = self.network._check_get_activations_args(['dense_1', 'dense_2'], mock_input_sample)
        self.assertEqual(
            layer_ids,
            ['dense_1', 'dense_2']
        )
        self.assertTrue(np.all(mock_input_sample == input_samples))

        rank_2_input = np.zeros((28, 28))
        layer_ids, input_samples = self.network._check_get_activations_args(['dense_1', 'dense_2'], rank_2_input)
        self.assertEqual((1, 1, 28, 28), input_samples.shape)

        batch_only_input = np.zeros((1, 28, 28))
        layer_ids, input_samples = self.network._check_get_activations_args(['dense_1', 'dense_2'], batch_only_input)
        self.assertEqual((1, 1, 28, 28), input_samples.shape)

        gray_channel_only_input = np.zeros((28, 28, 1))
        layer_ids, input_samples = self.network._check_get_activations_args(['dense_1', 'dense_2'], gray_channel_only_input)
        self.assertEqual((1, 1, 28, 28), input_samples.shape)