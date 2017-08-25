from unittest import TestCase
from network import build_keras_network
import numpy as np





class TestBaseNetwork(TestCase):
    def setUp(self):
        self.network = build_keras_network('../../models/example_keras_mnist_model_with_dropout.h5')


    def test__check_activation_args(self):

        mock_input_sample = np.zeros((1, 28, 28, 1))

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
        self.assertEqual((1, 28, 28, 1), input_samples.shape)

        batch_only_input = np.zeros((1, 28, 28))
        layer_ids, input_samples = self.network._check_get_activations_args(['dense_1', 'dense_2'], batch_only_input)
        self.assertEqual((1, 28, 28, 1), input_samples.shape)

        gray_channel_only_input = np.zeros((28, 28, 1))
        layer_ids, input_samples = self.network._check_get_activations_args(['dense_1', 'dense_2'], gray_channel_only_input)
        self.assertEqual((1, 28, 28, 1), input_samples.shape)


