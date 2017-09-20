from unittest import TestCase
import numpy as np
from network import BaseNetwork
from frozendict import FrozenOrderedDict

class MockLayer:

    def __init__(self, input_shape):
        self.input_shape = input_shape


class MockNetwork(BaseNetwork):
    """Mock to allow instantiation."""
    def __init__(self, **kwargs):
        self._data_format = kwargs['data_format']
        self.layer_dict = FrozenOrderedDict(input_layer=MockLayer(input_shape=kwargs['input_shape']))


class TestBaseNetwork(TestCase):
    def setUp(self):
        self.network_last = MockNetwork(data_format='channels_last', input_shape=(1, 28, 30, 1))
        self.network_first = MockNetwork(data_format='channels_first', input_shape=(1, 1, 28, 30))


    def test_transform_input(self):

        mock_input_sample = np.zeros((1, 28, 30, 1))

        # Check that the data format is changed correctly.
        input_sample = self.network_last._transform_input(mock_input_sample, data_format='channels_last')
        self.assertTrue(np.all(mock_input_sample == input_sample))

        input_sample = self.network_first._transform_input(mock_input_sample, data_format='channels_last')
        self.assertTrue(input_sample.shape == (1, 1, 28, 30))

    def test_fill_up_ranks(self):


        rank_2_input = np.zeros((28, 30))
        input_samples = self.network_last._fill_up_ranks(rank_2_input)
        self.assertEqual((1, 28, 30, 1), input_samples.shape)

        batch_only_input = np.zeros((1, 28, 30))
        input_samples = self.network_last._fill_up_ranks(batch_only_input)
        self.assertEqual((1, 28, 30, 1), input_samples.shape)

        gray_channel_only_input = np.zeros((28, 30, 1))
        input_samples = self.network_last._fill_up_ranks(gray_channel_only_input)
        self.assertEqual((1, 28, 30, 1), input_samples.shape)
