"""Testsuite for the abstract `Network` interface.
"""

# The following line allows the test to be run from within the test
# directory:
# if not __package__: import __init__

# standard imports
from typing import Tuple
from collections import OrderedDict  # new in version 2.7
from unittest import TestCase

# third party imports
import numpy as np

# toolbox imports
from dltb.network import Network as BaseNetwork


class MockLayer:

    def __init__(self, input_shape):
        self.input_shape = input_shape


class MockNetwork(BaseNetwork):
    """Mock to allow instantiation."""

    def __init__(self, data_format: str, input_shape: Tuple[int],
                 **kwargs) ->None:
        super().__init__(**kwargs)
        self._data_format = data_format
        self.layer = MockLayer(input_shape)
        self.layer_dict = OrderedDict(input_layer=self.layer)


class TestBaseNetwork(TestCase):
    def setUp(self):
        self.network_last = MockNetwork(data_format='channels_last',
                                        input_shape=(1, 28, 30, 1))
        self.network_first = MockNetwork(data_format='channels_first',
                                         input_shape=(1, 1, 28, 30))

    def test_transform_input(self):

        mock_input_sample = np.zeros((1, 28, 30, 1))

        # Check that the data format is changed correctly.
        input_sample, is_batch, is_internal = \
            self.network_last._transform_input(mock_input_sample,
                                               data_format='channels_last')
        self.assertEqual(mock_input_sample.shape, (1, 28, 30, 1))
        self.assertTrue(np.all(mock_input_sample == input_sample))

        mock_input_sample = np.zeros((1, 1, 28, 30))
        input_sample, is_batch, is_internal = \
            self.network_first._transform_input(mock_input_sample,
                                                data_format='channels_first')
        self.assertEqual(input_sample.shape, (1, 1, 28, 30))

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
