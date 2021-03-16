"""Tests for the :py:class:`ActivationTool` and the
:py:class:`ActivationWorker`
"""

# standard imports
from unittest import TestCase

# third party imports
import numpy as np

# toolbox imports
from dltb.network import Network
from dltb.base.data import Data
from dltb.util.image import imread
from dltb.tool.activation import ActivationsArchiveNumpy
from dltb.tool.activation import TopActivations

class TestActviation(TestCase):
    """Tests for the :py:class:`RegisterClass` meta class.
    """
    network = Network['alexnet-tf']
    network.prepare()

    image = imread('images/elephant.jpg')

    def test_actviation_01(self):
        activations = self.network.get_activations(self.image, 'conv2d_1')
        self.assertTrue(isinstance(activations, np.ndarray))
        self.assertEqual(activations.shape, (57, 57, 96))

    def test_numpy_archive_01(self):
        #network = Network['alexnet']
        #datasource = datasource['imagenet-val']
        # FIXME[todo]
        ...
