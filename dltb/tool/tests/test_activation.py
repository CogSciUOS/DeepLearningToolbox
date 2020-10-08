"""Tests for the :py:class:`ActivationTool` and the
:py:class:`ActivationWorker`
"""

# standard imports
from unittest import TestCase

# toolbox imports
from dltb.network import Network
from dltb.base.data import Data
from dltb.util.image import imread

network = Network['alexnet-tf']
network.prepare()

image = imread('images/elephant.jpg')
image = network.image_to_internal(image)

class TestActviation(TestCase):
    """Tests for the :py:class:`RegisterClass` meta class.
    """

    def test_actviation_01(self):
        activations = network.get_activations(image, 'conv2d_1')
        self.assertEqual(activations.shape, (1, 57, 57, 96))
