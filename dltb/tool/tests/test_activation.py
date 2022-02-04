"""Tests for the :py:class:`ActivationTool` and the
:py:class:`ActivationWorker`
"""

# standard imports
from unittest import TestCase, skipUnless

# third party imports
import numpy as np

# toolbox imports
from dltb.network import Network
from dltb.base.data import Data
from dltb.util.image import imread
from dltb.tool.activation import ActivationsArchiveNumpy
from dltb.tool.activation import TopActivations
from dltb.thirdparty.tensorflow import tensorflow_version_available


@skipUnless(tensorflow_version_available('v1'),
            "TensorFlow version 1 is not available.")
class TestActviation(TestCase):
    """Tests for the :py:class:`activation` module.
    """

    def setUp(cls) -> None:
        cls.network = Network['alexnet-tf']
        cls.image = imread('images/elephant.jpg')

    def test_actviation_01(self):
        activations = self.network.get_activations(self.image, 'conv2d_1')
        self.assertIsInstance(activations, np.ndarray)
        self.assertEqual(activations.shape, (57, 57, 96))

    def test_numpy_archive_01(self):
        #network = Network['alexnet']
        #datasource = datasource['imagenet-val']
        # FIXME[todo]
        ...
