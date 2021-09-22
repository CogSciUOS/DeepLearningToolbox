"""Tests for the :py:class:`ActivationTool` and the
:py:class:`ActivationWorker`
"""

# standard imports
from unittest import TestCase

# toolbox imports
from dltb.network import Network
from dltb.util.image import imread
from dltb.tool.classifier import Classifier, SoftClassifier, ImageClassifier
from dltb.tool.classifier import ClassScheme, ClassIdentifier


class TestAlexnet(TestCase):
    """Tests for the :py:class:`Alexnet` image classification network.
    """
    alexnet = Network['alexnet-tf']
    alexnet.prepare()

    image = imread('images/elephant.jpg')

    def test_alexnet_types(self):
        self.assertIsInstance(self.alexnet, Network)
        self.assertIsInstance(self.alexnet, Classifier)
        self.assertIsInstance(self.alexnet, SoftClassifier)
        self.assertIsInstance(self.alexnet, ImageClassifier)

    def test_alexnet_classification(self):
        scheme = self.alexnet.class_scheme
        self.assertIsInstance(scheme, ClassScheme)

        scheme.create_lookup_table('text')
        label_elephant = scheme.identifier('African elephant', lookup='text')
        self.assertIsInstance(label_elephant, ClassIdentifier)

        label = self.alexnet.classify(self.image)
        self.assertIsInstance(label, ClassIdentifier)

        self.assertEqual(label, label_elephant)
