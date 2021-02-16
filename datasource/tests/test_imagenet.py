from unittest import TestCase

from dltb.base.data import Data
from dltb.tool.classifier import ClassScheme


class TestData(TestCase):

    def setUp(self):
        self.imagenet_scheme = ClassScheme['ImageNet']
        self.imagenet_scheme.prepare()

    def test_scheme1(self):
        scheme = self.imagenet_scheme
        self.assertEqual(len(self.imagenet_scheme), 1000)

        # 'imagenet', 'caffe', 'torch', 'synset'
        self.assertEqual(scheme.get_label(5), 5)
        self.assertEqual(scheme.get_label(5), 5)
        self.assertEqual(scheme.get_label(5, name='imagenet'), 6)
        self.assertEqual(scheme.get_label(5, name='caffe'), 147)
        self.assertEqual(scheme.get_label(5, name='torch'), 147)
        self.assertEqual(scheme.get_label(5, name='text'), 'grey whale')
        self.assertEqual(scheme.get_label(5, name='synset'), 'n02066245')

    def test_scheme2(self):
        scheme = self.imagenet_scheme

        self.assertEqual(scheme.get_label([5, 6]), [5, 6])
        self.assertEqual(scheme.get_label((5, 6)), (5, 6))
