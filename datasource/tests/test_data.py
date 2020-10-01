from unittest import TestCase

from dltb.base.data import Data
from dltb.tool.classifier import ClassScheme


class TestData(TestCase):

    def setUp(self):
        self.imagenet_scheme = ClassScheme.register_initialize_key('ImageNet')
        self.imagenet_scheme.prepare()

    def test_data1(self):
        data = Data()
        self.assertFalse(data.is_batch)

    def test_batch(self):
        data = Data(batch=3)
        self.assertTrue(data.is_batch)
        self.assertEqual(len(data), 3)

    def test_batch2(self):
        data = Data(batch=3)
        data.add_attribute('a')
        data.add_attribute('b', batch=True)
        data.add_attribute('c', batch=False)

        self.assertFalse(data.is_batch_attribute('a'))
        self.assertTrue(data.is_batch_attribute('b'))
        self.assertFalse(data.is_batch_attribute('c'))
        self.assertFalse(data.is_batch_attribute('d'))

    def test_batch3(self):
        data = Data(batch=3)
        data.add_attribute('a')
        data.add_attribute('b', batch=True, initialize=True)
        data.a = 5
        data[1].b = 3
        data.b[2] = 4

        self.assertEqual(data.a, 5)
        self.assertEqual(data[1].a, 5)
        self.assertEqual(data[1].b, 3)
        self.assertEqual(data.b[1], 3)
        self.assertEqual(data[2].b, 4)
        self.assertEqual(data.b[2], 4)

    def test_scheme1(self):
        scheme = self.imagenet_scheme
        self.assertEqual(len(self.imagenet_scheme), 1000)

        # 'imagenet', 'caffe', 'torch', 'synset'
        self.assertEqual(scheme.get_label(5), 5)
        self.assertEqual(scheme.get_label(5, name='default'), 5)
        self.assertEqual(scheme.get_label(5, name='imagenet'), 6)
        self.assertEqual(scheme.get_label(5, name='caffe'), 147)
        self.assertEqual(scheme.get_label(5, name='torch'), 147)
        self.assertEqual(scheme.get_label(5, name='text'), 'grey whale')
        self.assertEqual(scheme.get_label(5, name='synset'), 'n02066245')

    def test_scheme2(self):
        scheme = self.imagenet_scheme

        self.assertEqual(scheme.get_label([5, 6]), [5, 6])
        self.assertEqual(scheme.get_label((5, 6)), (5, 6))
