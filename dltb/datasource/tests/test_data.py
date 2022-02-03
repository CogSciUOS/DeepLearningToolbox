"""Testsuite for the `Data` class.
"""

from unittest import TestCase

from dltb.base.data import Data
from dltb.tool.classifier import ClassScheme


class TestData(TestCase):
    """Testsuite for the `Data` class.
    """

    def setUp(self):
        """Setup a ClassScheme to be used in the tests.
        """
        self.imagenet_scheme = ClassScheme['ImageNet']
        self.imagenet_scheme.prepare()

    def test_data1(self):
        """Ensure that default data objects are not batches.
        """
        data = Data()
        self.assertFalse(data.is_batch)

    def test_batch(self):
        """Data objects initialized with the `batch` argument should
        be batches.
        """
        data = Data(batch=3)
        self.assertTrue(data.is_batch)
        self.assertEqual(len(data), 3)

    def test_batch2(self):
        """Using batch and non-batch attributes.
        """
        data = Data(batch=3)
        data.add_attribute('a')
        data.add_attribute('b', batch=True)
        data.add_attribute('c', batch=False)

        self.assertFalse(data.is_batch_attribute('a'))
        self.assertTrue(data.is_batch_attribute('b'))
        self.assertFalse(data.is_batch_attribute('c'))
        self.assertFalse(data.is_batch_attribute('d'))

    def test_batch3(self):
        """Testing attributes with `initalize` argument.
        """
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
