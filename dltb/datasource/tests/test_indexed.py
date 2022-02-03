"""Testsuite for the `Data` class.
"""

# standard imports
from unittest import TestCase

# thirdparty imports
import numpy as np

# toolbox imports
from dltb.base.data import Data
from dltb.datasource import Datasource, Indexed



class TestIndexed(TestCase):
    """Testsuite for the :py:class:`Indexed` datasource.
    """

    @classmethod
    def setUpClass(cls) -> None:  # pytest: setup_class(self)
        """Setup global `Indexed` datasource for testing
        """
        # pylint: disable=attribute-defined-outside-init
        cls.datasource = Datasource(module='mnist', section='test')
        cls.data_shape = cls.datasource.shape

    @classmethod
    def tearDownClass(cls) -> None:  # pytest: teardown_class(self)
        """Release datasource resources.
        """
        del cls.datasource

    def test_is_indexed(self) -> None:
        """The datasource is a :py:class:`Indexed` datasource.
        """
        self.assertIsInstance(self.datasource, Indexed)

    def test_single(self) -> None:
        """Access a single data item via index access:
        """
        data = self.datasource[3]
        self.assertIsInstance(data, Data)
        self.assertFalse(data.is_batch)
        self.assertEqual(data.shape, self.data_shape)

    def test_slice(self) -> None:
        """Access a multiple data items via slice index access:
        """
        data = self.datasource[2:6]
        self.assertIsInstance(data, Data)
        self.assertTrue(data.is_batch)
        self.assertEqual(len(data), 4)

    def test_single_attribute(self) -> None:
        """Access one attribute for one element.
        """
        data = self.datasource[1, 'array']
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape, self.data_shape)

    def test_slice_attribute(self) -> None:
        """Access one attribute for a slice of elements.
        """
        data = self.datasource[2:6, 'array']
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape, (4,) + self.data_shape)

    def test_single_index_multiple_attributes(self) -> None:
        """Access multiple attributes for a single element.
        """
        data, label = self.datasource[1, ('array', 'label')]
        self.assertIsInstance(data, np.ndarray)
        self.assertIsInstance(label, np.uint8)
        self.assertEqual(data.shape, self.data_shape)
