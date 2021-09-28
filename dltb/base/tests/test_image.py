"""Testsuite for the `dltb.base.image` module.
"""

# standard imports
from typing import get_origin, get_args
import unittest

# toolbox imports
from dltb.base.image import Size, Sizelike


class TestSize(unittest.TestCase):
    """Tests the the :py:class:`Size ` type.
    """

    def test_size_type(self):
        """Test :py:class:`Size` type.
        """
        size_tuple = (3, 4)
        size_list = [3, 4]
        size = Size(3, 4)

        self.assertNotIsInstance(size_tuple, Size)
        self.assertNotIsInstance(size_list, Size)
        self.assertIsInstance(size, Size)

    def test_sizelike_type(self):
        """Test :py:class:`Sizelike` type.
        """
        size_tuple = (3, 4)
        size_list = [3, 4]
        size_str = '3x4'
        size = Size(3, 4)

        # TypeError: Subscripted generics cannot be used with class
        # and instance checks
        sizelike_types = \
            tuple(get_origin(arg) or arg for arg in get_args(Sizelike))
        self.assertIsInstance(size_tuple, sizelike_types)
        self.assertIsInstance(size_list, sizelike_types)
        self.assertIsInstance(size, sizelike_types)
        self.assertIsInstance(size_str, sizelike_types)

    def test_size_constructor(self):
        """Test the constructor with different aruments.
        """

        # two arguments
        size = Size(3, 4)
        self.assertEqual(size.width, 3)
        self.assertEqual(size.height, 4)

        # pair of arguments
        size_tuple = (3, 4)
        size = Size(size_tuple)
        self.assertEqual(size.width, size_tuple[0])
        self.assertEqual(size.height, size_tuple[1])

        # list of two arguments
        size_list = [3, 4]
        size = Size(size_list)
        self.assertEqual(size.width, size_list[0])
        self.assertEqual(size.height, size_list[1])

        size_str1 = '3x4'
        size = Size(size_str1)
        self.assertEqual(size.width, 3)
        self.assertEqual(size.height, 4)

        size_str2 = '3,4'
        size = Size(size_str2)
        self.assertEqual(size.width, 3)
        self.assertEqual(size.height, 4)

        size = Size(5)
        self.assertEqual(size.width, 5)
        self.assertEqual(size.height, 5)

        size = Size('5')
        self.assertEqual(size.width, 5)
        self.assertEqual(size.height, 5)

        # size = Size((5,))  # TypeError
        # self.assertEqual(size.width, 5)
        # self.assertEqual(size.height, 5)

        # size = Size([5])  # TypeError
        # self.assertEqual(size.width, 5)
        # self.assertEqual(size.height, 5)

    def test_size_constructor2(self):
        """Check if the constructor can handle the singleton case, that is
        initializing from a `Size` object should return the same
        `Size` object.
        """
        size = Size(3, 4)
        size2 = Size(size)

        self.assertEqual(size, size2)
        self.assertIs(size, size2)

    def test_size_equal(self):
        """Test size equality between different types.
        """

        # equality of size objects
        size1 = Size(3, 4)
        size2 = Size(3, 4)
        self.assertEqual(size1, size2)

        # equality of size objects and tuples
        size_tuple = (3, 4)
        self.assertEqual(size1, size_tuple)
        self.assertEqual(size_tuple, size1)

        # equality of size objects and lists
        size_list = [3, 4]
        self.assertEqual(size1, size_list)
        self.assertEqual(size_list, size1)
