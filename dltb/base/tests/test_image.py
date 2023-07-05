"""Testsuite for the `dltb.base.image` module.
"""

# standard imports
import unittest
import os

# toolbox imports
from dltb.typing import get_origin, get_args
from dltb.util.importer import importable, import_module
from dltb.base.image import Size, Sizelike, Image


class TestSize(unittest.TestCase):
    """Tests the :py:class:`Size` type.
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


class TestImage(unittest.TestCase):
    """Test the :py:class:`Image` type.
    """

    example_image_filename = 'examples/dog.jpg'
    example_image_size = Size(1546, 1213)

    def test_supported_formats(self):
        """Test supported image formats.
        """
        self.assertIn('array', Image.supported_formats())
        self.assertIn('image', Image.supported_formats())

    @unittest.skipUnless(os.path.isfile(example_image_filename),
                         reason="Example image file is not available")
    def test_image_creation(self):
        """Test creation of `Image`.
        """
        image = Image(self.example_image_filename)
        self.assertEqual(image.size, self.example_image_size)

    @unittest.skipIf(importable("PyQt5") is None or
                     not os.path.isfile(example_image_filename),
                     reason="PyQt is not available")
    def test_qt(self):
        """Test :py:class:`Size` type.
        """
        module = import_module('qtgui.widgets.image')
        self.assertIn('qimage', Image.supported_formats())

        image = Image(self.example_image_filename)
        qimage = Image.as_qimage(image)
        self.assertIsInstance(qimage, module.QImage)

    @unittest.skipIf(importable("PIL") is None or
                     not os.path.isfile(example_image_filename),
                     reason="Python Image Library (PIL/pillow)"
                     " is not available")
    def test_pillow(self):
        """Test :py:class:`Size` type.
        """
        module = import_module('dltb.thirdparty.pil')
        self.assertIn('pil', Image.supported_formats())

        image = Image(self.example_image_filename)
        pil = Image.as_pil(image)
        self.assertIsInstance(pil, module.PIL.Image.Image)
