"""Tests for the :py:mod:`dltb.thirdparty` image classes:
:py:class:`ImageReader`, :py:class:`ImageResizer`.
"""

# standard imports
from unittest import TestCase, skipUnless

# third party imports
import numpy as np

# toolbox imports
from dltb.base.image import ImageReader, ImageResizer
from dltb.util.image import imread
from dltb.util.importer import importable


class ImageTest(TestCase):
    """Tests for the :py:mod:`dltb.thirdparty` image classes:
    :py:class:`ImageReader`, :py:class:`ImageResizer`.
    """

    #
    # Reading
    #

    image_file = 'images/elephant.jpg'

    def _image_reader_test(self, image_reader: ImageReader) -> None:
        image = image_reader.read(self.image_file)
        self.assertEqual(image.shape, (450, 300, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(tuple(image.mean(axis=(0, 1)).astype(np.uint8)),
                         (159, 153, 143))

    @skipUnless(importable('matplotlib'), "Skip matplotib tests")
    def test_reader_matplotlib(self) -> None:
        """Test matplotlib implementation of :py:class:`ImageReader`.
        """
        image_reader = ImageReader(module='matplotlib')
        self._image_reader_test(image_reader)

    @skipUnless(importable('imageio'), "Skip imageio tests")
    def test_reader_imageio(self) -> None:
        """Test imageio implementation of :py:class:`ImageReader`.
        """
        image_reader = ImageReader(module='imageio')
        self._image_reader_test(image_reader)

    @skipUnless(importable('dummy'), "Skip dummy tests")
    def test_reader_dummy(self) -> None:
        """Test dummy implementation (intended to be skipped).
        """
        image_reader = ImageReader(module='dummy')
        self._image_reader_test(image_reader)

    #
    # Resizing
    #

    # Attention: some of the resizing functions seem to change the
    # datatype (from uint to float) and also the range (from [0,256]
    # to [0.0,1.0]). This must not happen!  We need some stable and
    # well-documented resizing API.
    #
    # (originally we used scipy.misc.imresize here, which was
    # well-behaved for our purposes, but which has been deprecated and
    # is no longer present in modern versions of scipy)

    image = imread(image_file)

    def _image_resizer_test(self, image_resizer: ImageResizer) -> None:
        new_image = image_resizer.resize(self.image, size=(150, 100))
        self.assertEqual(new_image.shape, (100, 150, 3))
        self.assertEqual(new_image.dtype, np.uint8)

    @skipUnless(importable('skimage'), "Skip skimage tests")
    def test_resizer_skimage(self) -> None:
        """Test skimage implementation of :py:class:`ImageResizer`.
        """
        image_resizer = ImageResizer(module='skimage')
        self._image_resizer_test(image_resizer)

    @skipUnless(importable('opencv'), "Skip opencv tests")
    def test_resizer_opencv(self) -> None:
        """Test opencv implementation of :py:class:`ImageResizer`.
        """
        image_resizer = ImageResizer(module='opencv')
        self._image_resizer_test(image_resizer)
