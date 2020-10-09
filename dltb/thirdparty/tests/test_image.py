
# standard imports
from unittest import TestCase, skipUnless

# third party imports
import numpy as np

# toolbox imports
from ...util.image import imread
from .. import available, import_class


class ImageTest(TestCase):
    """Tests for the :py:mod:`dltb.thirdparty` image classes:
    :py:class:`ImageReader`, :py:class:`ImageResizer`.
    """

    #
    # Reading
    #

    image_file = 'images/elephant.jpg'

    def image_reader_test(self, ImageReader: type) -> None:
        image_reader = ImageReader()
        image = image_reader.read(self.image_file)
        self.assertEqual(image.shape, (450, 300, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(tuple(image.mean(axis=(0, 1)).astype(np.uint8)),
                         (159, 153, 143))

    @skipUnless(available('matplotlib'), "Skip matplotib tests")
    def test_reader_matplotlib(self) -> None:
        ImageReader = import_class('ImageReader', 'matplotlib')
        self.image_reader_test(ImageReader)

    @skipUnless(available('imageio'), "Skip imageio tests")
    def test_reader_imageio(self) -> None:
        ImageReader = import_class('ImageReader', 'imageio')
        self.image_reader_test(ImageReader)

    @skipUnless(available('dummy'), "Skip dummy tests")
    def test_reader_dummy(self) -> None:
        ImageReader = import_class('ImageReader', 'dummy')
        self.image_reader_test(ImageReader)

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

    def image_resizer_test(self, ImageResizer: type) -> None:
        image_resizer = ImageResizer()
        new_image = image_resizer.resize(self.image, size=(150, 100))
        self.assertEqual(new_image.shape, (100, 150, 3))
        self.assertEqual(new_image.dtype, np.uint8)

    @skipUnless(available('skimage'), "Skip skimage tests")
    def test_resizer_skimage(self) -> None:
        ImageResizer = import_class('ImageResizer', 'skimage')
        self.image_resizer_test(ImageResizer)

    @skipUnless(available('opencv'), "Skip opencv tests")
    def test_resizer_opencv(self) -> None:
        ImageResizer = import_class('ImageResizer', 'opencv')
        self.image_resizer_test(ImageResizer)
