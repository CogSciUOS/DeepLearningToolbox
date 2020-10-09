
# standard imports
from unittest import TestCase

# third party imports
import numpy as np

# toolbox imports
from dltb.util.image import imread, imresize


class ImageTest(TestCase):
    """Tests for the :py:mod:`dltb.util.image` module.
    """
    image_file = 'images/elephant.jpg'

    def test_imread(self) -> None:
        image = imread(self.image_file)
        self.assertEqual(image.shape, (450, 300, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(tuple(image.mean(axis=(0, 1)).astype(np.uint8)),
                         (159, 153, 143))

    def test_imresize(self) -> None:
        # uint8, RGB
        image = np.random.randint(256, size=(450, 300, 3), dtype=np.uint8)
        new_image = imresize(image, size=(150, 100))
        self.assertEqual(new_image.shape, (100, 150, 3))
        self.assertEqual(new_image.dtype, np.uint8)

        # uint8, grayscale
        image = np.random.randint(256, size=(450, 300), dtype=np.uint8)
        new_image = imresize(image, size=(150, 100))
        self.assertEqual(new_image.shape, (100, 150))
        self.assertEqual(new_image.dtype, np.uint8)

        # float32, RGB
        image = np.random.rand(450, 300, 3)
        new_image = imresize(image, size=(150, 100))
        self.assertEqual(new_image.shape, (100, 150, 3))
        self.assertEqual(new_image.dtype, image.dtype)

        # float32, grayscale
        image = np.random.rand(450, 300)
        new_image = imresize(image, size=(150, 100))
        self.assertEqual(new_image.shape, (100, 150))
        self.assertEqual(new_image.dtype, image.dtype)
