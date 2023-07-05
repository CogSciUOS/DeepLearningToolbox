"""Test the OpenCV implementations.
"""

# standard imports
from unittest import TestCase

# toolbox imports
from dltb.base.image import ImageReader


class TestOpencv(TestCase):
    """Test the OpenCV implementations.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Provide resources for the test suite
        """
        cls._image_reader = ImageReader(module='opencv')

    @classmethod
    def tearDownClass(cls) -> None:
        """Release test resources
        """
        del cls._image_reader

    def test_imread1(self):
        """Check that image reader delivers expected results.
        """
        image = self._image_reader.read('assets/logo.png')
        self.assertEqual(image.shape, (469, 469, 3))
