from unittest import TestCase

from .. import import_class


class TestOpencv(TestCase):

    def setUp(self):
        ImageReader = import_class('ImageReader', 'opencv')
        self._image_reader = ImageReader()

    def test_imread1(self):
        image = self._image_reader.read('assets/logo.png')
        self.assertEqual(image.shape, (469, 469, 3))
