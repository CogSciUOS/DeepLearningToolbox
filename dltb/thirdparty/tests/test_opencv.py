from unittest import TestCase

from .. import import_class


class TestOpencv(TestCase):

    def setUp(self):
        cls = import_class('ImageReader', 'opencv')
        self._image_reader = cls()

    def test_imread1(self):
        image = self._image_reader.imread('../../../assets/logo.png')
        self.assertEqual(image.shape, (24, 24))
