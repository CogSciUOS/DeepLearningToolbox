from unittest import TestCase

from .. import available, import_class


class TestOpencv(TestCase):

    def setUp(self):
        self._have_imageio = available('imageio')
        self._have_opencv = available('opencv')
        self._have_tensorflow = available('tensorflow')
        self._have_matplotlib = available('matplotlib')

    def test_available_imageio(self):
        have_imageio = available('imageio')
        self.assertTrue(have_imageio)

    def test_import_imageio(self):
        cls = import_class('ImageIO', 'opencv')
        self.assertIsInstance(cls, type)
