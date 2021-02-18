
# standard imports
from unittest import TestCase

# toolbox imports
from ...base.image import Image
from datasource.widerface import WiderFace
from datasource.helen import Helen


class WiderfaceTest(TestCase):
    """Tests for the :py:mod:`dltb.thirdparty.datasource` classes.
    WiderFace
    """

    def setUp(self):
        self._datasource = WiderFace()
        self._datasource.prepare()

    def test_extract_images(self) -> None:
        image = self._datasource[11862]
        self.assertEqual(type(image), Image)
        self.assertEqual(image.shape, (732, 1024, 3))
        # self.assertEqual(image.array.dtype, np.uint8)


class HelenTest(TestCase):
    """
    """

    def setUp(self):
        self._datasource = Helen()
        self._datasource.prepare()

    def test_load_image(self) -> None:
        image = self._datasource[1536]
        self.assertEqual(image.shape, (2054, 2145, 3))
