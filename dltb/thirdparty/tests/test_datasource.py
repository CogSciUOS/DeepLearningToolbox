"""General tests for the :py:mod:`dltb.thirdparty.datasource` classes.
"""

# standard imports
from unittest import TestCase

# thirdparty imports
import numpy as np

# toolbox imports
from ...base.image import Image
from ..datasource.widerface import WiderFace
from ..datasource.helen import Helen


class WiderfaceTest(TestCase):
    """Tests for the :py:mod:`dltb.thirdparty.datasource` classes.
    WiderFace
    """

    def setUp(self):
        self._datasource = WiderFace()
        self._datasource.prepare()

    def test_size(self) -> None:
        """Check that the number of images in the dataset is as expected.
        """
        self.assertEqual(len(self._datasource), 12880)

    def test_image_type(self) -> None:
        """Check the image image.
        """
        image = self._datasource[11862]
        self.assertEqual(type(image), Image)
        self.assertEqual(image.array.dtype, np.uint8)

        # Testing specific images may fail as indices seem to vary
        # at different installations
        # FIXME[todo]: check details
        # self.assertEqual(image.shape, (732, 1024, 3))


class HelenTest(TestCase):
    """Test the Helen face datasource.
    """

    def setUp(self):
        self._datasource = Helen()
        self._datasource.prepare()

    def test_size(self) -> None:
        """Check that the number of images in the dataset is as expected.
        """
        self.assertEqual(len(self._datasource), 2330)

    def test_image_type(self) -> None:
        """Check the image image.
        """
        image = self._datasource[1536]
        self.assertEqual(type(image), Image)
        self.assertEqual(image.array.dtype, np.uint8)

        # Testing specific images may fail as indices seem to vary
        # at different installations
        # FIXME[todo]: check details
        # self.assertEqual(image.shape, (2054, 2145, 3))
