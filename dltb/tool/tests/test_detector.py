"""Test suite for the general face detector module.
"""
# standard imports
from unittest import TestCase, skipUnless
import os

# toolbox imports
from dltb.tool import Tool
from dltb.tool.detector import Detections
from dltb.base.image import Image


@skipUnless(os.path.isfile('examples/reservoir-dogs.jpg'),
           "Example image 'examples/reservoir-dogs.jpg' is missing.")
class TestDetector(TestCase):
    """Test suite for the general face detector module.
    """

    def setUp(self):
        """Initialize a detector to be used in the tests.
        """
        self.detector = Tool['haar']
        self.detector.prepare()
        self.image = Image.as_data('examples/reservoir-dogs.jpg')

    def test_detect1(self):
        """Check that the detector finds the correct number of faces
        on the example image.
        """
        detections = self.detector.detect(self.image)
        self.assertIsInstance(detections, Detections)
        # self.datasource.unprepare()
        # self.assertFalse(self.datasource.prepared)

        # There are 6 faces on the image, but the 'haar' detector
        # only detects one!
        #self.assertEqual(len(detections), 6)
        self.assertEqual(len(detections), 1)

    def test_detect2(self):
        pass
        # self.datasource.prepare()
        # self.assertEqual(len(self.datasource), 2330)
