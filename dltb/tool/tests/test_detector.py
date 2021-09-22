from unittest import TestCase

from dltb.tool import Tool
from dltb.tool.detector import Detections
from dltb.base.image import Image


class TestDetector(TestCase):

    def setUp(self):
        self.detector = Tool['haar']
        self.detector.prepare()
        # self.image = imread('examples/reservoir-dogs.jpg')
        self.image = Image.as_data('examples/reservoir-dogs.jpg')

    def test_detect1(self):
        detections = self.detector.detect(self.image)
        self.assertTrue(isinstance(detections, Detections))
        # self.datasource.unprepare()
        # self.assertFalse(self.datasource.prepared)
        self.assertEqual(len(detections), 6)

    def test_detect2(self):
        pass
        # self.datasource.prepare()
        # self.assertEqual(len(self.datasource), 2330)
