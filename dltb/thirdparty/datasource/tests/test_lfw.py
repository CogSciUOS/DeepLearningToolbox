from unittest import TestCase

from dltb.thirdparty.datasource.lfw import LabeledFacesInTheWild

"""
from dltb.thirdparty.datasource.lfw import LabeledFacesInTheWild
lfw = LabeledFacesInTheWild()

lfw.pair_mode = True

pair = next(lfw.pairs())
"""

class TestLFW(TestCase):

    def setUp(self):
        self.lfw = LabeledFacesInTheWild()
        self.lfw.prepare()

    def test_prepare(self):
        self.assertTrue(self.lfw.prepared)
        self.lfw.unprepare()
        self.assertFalse(self.lfw.prepared)
        self.lfw.prepare()
        self.assertTrue(self.lfw.prepared)

    def test_len(self):
        self.assertEqual(len(self.lfw), 13233)

    def test_pair(self):
        self.assertFalse(self.lfw.pair_mode)
        self.lfw.pair_mode = True
        self.assertTrue(self.lfw.pair_mode)
        self.assertEqual(len(self.lfw), 6000)
