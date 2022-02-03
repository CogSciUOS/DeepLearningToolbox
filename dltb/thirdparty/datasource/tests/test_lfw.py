"""Testsuite for the `LabeledFacesInTheWild` datasource.


from dltb.thirdparty.datasource.lfw import LabeledFacesInTheWild
lfw = LabeledFacesInTheWild()

lfw.pair_mode = True

pair = next(lfw.pairs())
"""

from unittest import TestCase

from dltb.thirdparty.datasource.lfw import LabeledFacesInTheWild


class TestLFW(TestCase):
    """Testsuite for the `LabeledFacesInTheWild` datasource.
    """

    def setUp(self):
        """Setup the `LabeledFacesInTheWild` datasource for testing.
        """
        self.lfw = LabeledFacesInTheWild()
        self.lfw.prepare()

    def test_prepare(self):
        """Check that the prepare mechanism works as expected.
        """
        self.assertTrue(self.lfw.prepared)
        self.lfw.unprepare()
        self.assertFalse(self.lfw.prepared)
        self.lfw.prepare()
        self.assertTrue(self.lfw.prepared)

    def test_len(self):
        """Check that all 13,233 images are available.
        """
        self.assertEqual(len(self.lfw), 13233)

    def test_pair(self):
        """Check that LFW images pairs can be retrieved.
        """
        self.assertFalse(self.lfw.pair_mode)
        self.lfw.pair_mode = True
        self.assertTrue(self.lfw.pair_mode)
        self.assertEqual(len(self.lfw), 6000)
