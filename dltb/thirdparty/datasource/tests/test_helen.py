"""Test suite for the Helen Face dataset.

"""

from unittest import TestCase

from dltb.thirdparty.datasource.helen import Helen


class TestHelen(TestCase):
    """Test suite for the Helen Face dataset.
    """

    @classmethod
    def setUpClass(cls):
        """Globally initialized the datasets for testing
        """
        # pylint: disable=attribute-defined-outside-init
        cls.datasource = Helen()

    @classmethod
    def tearDownClass(cls):
        """Release datasource resources.
        """
        del cls.datasource

    def test_prepare(self):
        """Test the prepare mechanism.
        """
        self.datasource.prepare()
        self.assertTrue(self.datasource.prepared)
        self.datasource.unprepare()
        self.assertFalse(self.datasource.prepared)

    def test_len(self):
        """Test the length of the datasources.
        """
        self.datasource.prepare()
        self.assertEqual(len(self.datasource), 2330)
