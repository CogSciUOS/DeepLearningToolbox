from unittest import TestCase

from ..helen import Helen


class TestHelen(TestCase):

    def setUp(self):
        self.datasource = Helen()

    def test_prepare(self):
        self.datasource.prepare()
        self.assertTrue(self.datasource.prepared)
        self.datasource.unprepare()
        self.assertFalse(self.datasource.prepared)

    def test_len(self):
        self.datasource.prepare()
        self.assertEqual(len(self.datasource), 2330)
