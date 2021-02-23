from unittest import TestCase

from ..datasource import KerasDatasource


class TestKeras(TestCase):

    def setUp(self):
        self.datasource = KerasDatasource('mnist', 'test')

    def test_prepare(self):
        self.datasource.prepare()
        self.assertTrue(self.datasource.prepared)
        self.datasource.unprepare()
        self.assertFalse(self.datasource.prepared)

    def test_len(self):
        self.datasource.prepare()
        self.assertEqual(len(self.datasource), 10000)
