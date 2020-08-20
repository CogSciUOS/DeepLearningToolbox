from unittest import TestCase

from ..noise import Noise


class TestNoise(TestCase):

    def setUp(self):
        self.datasource = Noise(shape=(10, 10))

    def test_prepare(self):
        self.assertTrue(self.datasource.prepared)
        self.datasource.unprepare()
        self.assertTrue(self.datasource.prepared)

    def test_shape(self):
        data = self.datasource.get_data()
        self.assertEqual(data.shape, (10, 10))

    def test_shape2(self):
        data = self.datasource.get_random(shape=(5, 5))
        self.assertEqual(data.shape, (5, 5))
