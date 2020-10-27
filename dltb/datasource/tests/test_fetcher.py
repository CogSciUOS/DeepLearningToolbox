from unittest import TestCase

import numpy as np

from ..fetcher import Datafetcher
from ..array import DataArray


class TestFetcher(TestCase):

    def setUp(self):
        self.datasource = DataArray(np.asarray([3, 4, 5, 6]))
        self.fetcher = Datafetcher(self.datasource)

    def test_prepare(self):
        self.assertTrue(self.datasource.prepared)
        self.assertEqual(len(self.datasource), 4)

    def test_shape(self):
        data = self.datasource.get_data(index=1)
        self.assertEqual(data.array, 4)

    def test_iterator(self):
        self.fetcher.reset()
        self.assertEqual(self.fetcher.data, None)
        n = next(self.fetcher)
        self.assertEqual(n.array, 3)
        self.assertEqual(self.fetcher.index, 0)
        values = [n for n in self.fetcher]
        self.assertEqual(len(values), 3)
        #self.assertEqual(values[0].array, 4)
