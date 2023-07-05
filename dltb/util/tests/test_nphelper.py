"""Tests for the :py:mod:`nphelper` module.
"""

# standard imports
from unittest import TestCase

# third party imports
import numpy as np

# toolbox imports
from dltb.util import nphelper


class TestNPHelper(TestCase):
    """Tests for the :py:mod:`nphelper` module.
    """

    def test_multimax_01(self):
        """Test `nphelper.multimax` on a 1-dimensional value array.
        """
        values = np.asarray([4, 2, 5, 3, 7], dtype=nphelper.np_float)

        top3_indices = nphelper.argmultimax(values, num=3, sort=True)
        self.assertTrue(np.array_equal(top3_indices, [4, 2, 0]))

        top3 = nphelper.multimax(values, num=3, sort=True)
        self.assertTrue(np.array_equal(top3, [7, 5, 4]))

    def test_multimax_02a(self):
        """Test `nphelper.multimax` a 2-dimensional value array
        (along axis=0).
        """
        values = np.asarray([
            [4, 7, 3],
            [2, 5, 1],
            [1, 6, 2],
            [7, 3, 6],
            [3, 4, 4],
            [8, 0, 7]
        ], dtype=nphelper.np_float)

        top_indices = np.argsort(-values, axis=0)
        top4_indices = \
            nphelper.argmultimax(values, num=4, axis=0, sort=True)
        self.assertTrue(np.array_equal(top4_indices, top_indices[:4]))

        top = -np.sort(-values, axis=0)
        top4 = nphelper.multimax(values, num=4, axis=0, sort=True)
        self.assertTrue(np.array_equal(top4, top[:4]))

    def test_multimax_02b(self):
        """Test `nphelper.multimax` a 2-dimensional value array
        (along axis=1).
        """
        values = np.asarray([
            [4, 7, 3],
            [2, 5, 1],
            [1, 6, 2],
            [7, 3, 6],
            [3, 4, 4],
            [8, 0, 7]
        ], dtype=nphelper.np_float)

        top_indices = np.argsort(-values, axis=1)
        top2_indices = \
            nphelper.argmultimax(values, num=2, axis=1, sort=True)
        self.assertTrue(np.array_equal(top2_indices, top_indices[:, :2]))

        top = -np.sort(-values, axis=1)
        top2 = nphelper.multimax(values, num=2, axis=1, sort=True)
        self.assertTrue(np.array_equal(top2, top[:, :2]))
