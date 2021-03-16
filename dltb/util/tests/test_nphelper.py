"""Tests for the :py:class:`ActivationTool` and the
:py:class:`ActivationWorker`
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

    def test_top_actvation_01(self):
        activations = np.asarray([4, 2, 5, 3, 7], dtype=np.float)

        top_indices = np.argsort(-activations)
        top3_indices = nphelper.argtop(activations, top=3, sort=True)
        self.assertTrue(np.array_equal(top3_indices, top_indices[:3]))

        top = np.sort(-activations)
        top3 = nphelper.top(activations, top=3, sort=True)
        self.assertTrue(np.array_equal(top3, top[:3]))

    def test_top_actvation_02(self):
        activations = np.asarray([
            [4, 7, 3],
            [2, 5, 1],
            [1, 6, 2],
            [7, 3, 6],
            [3, 4, 4],
            [8, 0, 7]
        ], dtype=np.float)

        top_indices = np.argsort(-activations, axis=0)
        top4_indices = \
            nphelper.argtop(activations, top=4, axis=0, sort=True)
        self.assertTrue(np.array_equal(top4_indices, top_indices[:4]))

        top = np.sort(-activations, axis=0)
        top4 = nphelper.top(activations, top=4, axis=0, sort=True)
        self.assertTrue(np.array_equal(top4, top[:4]))
