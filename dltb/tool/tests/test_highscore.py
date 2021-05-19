"""Tests for the :py:class:`Highscore` classes.
"""

# standard imports
from unittest import TestCase

# third party imports
import numpy as np

# toolbox imports
from dltb.tool.highscore import HighscoreGroupNumpy


class TestHighscore(TestCase):
    """Tests for the :py:class:`RegisterClass` meta class.
    """
    def test_highscore_group_numpy_init1(self):
        """Test initialization of a :py:class:`HighscoreGroupNumpy`.
        """
        highscore_group = HighscoreGroupNumpy(top=5, size=3)
        self.assertEqual(highscore_group[0].owners.shape, (5,))
        self.assertEqual(highscore_group[0].scores.shape, (5,))

        self.assertEqual(highscore_group[0].owners[0], -1)
        self.assertEqual(highscore_group[0].scores[0], np.NINF)

    def test_highscore_group_numpy_init2(self):
        """Test initialization of a :py:class:`HighscoreGroupNumpy`
        with multidimensional owner representation.
        """
        highscore_group = \
            HighscoreGroupNumpy(top=5, size=3, owner_dimensions=2)
        self.assertEqual(highscore_group[0].owners.shape, (5, 2))
        self.assertEqual(highscore_group[0].scores.shape, (5,))

        self.assertEqual(highscore_group[0].owners[0].tolist(), [-1, -1])
        self.assertEqual(highscore_group[0].scores[0], np.NINF)

    def test_highscore_group_numpy_update1(self):
        """Test updating a :py:class:`HighscoreGroupNumpy`.
        """
        highscore_group = HighscoreGroupNumpy(top=5, size=3)
        owners = np.asarray([1, 2, 3, 4], np.int)
        scores = np.asarray([(10, 50, 33),
                             (20, 40, 31),
                             (30, 30, 32),
                             (40, 20, 34)], np.float)
        highscore_group.update(owners, scores)
        self.assertEqual(highscore_group[0].owners[0], owners[3])
        self.assertEqual(highscore_group[0].scores[0], scores[3, 0])

        owners2 = np.asarray([11, 22, 33], np.int)
        scores2 = np.asarray([(11, 51, 37),
                              (21, 41, 36),
                              (41, 21, 35)], np.float)

        highscore_group.update(owners2, scores2)
        self.assertEqual(highscore_group[0].owners[0], owners2[2])
        self.assertEqual(highscore_group[0].scores[0], scores2[2, 0])
        self.assertEqual(highscore_group[2].scores.tolist(),
                         [37, 36, 35, 34, 33])

    def test_highscore_group_numpy_update2(self):
        """Test updating a :py:class:`HighscoreGroupNumpy` with complex
        owner indexing.
        """
        highscore_group = \
            HighscoreGroupNumpy(top=5, size=3, owner_dimensions=2)
        owners = np.asarray([(1, 1), (2, 2), (3, 3), (4, 4)], np.int)
        scores = np.asarray([(10, 50, 33),
                             (20, 40, 31),
                             (30, 30, 32),
                             (40, 20, 34)], np.float)
        highscore_group.update(owners, scores)
        self.assertEqual(highscore_group[0].owners[0].tolist(),
                         owners[3].tolist())
        self.assertEqual(highscore_group[0].scores[0], scores[3, 0])

        owners2 = np.asarray([(11, 11), (22, 22), (33, 33)], np.int)
        scores2 = np.asarray([(11, 51, 37),
                              (21, 41, 36),
                              (41, 21, 35)], np.float)

        highscore_group.update(owners2, scores2)
        self.assertEqual(highscore_group[0].owners[0].tolist(),
                         owners2[2].tolist())
        self.assertEqual(highscore_group[0].scores[0], scores2[2, 0])
        self.assertEqual(highscore_group[2].scores.tolist(),
                         [37, 36, 35, 34, 33])
