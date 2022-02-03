"""Tests for the :py:mod:`dltb.util.itertools` module.
"""

# standard imports
from unittest import TestCase

# toolbox imports
from dltb.util.itertools import Selection


class ItertoolsTest(TestCase):
    """Tests for the :py:mod:`dltb.util.itertools` module.
    """

    def test_selection(self) -> None:
        """Test the `Selection` iterator.
        """
        result = list(Selection('6,3-5,9-11'))
        self.assertEqual(result, [6, 3, 4, 5, 9, 10, 11])
