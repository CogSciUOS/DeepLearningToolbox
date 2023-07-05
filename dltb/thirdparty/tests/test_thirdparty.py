"""Testsuite for the :py:mod:`dltb.thirdparty` module.
"""

# standard imports
from unittest import TestCase

# toolbox imports
from dltb import thirdparty


class TestThirdparty(TestCase):
    """Testsuite for the :py:mod:`dltb.thirdparty` module.
    """

    def test_names(self) -> None:
        """Test if global names are set correctly.
        """
        self.assertEqual(thirdparty.DLTB, 'dltb')
        self.assertEqual(thirdparty.THIRDPARTY, 'dltb.thirdparty')
