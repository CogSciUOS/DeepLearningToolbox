"""Testsuite for the `mixin` module.
"""

# standard imports
import unittest

# toolbox imports
from dltb.base.mixin import Mixin


class Mock(Mixin):
    # pylint: disable=too-few-public-methods
    """A Mock class intended to be used as Mixin.
    """

    def __init__(self, value: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.value = value



class Mock2(Mixin):
    # pylint: disable=too-few-public-methods
    """A Mock class intended to be used as Mixin.
    """

    def __new__(cls, value: int = 2, **kwargs) -> None:
        obj = super().__new__(cls, **kwargs)
        obj.value = value
        return obj


class TestMixin(unittest.TestCase):
    """Tests for the :py:class:`Mixin` class.
    """

    def test_mixin_arg1(self) -> None:
        """Check initializing a Mixin object with only an `__init__` method.

        """
        obj1 = Mock(value=3)
        self.assertEqual(obj1.value, 3)

    def test_mixin_arg2(self) -> None:
        """Check initializing a Mixin object with only an `__new__` method.

        """
        obj2 = Mock2(value=4)
        self.assertEqual(obj2.value, 4)
