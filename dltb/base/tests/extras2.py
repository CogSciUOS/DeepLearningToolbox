"""An extra module to support the tests in `test_implementation.py`.
"""

from .extras0 import BaseClass, Implementable


class BaseClass2(BaseClass, Implementable):
    """Another :py:class:`Implementable` base class extending
    :py:class:`BaseClass`
    """


class MyClass2(BaseClass2):
    """An implementation of :py:class:`BaseClass`.
    """

    def value(self) -> int:
        """The test value."""
        return 2
