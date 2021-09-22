"""An extra module to support the tests in `test_implementation.py`.
"""

from .extras0 import BaseClass


class MyClass1(BaseClass):
    """An implementation of :py:class:`BaseClass`.
    """

    def value(self) -> int:
        """The test value."""
        return 1
