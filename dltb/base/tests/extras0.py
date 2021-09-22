"""An extra module to support the tests in `test_implementation.py`
and `test_register.py`.
"""

# toolbox imports
from dltb.base.implementation import Implementable
from dltb.base.register import RegisterClass


class BaseClass(Implementable):
    """An abstract base class to be instantiated by a subclass.
    """

    def value(self) -> int:  # pylint: disable=no-self-use
        """Abstract method to be overloaded by subclasses.
        """
        return -1


# pylint: disable=too-few-public-methods
class MockClass(metaclass=RegisterClass):
    """MockClass to test the :py:class:`RegisterClass`.
    """


class MockSubclass(MockClass):
    """Subclass of :py:class:`MockClass` to test
    the :py:class:`RegisterClass`.
    """
