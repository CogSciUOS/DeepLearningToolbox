"""An extra module to support the tests in `test_implementation.py`
and `test_register.py`.
"""

# toolbox imports
from dltb.base.implementation import Implementable
from dltb.base.register import RegisterClass


class BaseClass(Implementable):
    """An abstract base class to be instantiated by a subclass.
    """

    def value(self) -> int:
        """Abstract method to be overloaded by subclasses.
        """
        return -1


# pylint: disable=too-few-public-methods
class MockClass(metaclass=RegisterClass):
    """MockClass to test the :py:class:`RegisterClass` meta classs.

    This class will automatically obtain a `class_register` and an
    `instance_register` property.
    """


class MockSubclass(MockClass):
    """Subclass of :py:class:`MockClass` to test
    the :py:class:`RegisterClass`.
    """


class MockClass2:
    """A Mockup class for testing initialization.
    """

    def __init__(self, word: str = 'noword', number: int = -1) -> None:
        self.word = word
        self.number = number

    def __str__(self) -> str:
        return f"MockClass(word='{self.word}', number={self.number})"
