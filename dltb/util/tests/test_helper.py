"""Testsuite for the `dltb.util.helper` module..
"""

# standard imports
from unittest import TestCase

# toolbox imports
from dltb.util.helper import classproperty


class Mockup:
    # pylint: disable=too-few-public-methods
    """Mockup class for testing the decorators.
    """
    _instance_counter: int = 0

    @classproperty
    def instances(cls) -> int:  # pylint: disable=no-self-argument
        """Number of instances in the class.
        """
        return cls._instance_counter

    @instances.setter
    def instances(cls, value) -> int:
        # pylint: disable=no-self-argument,no-self-use
        """Number of instances in the class.
        """
        raise ValueError(f"Cannot assign value {value} to read-only property")

    def __init__(self) -> None:
        type(self)._instance_counter += 1


class ImporterTest(TestCase):
    """Tests for the :py:mod:`dltb.util.helper` module.
    """

    def test_instances(self) -> None:
        """Testing the `@classmethod` decorator.
        """
        instances = Mockup.instances
        mockup = Mockup()
        self.assertEqual(Mockup.instances, instances+1)
        self.assertEqual(mockup.instances, instances+1)

    def no_test_instances_no_assign(self) -> None:
        """Testing the `@classmethod` decorator.
        """

        # This destroys the class property by replacing it with an integer
        # There seems to be no way to protect class properties from
        # being overwritten in this way.
        Mockup.instances = 7
        self.assertEqual(Mockup.instances, 7)

        instances = Mockup.instances
        mockup = Mockup()
        self.assertEqual(Mockup.instances, instances + 1)
        self.assertEqual(mockup.instances, instances + 1)
