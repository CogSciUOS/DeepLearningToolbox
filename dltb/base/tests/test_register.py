
# standard imports
from unittest import TestCase

# toolbox imports
from dltb.base.register import RegisterClass


class MockClass(metaclass=RegisterClass):
    """MockClass to test the :py:class:`RegisterClass`.
    """


class MockSubclass(MockClass):
    """Subclass of :py:class:`MockClass` to test
    the :py:class:`RegisterClass`.
    """


MockClass.register_class('dltb.base.test.test_register.MockSubclass')

MockClass.register_instance('test-1', 'dltb.base.tests.test_register',
                            'MockSubclass')
MockClass.register_instance('test-2', 'dltb.base.tests.test_register',
                            'MockSubclass')


class TestRegisterClass(TestCase):
    """Tests for the :py:class:`RegisterClass` meta class.
    """

    def test_instance_register_01_lookup(self):
        key = 'test-1'
        self.assertTrue(key in MockClass)
        self.assertTrue(key in MockClass.instance_register)
        self.assertFalse(key in MockClass.class_register)

    def test_instance_register_02_instantiation(self):
        key = 'test-1'
        entry = MockClass.instance_register[key]
        self.assertFalse(entry.initialized)

        obj = MockClass[key]
        self.assertTrue(isinstance(obj, MockClass))
        self.assertTrue(entry.initialized)
