"""Testsuite for the `register` module.
"""

# standard imports
import unittest

# toolbox imports
from dltb.base.register import Registrable, Register, RegisterEntry
from dltb.base.register import ClassRegister, InstanceRegister, RegisterClass
from dltb.base.tests.extras0 import MockClass, MockSubclass


MockClass.register_class('dltb.base.test.extras0.MockSubclass')

MockClass.register_instance('test-1', 'dltb.base.tests.extras0',
                            'MockSubclass')
MockClass.register_instance('test-2', 'dltb.base.tests.extras0',
                            'MockSubclass')


class MockRegisterEntry(RegisterEntry, register=True):
    """A Mockup class for testing autoamtic registration of entries.
    """
    # pylint: disable=too-few-public-methods


class TestRegister(unittest.TestCase):
    """Tests for the :py:class:`Register` class.
    """

    def test_types(self) -> None:
        """Check that register and entry types have the expected behaviour.
        """
        #self.assertIsSubclass(RegisterEntry, Registrable)
        #self.assertIsSubclass(MockRegisterEntry, Registrable)
        self.assertTrue(issubclass(RegisterEntry, Registrable))
        self.assertTrue(issubclass(MockRegisterEntry, Registrable))

        # FIXME[old]: '_register' should become 'register' once
        # InstanceRegisterEntry is updated.
        # pylint: disable=protected-access
        self.assertIsNone(RegisterEntry._register)
        self.assertIsInstance(MockRegisterEntry._register, Register)

    def test_register_registration(self) -> None:
        """Test the explicit registration in the basic :py:class:`Register`
        class.
        """
        register = Register()
        self.assertIs(type(register), Register)
        self.assertEqual(len(register), 0)

        entry1 = RegisterEntry()
        register.add(entry1)
        self.assertEqual(len(register), 1)
        self.assertIn(entry1, register)

        entry2 = RegisterEntry()
        register += entry2
        self.assertEqual(len(register), 2)
        self.assertIn(entry2, register)

        del register[entry2]
        self.assertEqual(len(register), 1)
        self.assertNotIn(entry2, register)

        del register[entry1]
        self.assertEqual(len(register), 0)
        self.assertNotIn(entry1, register)


    def test_automatic_registration(self) -> None:
        """Test automatic registration of `RegisterEntry`, using the
        mockup class :py:class:`MockRegisterEntry`.
        """
        # FIXME[old]: '_register' should become 'register' once
        # InstanceRegisterEntry is updated.
        # pylint: disable=protected-access
        register: Register = MockRegisterEntry._register
        self.assertIs(type(register), Register)
        self.assertEqual(len(register), 0)

        entry1 = MockRegisterEntry()
        self.assertEqual(len(register), 1)
        self.assertIn(entry1, register)

        entry2 = MockRegisterEntry()
        self.assertEqual(len(register), 2)
        self.assertIn(entry2, register)

        entry3 = MockRegisterEntry()
        self.assertEqual(len(register), 3)
        self.assertIn(entry3, register)

        # pylint: disable=unsupported-delete-operation
        del register[entry1]
        self.assertEqual(len(register), 2)
        self.assertNotIn(entry1, register)
        del entry1

        entry2.unregister()
        self.assertEqual(len(register), 1)
        self.assertNotIn(entry2, register)
        del entry2

        # deleting an entry will NOT autoamtically remove
        # it from the register!
        key3 = entry3.key
        del entry3
        self.assertEqual(len(register), 1)
        self.assertIn(key3, register)

        del register[key3]
        self.assertEqual(len(register), 0)
        self.assertNotIn(key3, register)


class TestRegisterClass(unittest.TestCase):
    """Tests for the :py:class:`RegisterClass` meta class.

    These tests use the `MockClass` defined in module `.extras0`, which
    has :py:class:`RegisterClass`.
    """

    def test_mock_types(self):
        """Check the `instance_register` vs. `class_register`.
        """
        self.assertTrue(issubclass(MockSubclass, MockClass))

        self.assertIsInstance(MockClass, RegisterClass)
        self.assertIsInstance(MockSubclass, RegisterClass)

    def test_class_registers(self):
        """Check the `instance_register` vs. `class_register`.
        """
        self.assertTrue(hasattr(MockClass, 'instance_register'))
        self.assertIsInstance(MockClass.instance_register, InstanceRegister)

        self.assertTrue(hasattr(MockClass, 'class_register'))
        self.assertIsInstance(MockClass.class_register, ClassRegister)

    def test_subclass_registers(self):
        """Check the `instance_register` vs. `class_register` of the subclass
        `MockSubclass` of `Mockclass`. This class should have inherited the
        registers from its superclass `Mockclass`.
        """
        self.assertTrue(hasattr(MockSubclass, 'instance_register'))
        self.assertIsInstance(MockSubclass.instance_register, InstanceRegister)
        self.assertIs(MockSubclass.instance_register, MockClass.instance_register)

        self.assertTrue(hasattr(MockSubclass, 'class_register'))
        self.assertIsInstance(MockSubclass.class_register, ClassRegister)
        self.assertIs(MockSubclass.class_register, MockClass.class_register)

    def test_class_registration(self):
        """Check that the definition of :py:class:`dltb.base.tests.extras0.MockSubclass`
        has led to automatic registration of that class in the class register.
        """
        register = MockClass.class_register

        self.assertIn('dltb.base.tests.extras0.MockClass', register)
        self.assertTrue(register.initialized('dltb.base.tests.extras0.MockClass'))
        self.assertIs(register['dltb.base.tests.extras0.MockClass'].cls, MockClass)

        self.assertIn('dltb.base.tests.extras0.MockSubclass', register)
        self.assertTrue(register.initialized('dltb.base.tests.extras0.MockSubclass'))
        self.assertIs(register['dltb.base.tests.extras0.MockSubclass'].cls, MockSubclass)

    def test_instance_register_01_lookup(self):
        """Check the `instance_register` vs. `class_register`.
        """
        key = 'test-1'
        self.assertTrue(key in MockClass)
        self.assertTrue(key in MockClass.instance_register)
        self.assertFalse(key in MockClass.class_register)

    def test_instance_register_02_instantiation(self):
        """Instantiation of a registered object via index access.
        `MockClass.instance_register[key]` should return the entry for
        the object but should not initialize it, while `MockClass[key]`
        should return the initialized instance.
        """
        key = 'test-1'
        entry = MockClass.instance_register[key]
        self.assertFalse(entry.initialized)

        obj = MockClass[key]
        self.assertTrue(isinstance(obj, MockClass))
        self.assertTrue(entry.initialized)


# FIXME[bug]: calling this from the main directory as
# python dltb/base/tests/test_register.py
# makes the imports fail.
if __name__ == '__main__':
    unittest.main()
