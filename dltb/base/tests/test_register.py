"""Testsuite for the `register` module.
"""

# standard imports
import unittest

# toolbox imports
from dltb.base.tests.extras0 import MockClass


MockClass.register_class('dltb.base.test.extras0.MockSubclass')

MockClass.register_instance('test-1', 'dltb.base.tests.extras0',
                            'MockSubclass')
MockClass.register_instance('test-2', 'dltb.base.tests.extras0',
                            'MockSubclass')


class TestRegisterClass(unittest.TestCase):
    """Tests for the :py:class:`RegisterClass` meta class.
    """

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


# FIXME[bug]: calling this from the main directy as
# python dltb/base/tests/test_register.py
# makes the imports fail.
if __name__ == '__main__':
    unittest.main()
