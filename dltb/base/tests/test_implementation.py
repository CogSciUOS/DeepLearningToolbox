"""Testsuite for the `implementation` module.
"""

# standard imports
import unittest
import sys

# toolbox imports
from dltb.base.implementation import Implementable
from dltb.base.tests.extras0 import BaseClass


PARENT = Implementable.__module__.rsplit('.', maxsplit=1)[0] + '.tests'

MODULE0 = PARENT + '.extras0'
MODULE1 = PARENT + '.extras1'
MODULE2 = PARENT + '.extras2'

IMPLEMENTATION1 = MODULE1 + '.MyClass1'
IMPLEMENTATION2 = MODULE2 + '.MyClass2'

BaseClass.add_implementation(IMPLEMENTATION1)
BaseClass.add_implementation(IMPLEMENTATION2)

BaseClass.register_module_alias(MODULE1, 'extras1')
BaseClass.register_module_alias(MODULE2, 'extras2')


class TestImplementatble(unittest.TestCase):
    """Tests for the :py:class:`Data` class.
    """

    def test_01_fully_qualified_name(self):
        """Test the static method
        :py:meth:`Implementable._fully_qualified_name`.
        """
        # pylint: disable=protected-access
        self.assertEqual(Implementable._fully_qualified_name(BaseClass),
                         BaseClass.__module__ + '.BaseClass')
        self.assertEqual(BaseClass._fully_qualified_name(BaseClass),
                         BaseClass.__module__ + '.BaseClass')

    def test_02_canonical_module_name1(self):
        """Passing a module name to
        :py:meth:`Implementable._canonical_module_name`
        should return that name.
        """
        # pylint: disable=protected-access
        self.assertEqual(BaseClass._canonical_module_name(MODULE1), MODULE1)

    def test_02_canonical_module_name2(self):
        """Passing a registered module alias to
        :py:meth:`Implementable._canonical_module_name` should expand it.
        """
        # pylint: disable=protected-access
        self.assertEqual(BaseClass._canonical_module_name('extras2'), MODULE2)

    def test_02_canonical_module_name3(self):
        """Passing a module to :py:meth:`Implementable._canonical_module_name`
        should return the module name.
        """
        # pylint: disable=protected-access
        module0 = sys.modules[MODULE0]
        self.assertEqual(BaseClass._canonical_module_name(module0), MODULE0)

    def test_03_from_modules(self):
        """Test the static method :py:meth:`Implementable._from_modules`.
        """
        # pylint: disable=protected-access
        from_modules = Implementable._from_modules

        modules = (MODULE1,)
        self.assertTrue(from_modules(IMPLEMENTATION1, modules))
        self.assertFalse(from_modules(IMPLEMENTATION2, modules))
        self.assertFalse(from_modules(BaseClass, modules))

        modules = (MODULE2,)
        self.assertFalse(from_modules(IMPLEMENTATION1, modules))
        self.assertTrue(from_modules(IMPLEMENTATION2, modules))
        self.assertFalse(from_modules(BaseClass, modules))

        modules = (BaseClass.__module__,)
        self.assertFalse(from_modules(IMPLEMENTATION1, modules))
        self.assertFalse(from_modules(IMPLEMENTATION2, modules))
        self.assertTrue(from_modules(BaseClass, modules))

        modules = (MODULE1, MODULE2, BaseClass.__module__)
        self.assertTrue(from_modules(IMPLEMENTATION1, modules))
        self.assertTrue(from_modules(IMPLEMENTATION2, modules))
        self.assertTrue(from_modules(BaseClass, modules))

    def test_10_register(self):
        """Test if the register contains the correct classes:
        """
        implementations = list(BaseClass.implementations(as_str=True))
        self.assertEqual(len(implementations), 2)
        self.assertIn(IMPLEMENTATION1, implementations)
        self.assertIn(IMPLEMENTATION2, implementations)

    def test_20_load_implementation(self):
        """Test if the register contains the correct classes:
        """
        name = IMPLEMENTATION1
        implementation = BaseClass.load_implementation(name)
        self.assertTrue(issubclass(implementation, BaseClass))
        self.assertIsNot(implementation, BaseClass)

        implementations = BaseClass.implementations(loaded=True, as_str=True)
        self.assertIn(name, implementations)

    def test_30_initialization(self):
        """Instantiating the (abstract) BaseClass class should yield an
        instance of a subclass.
        """
        obj = BaseClass()
        self.assertTrue(isinstance(obj, BaseClass))
        self.assertIsNot(type(obj), BaseClass)

    def test_30_initialization1(self):
        """Instantiating the (abstract) BaseClass class with
        `module='extras1'` should yield an instance of a
        `extras1.Myclass1`.

        """
        obj1 = BaseClass(module='extras1')
        self.assertTrue(isinstance(obj1, BaseClass))
        self.assertEqual(type(obj1).__name__, 'MyClass1')
        self.assertEqual(type(obj1).__module__, MODULE1)

    def test_30_initialization2(self):
        """Instantiating the (abstract) BaseClass class with
        `module='extras2'` should yield an instance of a
        `extras2.Myclass2`.

        """
        obj2 = BaseClass(module='extras2')
        self.assertTrue(isinstance(obj2, BaseClass))
        self.assertEqual(type(obj2).__name__, 'MyClass2')
        self.assertEqual(type(obj2).__module__, MODULE2)


# FIXME[bug]: calling this from the main directy as
# python dltb/base/tests/test_implementation.py
# makes the imports fail.
if __name__ == '__main__':
    print(BaseClass.__module__)
    unittest.main()
