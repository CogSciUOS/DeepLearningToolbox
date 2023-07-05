"""Testsuite for the `initialize` module.
"""

# standard imports
import unittest

# toolbox imports
from dltb.base.initialize import Initializable, Initialization
from dltb.base.tests.extras0 import MockClass, MockClass2


# pylint: disable=too-few-public-methods
class MyMockClass(Initializable):
    """A mockup class for testing the :py:class:`Initializable`
    class.
    """
    def __init__(self, arg: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.arg = arg


class TestInitialization(unittest.TestCase):
    """Tests for the :py:class:`Initialization` class.
    """

    def test_initialization_create(self) -> None:
        """Check that creation of Initalization.
        """
        initialization = Initialization(key='test-1', target=MyMockClass)
        self.assertIsInstance(initialization, Initialization)
        self.assertIs(initialization.cls, MyMockClass)
        self.assertFalse(initialization.initialized)

    def test_initialization_initialize(self) -> None:
        """Check that initalization (without arguments) works.
        """
        initialization = Initialization(key='test-2', target=MyMockClass)
        obj = initialization.initialize()
        self.assertIsInstance(obj, MyMockClass)
        self.assertEqual(obj.arg, 1)
        self.assertTrue(initialization.initialized)
        self.assertIs(initialization.obj, obj)

    def test_initialization_argument(self) -> None:
        """Check that initalization (without arguments) works.
        """
        initialization = Initialization(key='test-3', target=MyMockClass,
                                        kwargs={'arg': 17})
        obj = initialization.initialize()
        self.assertIsInstance(obj, MyMockClass)
        self.assertEqual(obj.arg, 17)


class TestInitializable(unittest.TestCase):
    """Tests for the :py:class:`Initializable` class.
    """

    def test_registration(self) -> None:
        """Check that initalization (without arguments) works.
        """
        initialization = MyMockClass(initialization='test-10')
        self.assertIsInstance(initialization, Initialization)
        # linters seem not to know that instances have a key attribute
        # pylint: disable=no-member
        self.assertEqual(initialization.key, 'test-10')
        self.assertIn('test-10', MyMockClass.initialization_register)
        self.assertIs(MyMockClass.initialization_register['test-10'],
                      initialization)

    def test_initialize(self) -> None:
        """Check that initalization (with arguments) works.
        """
        # registration
        MyMockClass(initialization='test-11')

        # initialization
        obj = MyMockClass(initialize='test-11')
        self.assertIsInstance(obj, MyMockClass)
        self.assertEqual(obj.arg, 1)

    def test_initialize_args(self) -> None:
        """Check that initalization (with arguments) works.
        """
        # registration
        MyMockClass(initialization='test-12', arg=23)

        # initialization
        obj = MyMockClass(initialize='test-12')
        self.assertIsInstance(obj, MyMockClass)
        self.assertEqual(obj.arg, 23)

    def test_initialization_cls(self) -> None:
        """Check that initalization gives object of correct type.
        """
        # registration
        MyMockClass(initialization=('test-13',
                                    'dltb.base.tests.extras0.MockClass2'))

        # initialization
        obj = MyMockClass(initialize='test-13')
        self.assertIsInstance(obj, MockClass2)
        # linters do not understand that we get an object of type MockClass2
        # pylint: disable=no-member
        self.assertEqual(obj.word, 'noword')
        self.assertEqual(obj.number, -1)

    def test_initialization_combined(self) -> None:
        """Check combined initalization and registration.
        """
        # initialization & registration
        obj = MyMockClass(initialize='test-14')
        self.assertIsInstance(obj, MyMockClass)
        self.assertEqual(obj.arg, 1)

        self.assertIn('test-14', MyMockClass.initialization_register)
        initialization = MyMockClass.initialization_register['test-14']
        self.assertEqual(initialization.key, 'test-14')
        self.assertTrue(initialization.initialized)
        self.assertIs(initialization.obj, obj)

    def test_initialization_combined_cls(self) -> None:
        """Check combined initalization and registration.
        """
        # initialization & registration
        obj = MyMockClass(initialize=('test-15',
                                      'dltb.base.tests.extras0.MockClass'))
        self.assertIsInstance(obj, MockClass)

    def test_initialization_combined_exception(self) -> None:
        """Check combined initalization and registration.
        """
        # registration
        MyMockClass(initialization='test-16', arg=23)

        # initialization & registration
        self.assertRaises(ValueError, MyMockClass,
                          initialize='test-16', arg=23)
