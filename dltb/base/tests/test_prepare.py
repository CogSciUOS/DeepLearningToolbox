"""Testsuite for the `prepare` module.
"""

# standard imports
import unittest

# toolbox imports
from dltb.config import config
from dltb.base.prepare import Preparable


class Mock:  # pylint: disable=too-few-public-methods
    """Superclass of MockPreparable
    """
    initialized = False

    def __init__(self, **kwargs) -> None:
        # See mypy issue:
        # https://github.com/python/mypy/issues/5887
        super().__init__(**kwargs)  # type: ignore
        self.initialized = True


class MyPreparable(Preparable):
    """MockClass to test the :py:class:`RegisterClass`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._mock = False

    def _prepared(self) -> bool:
        return super()._prepared() and self._mock

    def _prepare(self) -> None:
        super()._prepare()
        self._mock = True

    def _unprepare(self) -> None:
        self._mock = False
        super()._unprepare()


class MockPreparable(Mock, MyPreparable):
    """A subclass of `Mock` that indirectly derives from `Preparable`
    and hence should be preparable.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mock2 = False

    def _prepare(self) -> None:
        super()._prepare()
        self.mock2 = self.initialized


class PreparableMock(MyPreparable, Mock):
    """Another subclass of `Mock` that indirectly derives from `Preparable`
    and hence should be preparable.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mock2 = False

    def _prepare(self) -> None:
        super()._prepare()
        self.mock2 = self.initialized


class TestPreparableClass(unittest.TestCase):
    """Tests for the :py:class:`RegisterClass` meta class.
    """

    def test_prepare_false(self):
        """Test that `Preparable` object is not prepared, when called
        with `prepare=False`.
        """
        mock = MyPreparable(prepare=False)
        self.assertFalse(mock.prepared)

        mock.prepare()
        self.assertTrue(mock.prepared)

    def test_prepare_true(self):
        """Test that `Preparable` object is prepared, when called
        with `prepare=True`.
        """
        mock = MyPreparable(prepare=True)
        self.assertTrue(mock.prepared)

        mock.unprepare()
        self.assertFalse(mock.prepared)

    def test_prepare_on_init(self):
        """Test that `Preparable` objects are prepared, depending on
        the global configuration parameter `config.prepare_on_init`.
        """
        config.prepare_on_init = False
        mock1 = MyPreparable()
        self.assertFalse(mock1.prepared)

        config.prepare_on_init = True
        mock2 = MyPreparable()
        self.assertTrue(mock2.prepared)

    def test_init_1(self):
        """Check that preparation also works correctly for
        subclasses of `Preparable`.
        """
        mock = MockPreparable(prepare=False)
        self.assertTrue(mock.initialized)
        self.assertFalse(mock.prepared)
        self.assertFalse(mock.mock2)

        mock.prepare()
        self.assertTrue(mock.prepared)
        self.assertTrue(mock.mock2)

    def test_init_1b(self):
        """Check that preparation is called after `__init__`.
        """
        mock = MockPreparable(prepare=True)
        self.assertTrue(mock.initialized)
        self.assertTrue(mock.prepared)
        self.assertTrue(mock.mock2)

    def test_init_2(self):
        """Check that preparation also works correctly for
        subclasses of `Preparable`.
        """
        mock = PreparableMock(prepare=False)
        self.assertTrue(mock.initialized)
        self.assertFalse(mock.prepared)
        self.assertFalse(mock.mock2)

        mock.prepare()
        self.assertTrue(mock.prepared)
        self.assertTrue(mock.mock2)

    def test_init_2b(self):
        """Check that preparation is called after `__init__`.
        """
        mock = PreparableMock(prepare=True)
        self.assertTrue(mock.initialized)
        self.assertTrue(mock.prepared)
        self.assertTrue(mock.mock2)


if __name__ == '__main__':
    unittest.main()
