
# standard imports
from unittest import TestCase

# toolbox imports
from dltb.config import config
from dltb.base.prepare import Preparable

class Mock():
    """Superclass of MockPreparable
    """
    initialized = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
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
    pass


class PreparableMock(MyPreparable, Mock):
    pass


class TestPreparableClass(TestCase):
    """Tests for the :py:class:`RegisterClass` meta class.
    """

    def test_prepare_false(self):
        mock = MyPreparable(prepare=False)
        self.assertFalse(mock.prepared)

        mock.prepare()
        self.assertTrue(mock.prepared)

    def test_prepare_true(self):
        mock = MyPreparable(prepare=True)
        self.assertTrue(mock.prepared)

        mock.unprepare()
        self.assertFalse(mock.prepared)

    def test_prepare_on_init(self):
        config.prepare_on_init = False
        mock1 = MyPreparable()
        self.assertFalse(mock1.prepared)

        config.prepare_on_init = True
        mock2 = MyPreparable()
        self.assertTrue(mock2.prepared)

    def test_init_1(self):
        mock = MockPreparable(prepare=False)
        self.assertTrue(mock.initialized)
        self.assertFalse(mock.prepared)

        mock.prepare()
        self.assertTrue(mock.prepared)

    def test_init_2(self):
        mock = PreparableMock(prepare=False)
        self.assertTrue(mock.initialized)
        self.assertFalse(mock.prepared)

        mock.prepare()
        self.assertTrue(mock.prepared)
