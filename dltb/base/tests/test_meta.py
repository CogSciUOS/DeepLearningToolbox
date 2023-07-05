"""Testsuite for the `meta` module.
"""

# standard imports
from typing import Tuple
import unittest
import abc

# toolbox imports
from dltb.base import meta
from dltb.base.meta import Constructable, Postinitializable


class PostinitABC(Postinitializable, abc.ABC, metaclass=meta.ABCToolboxMeta):
    # pylint: disable=too-few-public-methods
    """A abstract postinitializable class (realized using the `abc.ABC`
    baseclass).

    """


class Postinit2ABC(Postinitializable, meta.ABC):
    # pylint: disable=too-few-public-methods,inherit-non-class
    """A abstract postinitializable class (realized using the `meta.ABC`
    baseclass).

    """


class Postinit1(PostinitABC):
    # pylint: disable=too-few-public-methods
    """A postinitializable class (should call ``__post_init__`` upon
    initialization)
    """

    def __init__(self, post: int = 1, **kwargs) -> None:
        # pylint: disable=unused-argument
        super().__init__(**kwargs)
        self.value = 1
        self._post = post

    def __post_init__(self) -> None:
        # pylint: disable=no-member
        super().__post_init__()
        self.value = self._post
        del self._post


class Postinit2(Postinit2ABC, meta.ABC):
    # pylint: disable=too-few-public-methods,disable=inherit-non-class
    """A postinitializable class (using a post init hooo).
    """
    def __init__(self, post: int = None, **kwargs) -> None:
        # pylint: disable=unused-argument
        super().__init__(**kwargs)
        self.value = 2
        if post is not None:
            self._add_post_init_hook(setattr, self, 'value', post)


class MockABC(Constructable, abc.ABC):
    # pylint: disable=too-few-public-methods
    """An abstract constructor hook class class.
    """

    @classmethod
    def _constructor_hook(cls, target: type = None,
                               **kwargs) -> Tuple[type, dict]:
        """Introduce the option to change the class of the created object.
        """
        if target is not None:
            return target, kwargs
        # pylint: disable=no-member
        return super()._constructor_hook(**kwargs)


class MockSubclass(MockABC):
    # pylint: disable=too-few-public-methods
    """A subclass of the mock class.
    """


class TestMeta(unittest.TestCase):
    """Tests for the :py:mod:`meta` module.
    """

    #
    # Postinitinalizable
    #

    def test_post_init(self) -> None:
        """Check the ``__post_init__`` method.
        """
        # check that standard classes do not call ``__post_init__``.
        obj1 = Postinit1()
        self.assertEqual(obj1.value, 1)

        obj2 = Postinit2()
        self.assertEqual(obj2.value, 2)

    def test_post_init_args(self) -> None:
        """Test passing arguments to the ``__post_init__`` method.
        """
        obj = Postinit1(post=11)
        self.assertEqual(obj.value, 11)

    def test_post_init_hook(self) -> None:
        """Test the post-init-hook mechanism, including the deletion
        of the `_post_init_hooks` attribute.
        """
        obj1 = Postinit2()
        self.assertEqual(obj1.value, 2)
        self.assertFalse(hasattr(obj1, '_post_init_hooks'))

        obj2 = Postinit2(post=23)
        self.assertEqual(obj2.value, 23)
        self.assertFalse(hasattr(obj2, '_post_init_hooks'))

    #
    # Constructable
    #

    def test_adapt_class(self) -> None:
        """Check that the class can be changed by the ``target`` argument.
        """
        # call: Mock2(target=Mock2), i.e. Mock2.__call__(target=Mock2)
        # self.assertRaises(TypeError, Mock2.__call__, target=Mock2)

        obj1 = MockABC()
        self.assertIs(type(obj1), MockABC)

        obj2 = MockSubclass()
        self.assertIs(type(obj2), MockSubclass)

        obj3 = MockABC(target=MockABC)
        self.assertIs(type(obj3), MockABC)

        obj4 = MockSubclass(target=MockABC)
        self.assertIs(type(obj4), MockABC)

        obj5 = MockABC(target=MockSubclass)
        self.assertIs(type(obj5), MockSubclass)

        obj6 = MockSubclass(target=MockSubclass)
        self.assertIs(type(obj6), MockSubclass)
