"""Functionality to suppont mixin classes. Mixin classes are small
classes intended to be used as additional base classes of other
classes to provide some specific functionality.

"""


# pylint: disable=too-few-public-methods
class Mixin:
    """A base class for mixin classes. There are two main motives for this
    class: (1) mark a class as intended to be used as mixin and (2)
    provide some methods used by the ``MixinMeta`` metaclass.

    """

    def __new__(cls_, **kwargs) -> 'Mixin':
        # Some classes allow a keyword argument `cls` for their
        # initialization, which would conflict with the standard name
        # `cls` for the positional class argument of the new method,
        # hence use `cls_` here.
        # pylint: disable=bad-classmethod-argument
        if super().__new__ is object.__new__:
            # avoid calling object.__new__ with too many arguments
            return super().__new__(cls_)
        return super().__new__(cls_, **kwargs)
