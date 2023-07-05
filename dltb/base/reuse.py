"""A mixin to mark reusable classes.
"""
from typing import Type, Optional, Callable, Any
from functools import wraps

class Reusable:
    """A class which can remember an instance to be reused.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if Reusable in cls.__bases__:
            cls._reuse = None

    @classmethod
    def incomplete__new__(cls, reuse: bool = False, update: bool = False,
                          **kwargs) -> 'Reusable':
        """The idea would be to call `o = Reusable(reuse=True)` and then get
        the reusable instance.  However, there are some technical
        difficulties (we have to avoid that init is called again) - I
        currently have no idea how to do this without introducing a
        metaclass.
        
        Until this problem is resolved, we provide the `reuse()`
        class method below, and the @reusable decorator

        """
        if reuse:
            if cls._reuse is not None and not update:
                # FIXME[problem]: make sure that __init__ is not called:
                return cls.reuse

        # allow for mixins: pass additional arguments if the super call
        # is not at object (which does expect cls to be the only argument)
        __new__ = super().__new__
        obj = (__new__(cls) if __new__ is object.__new__ else
               __new__(cls, **kwargs))
        if reuse:
            cls.reuse = obj
        return obj

    @classmethod
    def reuse(cls, update: bool = False, **kwargs):
        # find a suitable type hint
        #  -> returns an instance of cls.
        """Get the reusable object for this class. If `None` exists
        yet, a new one will be created.

        Arguments
        ---------
        update:
            Update the reusable object by creating a new one.
        """
        if cls._reuse is None or update:
            cls._reuse = cls(**kwargs)
        return cls._reuse


class reusable:
    # pylint: disable=invalid-name,too-few-public-methods
    # It is ok for a decorator to be lower case.
    """A decorator to mark a method to use the "reusable" object, if
    called on a class instead of an instance of the class.
    """
    # Note[pylint]: this decoration confuses pylint. Currently pylints
    # diagnoses the following error when a @reusable method is called
    # directly on the class, e.g. MyClass.method(3):
    #
    #    E1120: no-value-for-parameter
    #    No value for argument 'arg' in unbound method call
    #
    # with 'arg' being the last function argument (proably all
    # arguments are shifted as self is not found)
    #
    # Workarounds:
    #  - add 'signature-mutators=dltb.base.reuse.reusable' to the
    #    to [TYPECHECK] section in pylintrc.
    #    (Seems easiest, but disables signature checking completely)
    #  - call MyClass.reuse().method()
    #  - disable error: pylint: disable=no-value-for-parameter
    #
    # Potentially related:
    #  - Decorators confuse E1120 analysis #259
    #    https://github.com/PyCQA/pylint/issues/259

    def __init__(self, func) -> None:
        self.func = func
        self.__doc__ = func.__doc__

    def __get__(self, instance: Optional[Reusable],
                cls: Type[Reusable]) -> Callable:
        if instance is None:
            instance = cls.reuse()
        # Here we can not simply bind self.func to instance (as self.func
        # is the method from the base class, not the method from the derived
        # class. Hence the following does not work:
        #
        #   return self.func.__get__(instance, type(instance))
        #
        # If we want to account for overwriting, we need to lookup
        # that overwritten method from type(instance).
        return getattr(instance, self.func.__name__)


def reusable2(method) -> Callable:
    """Alternative implementation of the decorator.
    """
    # This implementation does not work!
    # When called on a class as MyClass.my_method(x), the wrapper seemingly
    # does not get that MyClass as the first argument cls_or_self. Instead
    # that argument is set to x.
    # When called on an object, the wrapper works like expected.
    name = method.__name__
    @wraps(method)
    def wrapper(cls_or_self, *args, **kwargs) -> Any:
        print(f"cls_or_self={cls_or_self}, args={args}, kwargs={kwargs}")
        obj = (cls_or_self.reuse() if isinstance(cls_or_self, type)
               else cls_or_self)
        return getattr(obj, name)(*args, **kwargs)
    return wrapper
