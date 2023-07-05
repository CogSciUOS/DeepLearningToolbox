"""Providing a general metaclass to be used in the Deep Learning
Toolbox.  This class provides several hooks that allow to extend the
class functionality without having to use extra metaclasses, as a
proliferation of metaclasses can make the development of complex class
hiearchies even more challenging.

"""
# standard imports
from typing import Callable, Tuple, Union, Optional
import abc
import logging
import inspect

# toolbox imports
from ..util.debug import debug_object


class ToolboxMeta(type):
    """A metaclass providing additional parameters and hooks to extend
    object creation and initialization.

    The main idea is that `ToolboxMeta` should be the only metaclass
    defined in the Deep Learning Toolbox, as a proliferation of
    metaclasses makes maintaining the class hiearchy extremely
    difficult.  Hence this one metaclass should provide all
    functionality of interest.

    Hooks
    -----

    `ToolboxMeta` supports the following hooks, when defined in the
    respective class:

    `_constructor_hook(cls, **kwargs)`
        A class method that is called before __new__. It may adapt
        the keyword arguments or return an already initialiezd object
        (which will not be initialized again).

    `__post_init__(self)`: 
        A method that is called after obj.__init__().  This is useful
        to calling methods that expect an initiaized object.

    
    Old
    ---

    Currently metaclasses are used in the following classes:
      dltb/base/register.py:class Registrable(metaclass=ABCMeta):

      dltb/datasource/datasource.py: class Datasource(metaclass=RegisterClass):
      dltb/tool/classifier.py:       class ClassScheme(metaclass=RegisterClass)
      dltb/tool/tool.py:             class Tool(metaclass=RegisterClass):
      dltb/base/tests/extras0.py:    class MockClass(metaclass=RegisterClass):
      dltb/network/network.py:       class Network(metaclass=RegisterClass):

    ./dltb/base/register.py:
       class Registrable(metaclass=ABCMeta)
       class RegisterClass(ABCMeta)

    ./old/meta.py:class Metaclass(type):

    """

    def __call__(cls_, *args, **kwargs) -> object:
        # pylint: disable=bad-mcs-method-argument
        if hasattr(cls_, '_constructor_hook'):
            target, kwargs = cls_._constructor_hook(**kwargs)
        else:
            target = cls_

        # FIXME[concept]: sometimes the __new__ method may return an already
        # initialized object - in such cases __init__
        # (and probably also __post_init__) should not be called.
        if kwargs is None:  # constructor hook has provided an object
            # kwargs is None indicates that target is already an
            # initialized object.
            obj = target
        elif target is cls_:
            obj = super().__call__(*args, **kwargs)
        else:
            obj = target.__call__(*args, **kwargs)

        if kwargs is not None:
            getattr(obj, '__post_init__', lambda: None)()
        return obj


class ABCToolboxMeta(ToolboxMeta, abc.ABCMeta):
    # pylint: disable=too-few-public-methods
    """A metaclass for abstract classes.  This class may be provided as
    metaclass for abstract classes.

    """


class ToolboxObject(debug_object, metaclass=ToolboxMeta):
    """A base class for various Toolbox classes. This base class provides
    some methods used by the ``ToolboxMeta`` metaclass.  So, instead
    of marking a class as `metaclass=ToolboxMeta`, it is usually more
    appropriate to just derive the class from `ToolboxObject`.

    """

class ABCToolboxObject(metaclass=ABCToolboxMeta):
    # pylint: disable=too-few-public-methods
    """A base class for marking abstract toolbox classes.

    An abstract toolbox object (derived from `abc.ABC`) has
    `abc.ABCMeta` as its metaclass.  To avoid conflicts with
    `ToolboxMeta` (TypeError: metaclass conflict: the metaclass of a
    derived class must be a (non-strict) subclass of the metaclasses
    of all its bases) [1] one should use `ABCToolboxMeta` as a
    metaclass, which combines `abc.ABCMeta` and `ToolboxMeta`.  This
    an be conveniently achieved by deriving the object from
    `ABCToolboxObject` (or by explicitly setting
    `metaclass=ABCToolboxMeta`).

    The `ABCToolboxObject` class can also become relevant when
    deriving an abstract class (having `abc.ABC` as a base class) from
    a non-abstract Toolbox base class, i.e. a class being
    (indirectly) derived from `ToolboxObject` and hence having
    `ToolboxMeta` as metaclass. Here one can add `ABCToolboxObject`
    as a baseclass, or explicitly provide `metaclass=ABCToolboxMeta`.

    A third alternative is to provide the `ABC` symbol defined below
    (an instance of the :py:class:`_MetaMROResolver`) as a base class.
    This will automatically add `abc.ABC` and/or `ABCToolboxObject` as
    base classes.

    References
    ----------
    [1] https://docs.python.org/3/reference/datamodel.html
        #determining-the-appropriate-metaclass

    """

class _MetaMROResolver:
    # pylint: disable=too-few-public-methods
    """The `_MetaMROResolver` is an auxiliary class to support the
    correct choice of the Toolbox Metaclass if combined with other
    metaclasses like `abc.ABCMeta`.

    The `_MetaMROResolver` implements the :py:meth:`__mro_entries__`
    method [1] to adapt the base classes in a way that should avoid
    metaclass conflicts when using the `ToolboxObject` as a base
    class.  It does so by inserting a subclass of `ToolboxObject` that
    has an appropriate metaclass compatible with the other base
    classes [2].

    Remark: this is not a perfect solution, as the `_MetaMROResolver`
    will only be used if `ABC` is explicitly listed as base class, but
    not, if the `ToolboxMeta` metaclass is introduced implicitly
    (e.g., by one of the base classes being a `ToolboxObject`).  It
    would be better if we could adapt the determination of the
    appropriate metaclass [2] directly, but I hav not found a way to
    achieve this.

    Remark 2: PyLint is not happy with this approach, as `ABC` is not
    a class (however, according to [3] it is required to not be an
    instance of type, i.e., a non-class, for the `__mro_entries__`
    mechanism to work).  Pylint can be made silent by disabling
    `inherit-non-class`.

    References
    ----------
    [1] https://docs.python.org/3/reference/datamodel.html
        #resolving-mro-entries
    [2] https://docs.python.org/3/reference/datamodel.html
        #determining-the-appropriate-metaclass

    """
    @staticmethod
    def __mro_entries__(bases):
        include_abc = (abc.ABC,)
        include_mixin = (ABCToolboxObject, )
        for base in bases:
            if base is abc.ABC:
                include_abc = ()
            elif base is not ABC and isinstance(base, ABCToolboxMeta):
                include_mixin = ()
        return include_mixin + include_abc


ABC = _MetaMROResolver()



class Constructable(ToolboxObject, metaclass=ABCToolboxMeta):
    """A baseclass for classes that implement the
    :py:meth:`_constructor_hook` method.

    Currently, the only class implementing that method is
    :py:class:`dltb.base.initialize.Initializable`.
    """

    @classmethod
    @abc.abstractmethod
    def _constructor_hook(cls_, **kwargs) \
            -> Tuple[Union[type, object], Optional[dict]]:
        # pylint: disable=bad-classmethod-argument
        """Prepare the class arguments passed to create a class.  These are
        the arguments passed to ``__new__`` and later to ``__init__``.

        Result
        ------
        target:
            The target class to be initialized or an already initialized
            object of that class.
        kwargs:
            Arguments to be used to initialize the class.  Should be `None`
            if ``target`` is an already initialized object.
        """
        return cls_, kwargs


class Postinitializable(ToolboxObject):
    # pylint: disable=attribute-defined-outside-init
    """Abstract base for classes that require some additional action after
    the class has been initialized.

    There are two ways how this additional action can be realized:

    * by overriding the `__post_init__()` method.  This method takes
      no arguments and should call `super().__post_init__()`.

    * by registering a post-init hook with `_add_post_init_hook`.
      This can be any `Callable` and include the option to pass arguments.
      
    """

    def __post_init__(self):
        """Perform post initialization actions.  This method is called
        once the `__init__` process is finished, that is, the object
        is fully initialized.

        """
        # run the post init hooks
        actions = getattr(self, '_post_init_hooks', None)
        if actions is not None:
            for func, args, kwargs in actions:
                func(*args, **kwargs)
            # post init hooks are no longer needed -> remove them
            del self._post_init_hooks

    def _add_post_init_hook(self, func: Callable, *args, **kwargs) -> None:
        """Register a ``Callable`` to be run after initialization, that
        is, after the ``__init__`` method has fininished.
        """
        actions = getattr(self, '_post_init_hooks', [])
        actions.append((func, args, kwargs))
        self._post_init_hooks = actions


#
# FIXME[old]
#

# Logging
LOG = logging.getLogger(__name__)

# FIXME[bug]: if a Preparable subclass' __init__ method does not have variadic
# arguments (**kwargs), this will currently cause an error, as the _cls
# argument cannot be passed ...
class OldPostinitializable:
    # pylint: disable=too-few-public-methods
    """A class supporting a post init hook :py:meth:`__post_init__`, that
    is a method that is called immediately after the class was
    initialized (by the :py:meth:`__init__`).
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if '__init__' in cls.__dict__:
            cls._postinit_original_init = cls.__init__
            cls.__init__ = Postinitializable.__init__

    def __init__(self, *args, _cls: type = None, **kwargs):
        # As we use this method, i.e.,  Postinitializable.__init__,
        # in multiple classes as __init__ method (via the assignment in
        # __init_subclass__), super() will not work as expected (it will
        # always refer to the superclass of Postinitializable, not to
        # the superclass of the class to which the method was assigned).
        # Hence we we have to do a manual lookup of the superclass.
        # We do this by passing an additional argument _cls, referring
        # to the last class in the MRO in which __init__ was involved and
        # then explicitly progress in the MRO:

        # obtain the original __init__ method ...
        meth = (self._postinit_original_init if _cls is None else
                super(_cls, self)._postinit_original_init)
        # ... get the class in which that method was defined ...
        meth_cls = getattr(inspect.getmodule(meth),
                           meth.__qualname__.rsplit('.', 1)[0])
        # ... and call it!
        meth(*args, _cls=meth_cls, **kwargs)

        # in the outermost __init__ method, we call __post_init__
        if _cls is None:
            self.__post_init__()

    def _postinit_original_init(self, *args, _cls: int = None, **kwargs):
        super().__init__(*args, **kwargs)

    def __post_init__(self):
        LOG.debug("Postinitializable.__post_init__")
