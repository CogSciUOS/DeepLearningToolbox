"""A mechanism for automagically providing implementations for a given
base class.

"""
# FIXME[todo]: the following ideas have not been realized yet, but
# would increase the utility of this class:
# * adress class hierarchies: if an Implementable class has an
#   Implementable subclass, than all implementations of that subclass
#   will also be implementations of the superclass
# * add a module mechanism: it should be possible to specify under
#   which conditions modules can/should be used (for example third-party
#   and other dependencies): for example, if required thirdparty modules
#   are already loaded, then loading the module would be preferred over
#   other modules. And if required thirdparty dependencies are not installed,
#   then there is no point in trying to import it.  One may even provide
#   an option for specifying preferences.
#
# Furthermore, there seems to be some conceptual overlap with
# metaclass=RegisterClass, which also allows to register subclasses.
# Make clear the ideas behind these two approaches and explain how they
# agree and differ -> try to remove redundancies.


# standard imports
from types import ModuleType
from typing import Iterable, Union, Sequence, List, Dict, Tuple, Type, Optional
import sys
import logging
import importlib
import itertools

# logging
LOG = logging.getLogger(__name__)

Implementation = Type['Implementable']
Implementationlike = Union[str, Implementation]
Modulelike = Union[str, ModuleType]
Moduleslike = Union[Modulelike, Sequence[Modulelike]]


class Implementable:
    """A base class that can be implemented.  Implementations can be
    registered and automagically loaded and initialized calling
    Constructor of the :py:class:`Implementable` base class.

    An :py:class:`Implementable` keeps track of available
    implementations (subclasses), either classes already loaded, or
    (fully qualified) names of classes that could be loaded to
    provide an implementation.

    .. code-block:: python

       from dltb.base.implementation import Implementable

       class BaseClass(Implementable):
           ...

       BaseClass.add_implementation('mymodule.MyClass')
       BaseClass.add_implementation('other.MyClass')

    Here we define a class `BaseClass` in module `base` and register two
    implementations by calling `add_implementation`.  When an instance
    of `BaseClass` is needed, on can obtain it by simply calling
    its constructor:

    .. code-block:: python

       my_object1 = BaseClass()

       my_other_object = BaseClass(module='other')

       my_thirdparty_object = BaseClass(implementation='thirdparty.TheirClass')

    The first call will instantiate `BaseClass` using one of the registered
    classes (`mymodule.MyClass`, or `other.MyClass`).  It is also possible
    to select a specific implementation by providing a module (may still
    be ambiguous if the module provides multiple implementations) or the
    fully qualified class name.  In the last case, no prior registration
    is required.

    Registering implementations for `Implementable` classes can also be
    done before that class was defined, using the
    `Implementable.register_implementation` function:

    .. code-block:: python

       from dltb.base.implementation import Implementable

       Implementable.register_implementation('base.BaseClass',
                                             'mymodule.MyClass')
       Implementable.register_implementation('base.BaseClass',
                                             'other.MyClass')

    The ratio behind this scheme is to provide class definitions and
    their implementations in different modules or even packages.  It
    also helps to reduce the number of modules that are imported
    without actually being used: the import of the implementation is
    postponed until the class is actually used.
    """

    _implementation_register: Dict[str, Tuple[List, List]] = {}
    _module_aliases: Dict[str, str] = {}

    _implementations: List[Implementation]
    _implementation_candidates: List[str]

    @staticmethod
    def register_implementation(implementable: Implementationlike,
                                implementation: Implementationlike) -> None:
        """Register an implementation for an :py:class:`Implementable`
        class.

        Arguments
        ---------
        implementable:
            Class or fully qualified name of the class that is implemented.
        implementation:
            The actual implementation or the fully qualified name of the
            class implementing `cls`.
        """
        # obtain the implementation and candidate registers for the
        # implementable
        implementations, candidates = \
            Implementable._get_implementation_registers(implementable)

        # register the implementation by updating the registers
        if isinstance(implementation, str):
            module_name, cls_name = implementation.rsplit('.', maxsplit=1)
            if module_name in sys.modules:
                implementation = getattr(sys.modules[module_name], cls_name)

        if isinstance(implementation, type):
            if implementation not in implementations:
                implementations.append(implementation)
            implementation_name = \
                Implementable._fully_qualified_name(implementation)
            try:
                candidates.remove(implementation_name)
            except ValueError:
                pass  # the new implementation wasn't registered as a candidate
        elif implementation not in candidates:
            if implementation not in candidates:
                candidates.append(implementation)

    @staticmethod
    def _get_implementation_registers(implementable: Implementationlike) \
            -> Tuple[List[Implementation], List[str]]:
        """Obtain implementation and candidate registers for a given
        :py:class:`Implementable`.  If not registered yet, a new empty
        register is created.
        """
        if isinstance(implementable, str):
            module_name, cls_name = implementable.rsplit('.', maxsplit=1)
            try:
                implementable = getattr(sys.modules[module_name], cls_name)
            except (KeyError, AttributeError):
                # the module may not be loaded yet or it may only be
                # partially initialized, not providing the class object.
                pass

        if isinstance(implementable, str):
            cls_full_name = implementable
        else:
            if not implementable.is_implementable():
                raise TypeError("Implementations can only by added "
                                "to Implementable classes.")
            cls_full_name = Implementable._fully_qualified_name(implementable)

        # Obtain implementations from the global implementation register
        if cls_full_name not in Implementable._implementation_register:
            Implementable._implementation_register[cls_full_name] = ([], [])
        return Implementable._implementation_register[cls_full_name]

    @staticmethod
    def register_module_alias(module: Modulelike, alias: str) -> None:
        """Register a module alias.  Such an alias can be used instead
        of a full qualified module name to refer to a module.

        Arguments
        ---------
        module:
            The module for which an alias should be registered.
        alias:
            The alias.
        """
        Implementable._module_aliases[alias] = \
            module.__name__ if isinstance(module, ModuleType) else module

    @staticmethod
    def registered_implementables() -> Iterable[str]:
        """Iterate the fully qualified names for registered
        :py:class:`Implementable`s.
        """
        return Implementable._implementation_register.keys()

    @staticmethod
    def registered_implementations(implementable: Implementationlike) \
            -> Iterable[Implementationlike]:
        """Iterate the implementations of and implementation candidates
        for a given :py:class:`Implementable`.
        """
        cls_full_name = (implementable if isinstance(implementable, str) else
                         Implementable._fully_qualified_name(implementable))
        implementations, candidates = \
            Implementable._implementation_register[cls_full_name]
        return itertools.chain(implementations, candidates)

    @classmethod
    def add_implementation(cls, implementation: Implementationlike) -> None:
        """Add an implementation to an :py:class:`Implementable` class.

        Arguments
        ---------
        implementation:
            The actual implementation or the fully qualified name of the
            class implementing `cls`.
        """
        Implementable.register_implementation(cls, implementation)

    @classmethod
    def implementations(cls, loaded: bool = None,
                        as_str: bool = False) -> Iterable[Implementationlike]:
        """Iterate over the implementations for the class.

        Arguments
        ---------
        loaded:
            Indicates if only loaded implementation (`True`) or only
            implementation candidates that have not been loaded yet (`False`)
            should be iterated.  If `None`, itaration runs over both
            variants.
        as_str:
            Usually loaded implementations will be reported as classes,
            while unloaded implementations as fully qualified class names.
            If `True`, all implementations will be reported by their
            fully qualified name, independed of their load state.
        """
        if loaded is not False:
            for implementation in cls._implementations:
                if implementation.is_implementable():
                    for impl in implementation.implementations(loaded=loaded,
                                                               as_str=as_str):
                        yield impl
                else:
                    yield (Implementable._fully_qualified_name(implementation)
                           if as_str else implementation)
        if loaded is not True:
            for candidate in cls._implementation_candidates:
                yield candidate

    @classmethod
    def is_implementable(cls) -> bool:
        """Check if this class is implementable, meaning it is
        possible to register implementations for this class.
        """
        return Implementable in cls.__bases__

    @classmethod
    def is_implementation(cls) -> bool:
        """Check if this class is can serve as an implementation of an
        :py:class:`Implementable` class.
        """
        return cls is not Implementable and Implementable not in cls.__bases__

    def __init_subclass__(cls, **kwargs) -> None:
        # mypy currently has a problem with __init_subclass__
        # see: https://github.com/python/mypy/issues/4660
        super().__init_subclass__(**kwargs)  # type: ignore

        # Implementable classes: Initialize the list of
        # implementations from suitable implementations already
        # registered with Implementable.
        if cls.is_implementable():
            cls_full_name = Implementable._fully_qualified_name(cls)
            cls._implementations, cls._implementation_candidates = \
                Implementable._get_implementation_registers(cls_full_name)

        # For classes that can serve as implementations: add this new class
        # as implementation to all implementable superclasses.
        if cls.is_implementation():
            for super_cls in (_ for _ in cls.__mro__
                              if Implementable in _.__bases__):
                super_cls.add_implementation(cls)

    def __new__(cls, implementation: Optional[Implementationlike] = None,
                module: Optional[Moduleslike] = None,
                **kwargs) -> 'Implementable':

        if cls.is_implementable():
            # Class is implementable: obtain an implementation and adapt
            # the construction process
            new_cls = cls.get_implementation(module=module,
                                             implementation=implementation)
            __new__ = super(cls, new_cls).__new__
        else:
            # Class is not implementable -> resume with standard constructor
            new_cls = cls
            __new__ = super().__new__
        return (__new__(new_cls) if __new__ is object.__new__ else
                __new__(new_cls, **kwargs))

    def __init__(self, implementation: Optional[Implementationlike] = None,
                 module: Optional[Moduleslike] = None, **kwargs) -> None:
        super().__init__(**kwargs)

    @classmethod
    def load_implementation(cls, implementation: Implementationlike,
                            module: Optional[Modulelike] = None
                            ) -> Implementation:
        """Load a given implementation for this class.  That class will be
        added to the list of availabe implementations of this class.

        Arguments
        ---------
        implementation:
            The implementation to be used.  This can be a class
            or a class name.  In case of a class name it has to either be
            fully qualified (when no module is provided) or unqualified
            (if a module is provided).

        module:
            The module to which the implementation belongs. This
            has to be provided if the implementation is passed as
            unqualified class name.  Otherwise, if provided it has
            to be consistent with the class.

        Result
        ------
        the_implementation:
            The implementation of this loaded from the given module.

        Raises
        ------
        ImportError:
            The import failed. This can be due to different reasons,
            either not finding the required module, inconsistent argument
            values, or incorrect implementation (the loaded implementation
            unsuitable).
        """

        module_name: Optional[str]
        cls_name: str

        if isinstance(implementation, str):
            if '.' in implementation:
                module_name, cls_name = implementation.rsplit('.', maxsplit=1)
            else:
                module_name, cls_name = (None, implementation)
        else:
            module_name, cls_name = \
             implementation.__module__, implementation.__name__

        if module_name is None and module is None:
            raise ImportError("No module provided for loading implementation "
                              f"{implementation} of {cls}")

        if module is None:
            module = module_name

        if isinstance(module, str):
            if module_name is not None and module_name != module:
                raise ImportError("Module name mismatch for implementation "
                                  f"{implementation} of {cls}: is {module}"
                                  " but should be {module_name}.")
            module = importlib.import_module(module)
        elif (isinstance(module, ModuleType) and
              module_name is not None and module_name != module.__name__):
            raise ImportError("Inconsistent module provided for "
                              f"implementation {implementation} of {cls}: "
                              f"should be {module_name} "
                              f"but got {module.__name__}.")

        # get the class from the module
        new_cls = getattr(module, cls_name)

        # check if the class is suitable as an implementation
        if not issubclass(new_cls, cls) or new_cls is cls:
            LOG.error("Bad implementation '%s' for '%s.%s'",
                      implementation, cls.__module__, cls.__name__)
            raise ImportError(f"{Implementable._fully_qualified_name(new_cls)}"
                              " is not a proper implementation of "
                              f"Implementable._fully_qualified_name(cls)")

        LOG.info("Loaded implementation '%s.%s' for '%s.%s'.",
                 new_cls.__module__, new_cls.__name__,
                 cls.__module__, cls.__name__)

        return new_cls

    @classmethod
    def get_implementation(cls, module: Optional[Moduleslike] = None,
                           implementation: Optional[Implementationlike] = None,
                           ) -> Implementation:
        """Obtain an implementation of this class.

        Arguments
        ---------
        implementation:
            Either an implementation (a subclass of this class that serves
            as an implementation), or the name of such a class.  The name
            should be fully qualified except when a single module is provided
            in which case the name will be taken from that module.
        module:
            A module or a list of modules, from which the implementation
            could be obtained.  Implementations already loaded will be
            preferred over implementations that still have to be loaded.
            If multiple modules are provided, the order of the list
            list

        Result
        ------
        implementaton:
            The implementation.

        Raises
        ------
        NotImplementedError:
            No implementation could be obtained.
        """
        # Obtain module_names from the 'module' parameter
        if module is None:
            module_names = None
        else:
            if isinstance(module, str) or not isinstance(module, Sequence):
                module = (module, )
            module_names = \
                tuple(cls._canonical_module_name(mod) for mod in module)

        # If a implementation is provided, check that it matches
        # the module constratin and if so return it
        if implementation is not None:
            if not cls._from_modules(implementation, module_names):
                raise ValueError(f"Class '{implementation}' does not fit "
                                 f"the module constraint {module}.")

            if isinstance(implementation, type):
                return implementation

            # implementation is a class name -> load it
            # we need no or a single module specification
            if module_names is not None and len(module_names) > 1:
                raise ValueError(f"Incompatible module specification {module} "
                                 f"for implementation '{implementation}'")
            if module is not None:
                module = module[0]

            try:
                return cls.load_implementation(implementation=implementation,
                                               module=module)
            except ImportError:
                raise NotImplementedError(f"Implementatation {implementation} "
                                          "could not be imported")

        # Check if we already have loaded an implementation
        # and if so return it.
        try:
            return next(_ for _ in cls.implementations(loaded=True)
                        if cls._from_modules(_, module_names))
        except StopIteration:
            pass  # no suitable implementation was found

        # Try to load one of the registered implementation candidates
        # and return it.
        for candidate in (_ for _ in cls.implementations(loaded=False)
                          if cls._from_modules(_, module_names)):
            # candidate is a fully qualified class name
            try:
                return cls.load_implementation(implementation=candidate)
            except ImportError as ex:
                LOG.warning("Implementation '%s' for '%s.%s' failed: %s",
                            implementation, cls.__module__, cls.__name__, ex)

        # We have not found an implementation
        # -> raise an exception
        raise NotImplementedError("No implementation for class "
                                  f"{Implementable._fully_qualified_name(cls)}"
                                  " could be loaded "
                                  f"with module constraints {module_names}.")

    @staticmethod
    def _fully_qualified_name(implementation: Implementation):
        """Fully qualified name of a class.
        """
        return f"{implementation.__module__}.{implementation.__name__}"

    @staticmethod
    def _canonical_module_name(module: Modulelike) -> str:
        """Transform a `Modulelike` into a fully qualified module name.
        """
        return (module.__name__ if isinstance(module, ModuleType) else
                Implementable._module_aliases.get(module, module))

    @staticmethod
    def _from_modules(implementation: Implementationlike,
                      module_names: Optional[Sequence[str]] = None) -> bool:
        """Check if an implementation (`Implementationlike`) belongs to a
        set of modules, provided by a list of module names.
        """
        if module_names is None:
            return True  # no module restrictions

        # obtain the module name for the implementation
        name = (implementation.__module__
                if isinstance(implementation, type) else
                implementation.rsplit('.', maxsplit=1)[0])

        return name in module_names
