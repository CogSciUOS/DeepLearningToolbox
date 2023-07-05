"""Utilities to adapt the import process.
"""

# standard imports
from typing import Union, Callable, Optional, Iterable, Any
from typing import Tuple, List, Dict, Set
from types import ModuleType
import sys
import logging
import importlib
import importlib.abc  # explicit import required for python >= 3.8
from importlib.machinery import ModuleSpec
from importlib import import_module
from pathlib import Path
from contextlib import contextmanager
from itertools import chain

# toolbox imports
from ..base.busy import BusyObservable, busy
from .debug import debug_stack

# logging
LOG = logging.getLogger(__name__)


# FIXME[bug]: the current implementation of the preimport_hooks,
# preimport_dependcies and aliases will run the hook at the first
# invocation of importlib.util.find_spec.  This will occur when
# importing a module but it may also be used to just check if a module
# is available, without actually importing it.  It would be more
# accurate to invoce preimport_hook, ... only if the module is actually
# actually imported!
#
# The importable() function mitigates this problem by caching results
# obtaind from find_spec(), but direct invocations of find_spec() will
# still trigger the hooks
#
# Example: importlib.util.find_spec('keras') currently loads the
# tensorflow.keras module as 'keras', triggered by the pre-import hook
# on the 'keras' module!


class Importer(BusyObservable):
    """A class for doing (asynchronous) imports.
    """

    @staticmethod
    def importable(name: str,
                   directory: Union[str, Path, None] = None) -> bool:
        """Check if the given module can be imported.
        """
        return importlib.util.find_spec(name) is not None

    @busy("importing")
    @staticmethod
    def import_module(name: str,
                      directory: Union[str, Path, None] = None) -> None:
        """Asynchronously imports a given module.

        Arguments
        ---------
        name:
            The fully qualified module name, with components separated
            by dots (`.`).
        directory:
            A directory from which the module is to be imported.
        """
        return import_module(name)

    @busy("importing")
    @staticmethod
    def import_class(name: str) -> None:
        """Asynchronously imports a module containing a given class.

        Arguments
        ---------
        name:
            The fully qualified class name, that is the fully qualified
            module name followed by dot (`.`) and the class name.
        """
        module, _cls = name.rsplit('.', maxsplit=1)
        import_module(module)

    @staticmethod
    def import_module_from(name: str,
                           directory: Union[str, Path]) -> ModuleType:
        """Synchronous imports a given module from some directory.

        Arguments
        ---------
        name:
            The fully qualified module name, with components separated
            by dots (`.`).
        directory:
            A directory from which the module is to be imported.

        Result
        ------
        module:
            The imported module
        """
        print(f"import_module_from({name}, {directory})")
        sys.path.insert(0, str(directory))
        try:
            # module_name, _cls_name = name.rsplit('.', maxsplit=1)
            module_name = name
            module = import_module(module_name)
        finally:
            sys.path.remove(str(directory))
        return module

    @staticmethod
    def import_module_class(module:
                            Optional[Union[ModuleType, str]] = None,
                            cls: Optional[Union[type, str]] = None) \
            -> Tuple[ModuleType, type]:
        """Ensure that module and class are imported.

        Arguments
        ---------
        module:
            Either an (already loaded) module or the name of the module.
            If `None`, the module will be derived from the class.
        cls:
            Either an (already loaded) class (type) or the name of a
            class, either relative or fully qualified.

        Result
        ------
        module:
            The imported module.
        cls:
            The class.
        """
        if isinstance(cls, type):
            return cls.__module__, cls

        # module_name = None
        if module is None:
            if cls is None:
                raise ValueError("Neither module nor class are given.")

            if not isinstance(cls, str):
                raise TypeError(f"Invalid type for class: {type(cls)}")

            if '.' not in cls:
                raise ValueError("Either Fully qualified class name expected")

            module, cls = cls.rsplit('.', maxsplit=1)
        elif isinstance(cls, str):
            if '.' in cls:
                _module_name, cls = cls.rsplit('.', maxsplit=1)
        else:
            raise TypeError(f"Invalid type for class: {type(cls)}")

        # now cls is str and module is not None

        if isinstance(module, str):
            module = import_module(module)
        elif not isinstance(module, ModuleType):
            raise TypeError(f"Invalid type for module: {type(module)}")

        return module, getattr(module, cls)

    @staticmethod
    def module_is_imported(name: str) -> bool:
        """Check wether a given module has already been imported.

        Arguments
        ---------
        name:
            The fully qualified module name, with components separated
            by dots (`.`).

        Result
        ------
        imported:
            `True` if the module was imported, otherwise `False`.
        """
        return name in sys.modules

    @staticmethod
    def class_is_imported(name: str) -> bool:
        """Checks wether a given class has already been imported.

        Arguments
        ---------
        name:
            The fully qualified class name, that is the fully qualified
            module name followed by dot (`.`) and the class name.

        Result
        ------
        imported:
            `True` if the module was imported and contains the class,
            otherwise `False`.
        """
        module, cls = name.rsplit('.', maxsplit=1)
        return module in sys.modules and hasattr(sys.modules[module], cls)

    @staticmethod
    def imported_module(name: str) -> type:
        """Get a module object for a fully qualified module name.

        Arguments
        ---------
        name:
            The fully qualified module name, with components separated
            by dots (`.`).

        Raises
        ------
        ImportError
            The module has not been imported yet.
        """
        if name not in sys.modules:
            raise ImportError(f"Module {name} has not been imported.")
        return sys.modules[name]

    @staticmethod
    def imported_class(name: str) -> type:
        """Get a class object from a fully qualified class name.

        Arguments
        ---------
        name:
            The fully qualified class name, that is the fully qualified
            module name followed by dot (`.`) and the class name.

        Result
        ------
        cls:
            The class object representing that class.

        Raises
        ------
        ImportError
            The module has not been imported yet or does not contain a
            class with the given name.
        """
        module, cls = name.rsplit('.', maxsplit=1)
        module = Importer.imported_module(module)
        if not hasattr(module, cls):
            raise ImportError(f"Module {module} has no class named '{cls}'.")
        return getattr(module, cls)

    @staticmethod
    def __call__(name: str, package: str = None) -> ModuleType:
        """
        """
        return import_module(name, package)


class Importable:
    """An auxiliary class to check if modules can be imported.

    This class allows to write things like `'module_name' in Importable`
    or `Importable('module_name')`

    The class also realizes a caching mechanism to avoid looking up
    the package multiple times.
    """

    _importable_hash: Dict[str, bool] = {}

    @classmethod
    def __contains__(cls, name: str) -> bool:
        """Checks if the module with the given name can be imported.
        """
        return cls(name)

    @classmethod
    def __call__(cls, name: str, path: Optional[str] = None) -> bool:
        """Checks if the module with the given name can be imported.
        """
        try:
            # if we already know the answer, return it immediately
            return cls._importable_hash[name]
        except KeyError:
            # It may happen that a module is loaded (in sys.modules), but
            # does not provide a __spec__ attribute, or that this
            # attribute is None. I these cases, importlib.util.find_spec
            # raises a ValueError. Hence we check beforhand, if the module
            # is loaded, which is sufficient for us (we do not need the
            # module spec) und only refer to importlib in case it is not
            # loaded.
            if name in sys.modules:
                result = sys.modules[name] is not None
            else:
                result = importlib.util.find_spec(name, path) is not None

            # remember the result for better performance at future invocations
            cls._importable_hash[name] = result
            return result


# preimport_hook(fullname: str, path, target=Nonex)
PreimportHook = \
    Callable[[str, Optional[Iterable[str]], Optional[ModuleSpec]], None]

# preimport_hook(module: ModuleType)
PostimportHook = Callable[[ModuleType], None]


# Tracing the imported modules:
# - MetaPathFinder: goes first and can override builtin modules
# - PathEntryFinder: specifically for modules found on sys.path
class ImportInterceptor(importlib.abc.MetaPathFinder):
    """The purpose of the :py:class:`ImportInterceptor` is to adapt
    the import machinery. We want to have some influence on
    choosing what to import (e.g. tensorflow.keras instead of keras.io,
    or tensorflow.compat.v1 as tensorflow).

    In order to work, an instance of this class should be put
    into `sys.meta_path` before those modules are imported.

        if not hasattr(sys, 'frozen'):
            sys.meta_path = [ImportInterceptor()] + sys.meta_path

    """
    # _preimport_hooks and _postimport_hooks:
    #    This dictionary maps full names of modules to functions that
    #    should be called before the module was imported.  The
    #    preimport_hook functions are expected to take the same
    #    arguments as `find_spec`, the postimport_hook function should
    #    take the module as argument.
    _preimport_hooks: Dict[str, PreimportHook] = None
    _postimport_hooks: Dict[str, PostimportHook] = None

    # _preimport_dependencies and _postimport_dependencies:
    #    This dictionary maps full names of thirdparty modules (as
    #    stored in sys.modules) to (DLTB) modules to be imported after
    #    these modules have been imported.  The (DLTB) modules are given
    #    as list of arguments to be passed to importlib.import_module().
    _preimport_dependencies: Dict[str, List[str]] = None
    _postimport_dependencies: Dict[str, List[str]] = None

    # _aliases:
    #    A dictionary mapping strings to modules or module names.
    _aliases: Dict[str, Union[str, ModuleType]] = None

    # _current_imports:
    #    A set of currently ongoing imports - used to avoid infinite
    #    recursion in calling find_spec
    _current_imports: Set[str]
    
    ALL_MODULES = '*all*'
    THIRDPARTY_MODULES = '*thirdparty*'

    def __init__(self) -> None:
        super().__init__()

        self._preimport_dependencies = {}
        self._preimport_hooks = {}
        self._postimport_dependencies = {}
        self._postimport_hooks = {}
        self._aliases = {}
        self._current_imports = set()

        # Is the application started from source or is it frozen (bundled)?
        # The PyInstaller bootloader adds the name 'frozen' to the sys module:
        # A frozen python module is the compiled byte-code object,
        # incorporated into a custom build Python interpreter, using Python's
        # freeze utility.  This essentially is a binary executable that
        # can be run on machines without python interpreter.
        # https://wiki.python.org/moin/Freeze
        if hasattr(sys, 'frozen'):
            LOG.debug("sys is frozen")
        else:
            LOG.debug("sys is not frozen")
            sys.meta_path.insert(0, self)
            # sys.meta_path = [ImportInterceptor()] + sys.meta_path

    def __del__(self) -> None:
        if hasattr(sys, 'meta_path') and sys.meta_path is not None:
            sys.meta_path.remove(self)

    def find_spec(self, fullname: str, path: Optional[Iterable[str]],
                  target: Optional[ModuleType] = None) -> ModuleSpec:
        """Implementation of the PathFinder API.
        """

        if fullname in self._aliases:
            module = self._aliases[fullname]
            if not isinstance(module, ModuleType):
                module = importlib.import_module(module)
            sys.modules[fullname] = module
            del self._aliases[fullname]
            return module.__spec__

        is_thirdparty = ('.' not in fullname and
                         fullname not in sys.builtin_module_names)
        module_lookup = (fullname, self.ALL_MODULES)
        if is_thirdparty:
            module_lookup += (self.THIRDPARTY_MODULES, )

        #
        # Apply the pre-imports
        #
        if fullname in self._preimport_dependencies:
            LOG.info("ImportInterceptor['%s']: importing preimport "
                     "dependencies", fullname)
            for args in self._preimport_dependencies.pop(fullname):
                LOG.debug("ImportInterceptor['%s']: importing module %s.",
                          fullname, args)
                importlib.import_module(*args)

        if any(n in self._preimport_hooks for n in module_lookup):
            LOG.info("ImportInterceptor['%s']: applying preimport hooks",
                     fullname)
            for hook in chain(self._preimport_hooks.pop(fullname, ()),
                              *(self._preimport_hooks.get(n, ())
                                for n in module_lookup[1:])):
                LOG.debug("ImportInterceptor['%s']: running preiport hook %s "
                          "with path=%s, target=%s",
                         fullname, hook, path, target)
                hook(fullname, path, target)

        #
        # Prepare applying the post-imports
        # (the actuall action will take place in the Loader, after the
        # module has been loaded).
        #
        if (fullname not in self._current_imports and
            (fullname in self._postimport_dependencies or
             any(n in self._postimport_hooks for n in module_lookup))):
            # we have some postimport dependency or postimport hook of interest
            LOG.debug("ImportInterceptor['%s']: preparing post import.",
                      fullname)

            hooks = tuple(chain(self._postimport_hooks.pop(fullname, ()),
                                *(self._preimport_hooks.get(n, ())
                                  for n in module_lookup[1:])))
            dependencies = self._postimport_dependencies.pop(fullname, None)

            # use default path finder to get module loader (note that it
            # is important that we temporarily remove the fullname from
            # _postimport_dependencies to avoid infinite recursion)
            self._current_imports.add(fullname)
            module_spec = importlib.util.find_spec(fullname)
            self._current_imports.remove(fullname)

            if module_spec is not None:

                LOG.debug("ImportInterceptor['%s']: creating post import "
                          "LoaderWrapper with %d hooks and %d dependencies",
                          fullname, 0 if hooks is None else len(hooks),
                          0 if dependencies is None else len(dependencies))
                # Adapt the Loader of the module_spec
                module_spec.loader = ImportInterceptor.\
                    LoaderWrapper(module_spec.loader,
                                  postimport_hooks=hooks,
                                  postimport_dependencies=dependencies)
            return module_spec

        # None means: proceed with the standard procedure, using the next
        # MetaPathFinder in sys.meta_path ...
        return (sys.modules[fullname].__spec__
                if fullname in sys.modules else None)

    def add_preimport_hook(self, fullname: str,
                           action: PreimportHook) -> None:
        """Add a pre-import operation for a module.

        Arguments
        ---------
        action:
            A function to be called before the import is performed.
        fullname:
            The full name of the module for which the pre-import hook
            shall be called.
        """
        if fullname in sys.modules:
            LOG.warning("Module '%s' has already been imported. "
                        "Preimport hook will be ignored.", fullname)
        else:
            LOG.info("Registering preimport hook for module '%s'.", fullname)
            self._preimport_hooks.setdefault(fullname, []).append(action)

    def add_postimport_hook(self, fullname: str,
                            action: PostimportHook) -> None:
        """Add a post-import operation for a module.

        Arguments
        ---------
        fullname:
            The full name of the module for which a pre-import shall
            be performed.
        action:
            A function to be called after the import was performed
            successfully. The function is expected to take the imported
            module as its first and only argument.
        """
        if fullname in sys.modules:
            LOG.warning("Module '%s' has already been imported. "
                        "Will apply postimport hook now.", fullname)
            action()
        else:
            LOG.info("Registering postimport hook for module '%s'.", fullname)
            self._postimport_hooks.setdefault(fullname, []).append(action)

    def add_preimport_depency(self, fullname: str, args: Tuple) -> None:
        """Add a pre-import operation for a module.

        Arguments
        ---------
        fullname:
            The full name of the module for which a pre-import shall
            be performed.
        action:
            A function to be called when the import is to be performed.
        """
        if fullname in sys.modules:
            LOG.warning("Module '%s' has already been imported. "
                        "Preimport dependency will have no effect.", fullname)
        LOG.info("Registering preimport dependency %s for module '%s'.",
                 args, fullname)
        self._preimport_dependencies.setdefault(fullname, []).append(args)

    def add_postimport_depency(self, fullname: str, args: Tuple) -> None:
        """Add post import modules for a module.

        Arguments
        ---------
        fullname:
            The full name of the module for which a post import shall
            be performed.
        args:
            Arguments describing the additional module to be imported.
            These will be passed as arguments to `importlib.import_module`.
        """
        if fullname in sys.modules:
            # The module was already imported, so registering
            # postimport dependencies is useless.  Instead, we will
            # import that dependency right now, however, that may be
            # too late, as everything that has been done until know
            # was done without the postimport being in place.  Hence
            # inform the user that there may be a problem.
            LOG.warning("Module '%s' has already been imported. "
                        "Will import postimport dependency now.", fullname)
            importlib.import_module(*args)
        else:
            LOG.info("Registering postimport dependency %s for module '%s'.",
                     args, fullname)
            self._postimport_dependencies.setdefault(fullname, []).append(args)

    # FIXME[old]: seems not to be used
    def _pop_postimport_depency(self, fullname: str) -> Tuple:
        """Pop a post import. Removes the post import from this interceptor
        and returns the arguments associated with that import.

        Argments
        --------
        fullname:
            Full name of the module for which post imports are registered.

        Result
        ------
            The arguments that have been supplied when registering the
            post import with :py:meth:`add_postimport_depency`.
        """
        return self._postimport_dependencies.pop(fullname)

    def add_alias(self, alias: str, module: Union[str, ModuleType]) -> None:
        """Allow to import a module under a different name.  This
        should usually be avoided, as it may lead to inconsistencies
        (e.g. `module.__name__` will not match the module name) but
        it may help to quickly monkey patch the system.

        Arguments
        ---------
        alias:
            The name as which the module should be imported.
        module:
            The module or the name of the module to be imported instead.
        """
        self._aliases[alias] = module

    class LoaderWrapper(importlib.abc.Loader):
        """A wrapper around a :py:class:`importlib.abc.Loader`,
        perfoming additional imports after a module has been loaded.
        """
        def __init__(self, loader: importlib.abc.Loader,
                     postimport_hooks: Tuple[PostimportHook],
                     postimport_dependencies: Tuple[str]):
            self._loader = loader
            self._postimport_hooks = postimport_hooks
            self._postimport_dependencies = postimport_dependencies

        def create_module(self, spec) -> ModuleType:
            """A method that returns the module object to use when importing a
            module. This method may return None, indicating that
            default module creation semantics should take place.

            """
            LOG.debug("ImportInterceptor['%s']: performing create_module.",
                      spec.name)
            return self._loader.create_module(spec)

        def module_repr(self, module: ModuleType) -> str:
            """Obtain module representation as string.
            """
            return self._loader.module_repr(module)

        def exec_module(self, module):
            """An abstract method that executes the module in its own namespace
            when a module is imported or reloaded. The module should
            already be initialized when exec_module() is called.

            """
            module_name = module.__name__
            LOG.debug("ImportInterceptor['%s']: performing exec_module",
                      module_name)
            self._loader.exec_module(module)

            # Perform post-import hooks
            if self._postimport_hooks is not None:
                LOG.info("ImportInterceptor['%s']: applying postimport hoos",
                         module_name)
                for hook in self._postimport_hooks:
                    LOG.debug("ImportInterceptor['%s']: running hook %s",
                              module_name, hook)
                    hook(module)

            # Perform post-imports
            if self._postimport_dependencies is not None:
                LOG.info("ImportInterceptor['%s']: applying postimport "
                         "dependencies", module_name)
                for dependency in self._postimport_dependencies:
                    LOG.debug("ImportInterceptor['%s']: importing module %s",
                              module_name, dependency)
                    importlib.import_module(*dependency)


@contextmanager
def managed_resource(*_args, **_kwds):
    # FIXME[old/new]: seems not to be used anymore/yet?
    # What is this supposed to do?
    # -> there is a reference to this function in a comment
    #    in dltb/base/resource.py
    # see for example:
    #   https://docs.python.org/3/library/contextlib.html

    # Code to acquire resource, e.g.:
    modules_before = set(sys.modules)
    try:
        yield None
    finally:
        modules_after = set(sys.modules)
        for name in modules_after - modules_before:
            if name.startswith('dltb.'):
                pass
            elif name.startswith('numpy.'):
                pass
            elif name in sys.builtin_module_names:
                pass
            else:
                print(name)

importer = Importer()
importable = Importable()
import_interceptor = ImportInterceptor()

add_preimport_depency = import_interceptor.add_preimport_depency
add_preimport_hook = import_interceptor.add_preimport_hook
add_postimport_depency = import_interceptor.add_postimport_depency
add_postimport_hook = import_interceptor.add_postimport_hook

def debug_import(module: str) -> None:
    """Output debug information allowing to trace when a module
    is imported.
    """
    import_interceptor.add_preimport_hook(lambda f, p, t: debug_stack(),
                                          module)


def import_from_module(module: Union[ModuleType, str],
                       names: Union[str, Iterable[str]],
                       package: Optional[str] = None) -> Any:
    """This is essentially an equivalent to `from module import names`.
    """
    if isinstance(module, str):
        module = import_module(module, package)

    if isinstance(names, str):
        return getattr(module, names)

    return tuple(getattr(module, name) for name in names)
