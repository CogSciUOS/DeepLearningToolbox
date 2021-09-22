"""Utilities to adapt the import process.
"""

# standard imports
from typing import Tuple, Dict, Callable, Union
from types import ModuleType
import os
import sys
import logging
import inspect
import importlib
import importlib.abc  # explicit import required for python >= 3.8

# logging
LOG = logging.getLogger(__name__)

# FIXME[bug]: the current implementation of the preimport_hooks,
# preimport_dependcies and aliases will run the hook at the first
# invocation of importlib.util.find_spec.  This will occur when
# importing a module but it may also be used to just check if a module
# is available, without actually importing it.  It would be more
# accurate to invoce preimport_hook, ... only if the module is actually
# actually imported!

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
    _preimport_hooks: Dict[str, Callable] = None
    _postimport_hooks: Dict[str, Callable] = None

    # _preimport_depencies and _postimport_depencies:
    #    This dictionary maps full names of thirdparty modules (as
    #    stored in sys.modules) to (DLTB) modules to be imported after
    #    these modules have been imported.  The (DLTB) modules are given
    #    as list of arguments to be passed to importlib.import_module().
    _preimport_depencies: Dict[str, Tuple] = None
    _postimport_depencies: Dict[str, Tuple] = None

    # _aliases:
    #    A dictionary mapping strings to modules or module names.
    _aliases: Dict[str, Union[str, ModuleType]] = None

    def __init__(self) -> None:
        super().__init__()

        self._preimport_depencies = dict()
        self._preimport_hooks = dict()
        self._postimport_depencies = dict()
        self._postimport_hooks = dict()
        self._aliases = dict()

        # Is the application started from source or is it frozen (bundled)?
        # The PyInstaller bootloader adds the name 'frozen' to the sys module:
        # A frozen python module is the compiled byte-code object,
        # incorporated into a custom build Python interpreter, using Python's
        # freeze utility.  This essentially is a binary executable that
        # can be run on machines without python interpreter.
        # https://wiki.python.org/moin/Freeze
        if hasattr(sys, 'frozen'):
            LOG.info("sys is frozen")
        else:
            LOG.info("sys is not frozen")
            sys.meta_path.insert(0, self)
            # sys.meta_path = [ImportInterceptor()] + sys.meta_path

    def __del__(self) -> None:
        if hasattr(sys, 'meta_path') and sys.meta_path is not None:
            sys.meta_path.remove(self)

    if 'keras' in sys.modules:
        LOG.warning("Module 'keras' was already import, hence "
                    "patching the import machinery will have no effect")

    def find_spec(self, fullname, path, target=None):
        """Implementation of the PathFinder API.
        """

        if fullname in self._aliases:
            module = self._aliases[fullname]
            if not isinstance(module, ModuleType):
                module = importlib.import_module(module)
            sys.modules[fullname] = module
            del self._aliases[fullname]
            return module.__spec__

        #
        # Apply the pre-imports
        #
        if fullname in self._preimport_depencies:
            args = self._preimport_depencies.pop(fullname)
            LOG.debug("ImportInterceptor['%s']: path=%s, target=%s. Args=%s",
                      fullname, path, target, args)
            importlib.import_module(*args)

        if fullname in self._preimport_hooks:
            LOG.debug("ImportInterceptor['%s']: path=%s, target=%s",
                      fullname, path, target)
            self._preimport_hooks.pop(fullname)(fullname, path, target)

        #
        # Prepare applying the post-imports
        # (the actuall action will take place in the Loader, after the
        # module has been loaded).
        #
        if (fullname in self._postimport_depencies or
                fullname in self._postimport_hooks):
            hook = self._postimport_hooks.pop(fullname, None)
            args = self._postimport_depencies.pop(fullname, None)

            LOG.info("Preparing post import hook for module '%s': %s",
                     fullname, args)

            # use default path finder to get module loader (note that it
            # is important that we temporarily remove the fullname from
            # _postimport_depencies to avoid infinite recursion)
            module_spec = importlib.util.find_spec(fullname)

            if module_spec is not None:

                # Adapt the Loader of the module_spec
                module_spec.loader = \
                    ImportInterceptor.LoaderWrapper(module_spec.loader,
                                                    postimport_hook=hook,
                                                    postimport_depency=args)
            return module_spec

        # None means: proceed with the standard procedure, using the next
        # MetaPathFinder in sys.meta_path ...
        return (sys.modules[fullname].__spec__
                if fullname in sys.modules else None)

    def add_preimport_hook(self, fullname: str, action: Callable) -> None:
        """Add a pre-import operation for a module.

        Arguments
        ---------
        fullname:
            The full name of the module for which a pre-import shall
            be performed.
        action:
            A function to be called before the import is performed.
        """
        self._preimport_hooks[fullname] = action

    def add_postimport_hook(self, fullname: str, action: Callable) -> None:
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
        self._postimport_hooks[fullname] = action

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
        self._preimport_depencies[fullname] = args

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
            importlib.import_module(*args)
        else:
            self._postimport_depencies[fullname] = args

    def pop_postimport_depency(self, fullname: str) -> Tuple:
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
        return self._postimport_depencies.pop(fullname)

    def add_alias(self, alias: str, module: Union[str, ModuleType]) -> None:
        """Allow to import a module under da different name.  This
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

    def debug_import(self, fullname: str) -> None:
        """Output debug information allowing to trace when a module
        is imported.
        """
        self.add_preimport_hook(fullname, self._debug_stack)

    @staticmethod
    def _debug_stack(fullname, path, target=None) -> None:
        stack = inspect.stack()
        # each line in the stack consists of the following six fields:
        # [0] <class 'frame'>:
        # [1] <class 'str'>: file name
        # [2] <class 'int'>: line number
        # [3] <class 'str'>: the function
        # [4] <class 'list'>: the lines
        # [3] <class 'int'>: ?
        cwd = os.getcwd()
        for idx, line in enumerate(stack):
            if line[4] and "import " in line[4][0]:
                file = line[1]
                if file.startswith(cwd):
                    file = '.' + file[len(cwd):]
                if file.startswith('.'):
                    print(f"[{idx}/{len(stack)}] {file}:{line[2]}: "
                          "{line[4][0]}", end='')
                # break
        print(f"-> find_module({fullname}, {path}, {target})")

    class LoaderWrapper(importlib.abc.Loader):
        """A wrapper around a :py:class:`importlib.abc.Loader`,
        perfoming additional imports after a module has been loaded.
        """
        def __init__(self, loader: importlib.abc.Loader,
                     postimport_hook, postimport_depency):
            LOG.debug("Creating post import LoaderWrapper")
            self._loader = loader
            self._postimport_hook = postimport_hook
            self._postimport_depency = postimport_depency

        def create_module(self, spec) -> ModuleType:
            """A method that returns the module object to use when importing a
            module. This method may return None, indicating that
            default module creation semantics should take place.

            """
            LOG.debug("Performing create_module for module for spec: %s", spec)
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
            LOG.debug("Performing exec_module for module '%s'", module_name)
            self._loader.exec_module(module)

            # Perform post-import hooks
            if self._postimport_hook is not None:
                self._postimport_hook(module)

            # Perform post-imports
            if self._postimport_depency is not None:
                LOG.debug("Performing post import for module '%s': %s",
                          module_name, self._postimport_depency)
                importlib.import_module(*self._postimport_depency)


import_interceptor = ImportInterceptor()
