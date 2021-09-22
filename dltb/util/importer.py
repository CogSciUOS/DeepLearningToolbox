"""Utilities to adapt the import process.
"""

# standard imports
from types import ModuleType
from typing import Union, Optional
import sys
import logging
import importlib
from pathlib import Path

# toolbox imports
from ..base.busy import BusyObservable, busy

# logging
LOG = logging.getLogger(__name__)


class Importer(BusyObservable):
    """A class for doing (asynchronous) imports.
    """

    @busy("importing")
    def import_module(self, name: str,
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
        return importlib.import_module(name)

    @busy("importing")
    def import_class(self, name: str) -> None:
        """Asynchronously imports a module containing a given class.

        Arguments
        ---------
        name:
            The fully qualified class name, that is the fully qualified
            module name followed by dot (`.`) and the class name.
        """
        module, _cls = name.rsplit('.', maxsplit=1)
        importlib.import_module(module)

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
            module = importlib.import_module(module_name)
        finally:
            sys.path.remove(str(directory))
        return module

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
