# standard imports
import sys
import importlib

# toolbox imports
from ..base.busy import BusyObservable, busy


class Importer(BusyObservable):
    """A class for doing (asynchronous) imports.
    """

    @busy("importing")
    def import_module(self, name: str) -> None:
        importlib.import_module(name)

    @busy("importing")
    def import_class(self, name: str) -> None:
        module, cls = name.rsplit('.', maxsplit=1)
        importlib.import_module(module)

    @staticmethod
    def module_is_imported(name: str) -> bool:
        """
        """
        return name in sys.modules

    @staticmethod
    def class_is_imported(name: str) -> bool:
        module, cls = name.rsplit('.', maxsplit=1)
        return module in sys.modules and hasattr(sys.modules[module], cls)

    @staticmethod
    def imported_module(name: str) -> type:
        if name not in sys.modules:
            raise ImportError(f"Module {name} has not been imported.")
        return sys.modules[name]

    @staticmethod
    def imported_class(name: str) -> type:
        module, cls = name.rsplit('.', maxsplit=1)
        module = Importer.imported_module(module)
        if not hasattr(module, cls):
            raise ImportError(f"Module {module} has no class named '{cls}'.")
        return getattr(module, cls)
