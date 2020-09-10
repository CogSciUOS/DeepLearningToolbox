""".. moduleauthor:: Ulf Krumnack

.. module:: dltb.thirdparty

This package provides utility functions for checking dealing with
third party libraries, ranging from an interface to check availability
and version matching, over package and data installation to abstract
interfaces that allow to access common functionionality (like images,
sound, video) using different libraries.

"""

# standard imports
from typing import Union, List, Iterator, Any
import sys
import logging
import importlib

# toolbox imports
from ..config import config 

# logging
LOG = logging.getLogger(__name__)


_modules = {
    'imageio': {
        'modules': ['imageio', 'imageio_ffmpeg', 'numpy'],
        'classes': {
            'ImageReader': 'ImageIO',
            'ImageWriter': 'ImageIO',
            'VideoReader': 'VideoFileReader',
            'Webcam': 'Webcam'
        },
    },
    'opencv': {
        'modules': ['cv2', 'numpy'],
        'classes': {
            'ImageReader': 'ImageIO',
            'ImageWriter': 'ImageIO',
            'ImageDisplay': 'ImageIO',
            'ImageResizer': 'ImageUtils',
            'VideoReader': 'VideoFileReader',
            'Webcam': 'Webcam'
        }
    },
    'tensorflow': {
        'modules': ['tensorflow', 'numpy']
    },
    'keras': {
        # FIXME[todo]: allow alternatives: 'tensorflow' or 'keras'
        'modules': ['tensorflow'],
        'config': {
            'preference': ['tensorflow', 'keras']
        }
    },
    'matplotlib': {
        'modules': ['matplotlib'],
        'classes': {
            'ImageReader': 'ImageIO',
            'ImageWriter': 'ImageIO',
            'ImageDisplay': 'ImageIO'
        }
    },
    'skimage': {
        'modules': ['skimage'],
        'classes': {
            'ImageResizer': 'ImageUtil'
        }
    },
    'qt': {
        'modules': ['PyQt5'],
        'classes': {
            'ImageDisplay': 'ImageDisplay'
        }
    }
}


def modules() -> Iterator[str]:
    """An iterator for names of known thirdparty modules.
    """
    return _modules.keys()


def classes() -> Iterator[str]:
    classnames = set()
    for description in _modules.values():
        if 'classes' in description:
            classnames |= description['classes'].keys()
    return iter(classnames)


def module_provides_class(module: str, name: str) -> bool:
    if module not in _modules:
        raise KeyError(f"Unknown module name: {module}")
    description = _modules[module]
    if 'classes' not in description:
        return False
    return name in description['classes']


def modules_with_class(name: str) -> Iterator[str]:
    for module in _modules:
        if module_provides_class(module, name):
            yield module


def available(module: str) -> bool:
    """Check if a third party module is available.

    Arguments
    ---------
    module: str
        The name of the thirdparty module
    """
    # Check if we know that module
    if module not in _modules:
        raise ValueError(f"Unsupported third party module '{module}'. "
                         f"Valid values are: " +
                         ', '.join(f"'{name}'" for name in _modules))
    ok = True

    # Check if the module was already imported
    if __name__ + '.' + module in sys.modules:
        return True  # module already loaded

    # Check if required third party modules are installed
    description = _modules[module]
    if 'modules' in description:
        for name in description['modules']:
            if importlib.util.find_spec(name) is None:
                ok = False
                break
    return ok


def import_class(name: str, module: Union[str, List[str]] = None):
    """Import a class.

    Arguments
    ---------
    name: str
        The name of the class to import.

    module:
        A single module or a list of modules. If no value is provided,
        all modules providing this class will be considered.
    """

    if isinstance(module, str):
        if not available(module):
            raise ImportError(f"Requested third party module '{module}' "
                              "is not available")
        module_name = module
    else:
        modules = list(modules_with_class(name)) if module is None else module
        module_name = None
        for candidate in modules:
            if __name__ + '.' + candidate in sys.modules:
                module_name = candidate
                break
            elif available(candidate) and module is None:
                module_name = candidate
        if module_name is None:
            raise ImportError(f"No third party module providing '{name}' "
                              f"is available. Checked: {modules}")

    # check if module was already imported:
    module_full_name = f'{__name__}.{module_name}'
    if module_full_name in sys.modules:
        module = sys.modules[module_full_name]
    else:
        if not available(module_name):
            raise ImportError(f"Third party module '{module_name}' "
                              "is not available.")
        module = importlib.import_module(module_full_name)

    # get class from module
    class_name = _modules[module_name]['classes'][name]
    if not hasattr(module, class_name):
        raise AttributeError(f"Third party module '{module.__name__}' "
                             f"has no attribute '{class_name}'.")
    return getattr(module, class_name)


def _check_module_config(module: str, attribute: str, value) -> None:
    if module not in _modules:
        raise ValueError(f"Unsupported third party module '{module}'. "
                         f"Valid values are: " +
                         ', '.join(f"'{name}'" for name in _modules))
    description = _modules[module]
    if 'config' not in description:
        raise ValueError(f"Third party module '{module}' "
                         "can not be configured.")

    config = description['config']
    if attribute not in config:
        raise ValueError(f"Unknown configuration attribute '{attribute}' "
                         f"for third party module '{module}'.")


def configure_module(module: str, attribute: str, value) -> None:
    _check_module_config(module, attribute)
    _modules[module]['config'][attribute] = value


def module_configuration(module: str, attribute: str) -> Any:
    _check_module_config(module, attribute)
    return _modules[module]['config'][attribute]


class ImportInterceptor(importlib.abc.MetaPathFinder):
    """The purpose of the :py:class:`ImportInterceptor` is to adapt
    the import machinery. We want to have some influence on
    choosing what to import (e.g. tensorflow.keras instead of keras.io,
    or tensorflow.compat.v1 as tensorflow).

    In order to work, an instance of this class should be put
    into `sys.sys.meta_path` before those modules are imported.
    """

    def find_spec(self, fullname, path, target=None):
        # keras: we want to use 'tensorflow.keras' instead of keras,
        # when available.
        if fullname == 'keras':
            LOG.debug("ImportInterceptor['%s']: path=%s, target=%s",
                      fullname, path, target)
            
            keras = None
            if 'tensorflow.keras' in sys.modules:
                keras = sys.modules['tensorflow.keras']
            elif 'tensorflow' in sys.modules:
                keras = sys.modules['tensorflow'].keras
            else:
                module_spec = importlib.util.find_spec('tensorflow.keras')
                if module_spec is not None:
                    # Load the module from module_spec.
                    # Remark: This actually seems not be necessary,
                    # as for some reason find_spec() already puts
                    # the module in sys.modules.
                    keras = importlib.util.module_from_spec(module_spec)
                    module_spec.loader.exec_module(keras)

            if keras is not None:
                LOG.info("Mapping 'keras' -> 'tensorflow.keras'")
                sys.modules['keras'] = keras
            else:
                LOG.info("Not mapping 'keras' -> 'tensorflow.keras'")

        return None  # Proceed with the standard procedure ...


# Is the application started from source or is it frozen (bundled)?
# The PyInstaller bootloader adds the name 'frozen' to the sys module:
# FIXME[todo]: explain what frozen modules are and implications
# they have for us!
if hasattr(sys, 'frozen'):
    LOG.info("sys is frozen")
else:
    LOG.info("sys is not frozen")
    sys.meta_path = [ImportInterceptor()] + sys.meta_path

    if 'keras' in sys.modules:
        LOG.warning("Module 'keras' was already import, hence "
                    "patching the import machinery will have no effect")

#
# Check for some optional packages
#

def warn_missing_dependencies():
    module_spec = importlib.util.find_spec('appdirs')
    if module_spec is None:
        LOG.warning(
            "--------------------------------------------------------------\n"
            "info: module 'appdirs' is not installed.\n"
            "We can live without it, but having it around will provide\n"
            "additional features.\n"
            "See: https://github.com/ActiveState/appdirs\n"
            "--------------------------------------------------------------\n")

    module_spec = importlib.util.find_spec('setproctitle')
    if module_spec is None:
        LOG.warning(
            "--------------------------------------------------------------\n"
            "info: module 'setproctitle' is not installed.\n"
            "We can live without it, but having it around will provide\n"
            "additional features.\n"
            "See: https://github.com/dvarrazzo/py-setproctitle\n"
            "--------------------------------------------------------------\n")


def list_modules():
    """List the state of registered third party modules.
    """
    LOG.warn("Status of thirdparty modules:")
    for name in modules():
        LOG.warn("module '%s': %s", name, available(name))


def list_classes():
    """List classes that are provide by thirdparty modules.
    """
    LOG.warn("Classes provided by third party modules:")
    for name in classes():
        LOG.warn("class '%s': %s", name, ", ".join(modules_with_class(name)))


if config.warn_missing_dependencies:
    warn_missing_dependencies()

if config.thirdparty_info:
    list_modules()
    list_classes()
