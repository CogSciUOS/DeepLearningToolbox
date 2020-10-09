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


_MODULES = {
    'imageio': {
        'modules': ['imageio', 'imageio_ffmpeg', 'numpy'],
        'classes': {
            'ImageReader': 'ImageIO',
            'ImageWriter': 'ImageIO',
            'VideoReader': 'VideoReader',
            'VideoWriter': 'VideoWriter',
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
    },
    'mtcnn': {
        'modules': ['mtcnn', 'tensorflow'],
        'classes': {
            'FaceDetector': 'DetectorMTCNN'
        }
    },
    'dlib': {
        'modules': ['dlib', 'imutils'],
        'classes': {
            'FaceDetector': 'DetectorCNN'
        }
    }
}


def modules() -> Iterator[str]:
    """An iterator for names of known thirdparty modules.
    """
    return _MODULES.keys()


def check_module_requirements(module: str) -> bool:
    """Check if the given module requires other modules, and
    if these modules can be found.
    """
    if module.startswith(__package__):
        if module == __package__:
            return True
        module = module[len(__package__) + 1:]
        if (module not in _MODULES or 'modules' not in _MODULES[module]):
            return True  # no requirements for that module
        for requirement in _MODULES[module]['modules']:
            if not importlib.util.find_spec(requirement):
                return False
    return bool(importlib.util.find_spec(module))


def classes() -> Iterator[str]:
    """Iterate class names for which implementations are provided by
    a third-party module.
    """
    classnames = set()
    for description in _MODULES.values():
        if 'classes' in description:
            classnames |= description['classes'].keys()
    return iter(classnames)


def module_provides_class(module: str, name: str) -> bool:
    """Check if a third-party module provides an implementation
    for a given class.

    Arguments
    ---------
    module: str
        Name of the third party module (as used in this toolbox, e.g.
        `opencv` for the OpenCV module).
    name: str
        The name of the abstract Deep Learning Toolbox class that
        should be implemented by the third party module.
    """
    if module not in _MODULES:
        raise KeyError(f"Unknown module name: {module}")
    description = _MODULES[module]
    if 'classes' not in description:
        return False
    return name in description['classes']


def modules_with_class(name: str) -> Iterator[str]:
    """Iterate names of all third party modules that provide
    an implementation of a given class.

    Arguments
    ---------
    name: str
        Name of a Deep Learning ToolBox class.
    """
    for module in _MODULES:
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
    if module not in _MODULES:
        return False
        # raise ValueError(f"Unsupported third party module '{module}'. "
        #                  f"Valid values are: " +
        #                  ', '.join(f"'{name}'" for name in _MODULES))
    found = True

    # Check if the module was already imported
    if __name__ + '.' + module in sys.modules:
        return True  # module already loaded

    # Check if required third party modules are installed
    description = _MODULES[module]
    if 'modules' in description:
        for name in description['modules']:
            if importlib.util.find_spec(name) is None:
                found = False
                break
    return found


def import_class(name: str, module: Union[str, List[str]] = None) -> type:
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
        module_list = (list(modules_with_class(name))
                       if module is None else module)
        module_name = None
        for candidate in module_list:
            if __name__ + '.' + candidate in sys.modules:
                module_name = candidate
                break
            if available(candidate) and module is None:
                module_name = candidate
        if module_name is None:
            raise ImportError(f"No third party module providing '{name}' "
                              f"is available. Checked: {module_list}")

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
    class_name = _MODULES[module_name]['classes'][name]
    if not hasattr(module, class_name):
        raise AttributeError(f"Third party module '{module.__name__}' "
                             f"has no attribute '{class_name}'.")
    return getattr(module, class_name)


def _check_module_config(module: str, attribute: str) -> None:
    """Check if the configuration for a third party module contains
    the given attribute.

    Raises
    ------
    ValueError:
        If the name is not a valid third party module, or if that
        module does not have the given attribute in its configuration.
    """
    if module not in _MODULES:
        raise ValueError(f"Unsupported third party module '{module}'. "
                         f"Valid values are: " +
                         ', '.join(f"'{name}'" for name in _MODULES))
    description = _MODULES[module]
    if 'config' not in description:
        raise ValueError(f"Third party module '{module}' "
                         "can not be configured.")

    module_config = description['config']
    if attribute not in module_config:
        raise ValueError(f"Unknown configuration attribute '{attribute}' "
                         f"for third party module '{module}'.")


def configure_module(module: str, attribute: str, value: Any) -> None:
    """Configure a third party module.

    Arguments
    ---------
    module: str
        Name of a third party module.
    attribute: str
        Name of a configuration attribute.
    value: Any
        The new value for the attribute.
    """
    _check_module_config(module, attribute)
    _MODULES[module]['config'][attribute] = value


def module_configuration(module: str, attribute: str) -> Any:
    """Get a configuration value for a third party module.

    Arguments
    ---------
    module: str
        Name of the third party module
    attribute: str
        Configuration attribute.
    """
    _check_module_config(module, attribute)
    return _MODULES[module]['config'][attribute]


class ImportInterceptor(importlib.abc.MetaPathFinder):
    """The purpose of the :py:class:`ImportInterceptor` is to adapt
    the import machinery. We want to have some influence on
    choosing what to import (e.g. tensorflow.keras instead of keras.io,
    or tensorflow.compat.v1 as tensorflow).

    In order to work, an instance of this class should be put
    into `sys.sys.meta_path` before those modules are imported.
    """

    patch_keras: bool = True

    _post_imports = {
        'PIL': ('.pil', __name__),
        'torchvision': ('.pil', __name__),
        'torch': ('.torch', __name__),
    }

    def find_spec(self, fullname, path, target=None):
        """Implementation of the PathFinder API.
        """
        # keras: we want to use 'tensorflow.keras' instead of keras,
        # when available.
        if fullname == 'keras' and self.patch_keras:
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

        if fullname in self._post_imports:
            args = self._post_imports.pop(fullname)
            LOG.info("Preparing post import for module '%s': %s",
                     fullname, args)
            module_spec = importlib.util.find_spec(fullname)
            module_spec.loader = \
                ImportInterceptor.LoaderWrapper(module_spec.loader, args)
            return module_spec

        # Proceed with the standard procedure ...
        return None

    class LoaderWrapper(importlib.abc.Loader):
        def __init__(self, loader, args):
            self._loader = loader
            self._args = args

        def create_module(self, spec):
            self._loader.create_module(spec)

        def exec_module(self, module):
            self._loader.exec_module(module)
            LOG.info("Performing post import for module '%s': %s",
                     module.__name__, self._args)
            importlib.import_module(*self._args)

#
# Post import hooks
#

# FIXME[hack]: check if there is a better way of doing this ...
# import builtins
# _builtin_import = builtins.__import__

# def _import_adapter(name, globals=None, locals=None, fromlist=(), level=0):
#    already_imported = name in sys.modules
#
#    module = _builtin_import(name, globals=globals, locals=locals,
#                             fromlist=fromlist, level=level)
#
#    if not already_imported:
#        # if name == 'PIL.Image' or name == 'torchvision':
#        #     importlib.import_module('.pil', __name__)
#        if name == 'torch':
#            importlib.import_module('.torch', __name__)
#    return module
#
# builtins.__import__ = _import_adapter


# Is the application started from source or is it frozen (bundled)?
# The PyInstaller bootloader adds the name 'frozen' to the sys module:
# FIXME[question]: explain what frozen modules are and implications
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
    """Emit warnings concerning missing dependencies, i.e. third party
    modules not install on this system.
    """
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
    LOG.warning("Status of thirdparty modules:")
    for name in modules():
        LOG.warning("module '%s': %s", name, available(name))


def list_classes():
    """List classes that are provide by thirdparty modules.
    """
    LOG.warning("Classes provided by third party modules:")
    for name in classes():
        LOG.warning("class '%s': %s", name,
                    ", ".join(modules_with_class(name)))


if config.warn_missing_dependencies:
    warn_missing_dependencies()

if config.thirdparty_info:
    list_modules()
    list_classes()
