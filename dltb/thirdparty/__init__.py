""".. moduleauthor:: Ulf Krumnack

.. module:: dltb.thirdparty

This package provides utility functions for checking dealing with
third-party libraries, ranging from an interface to check availability
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


_DLTB = 'dltb'

_BASE_CLASSES = {
    'ImageReader': _DLTB + '.base.image.ImageReader',
    'ImageWriter': _DLTB + '.base.image.ImageWriter',
    'ImageResizer': _DLTB + '.base.image.ImageResizer',
    'VideoReader': _DLTB + '.base.video.VideoReader',
    'VideoWriter': _DLTB + '.base.video.VideoWriter',
    'Webcam': _DLTB + '.base.video.Webcam',
    'FaceDetector': _DLTB + '.tool.face.FaceDetector',
    'ImageGAN': _DLTB + '.tool.gan.ImageGAN',
}

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
            'ImageDisplay': 'ImageDisplay',
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
            'ImageDisplay': 'Display'
        }
    },
    'skimage': {
        'modules': ['skimage'],
        'classes': {
            'ImageResizer': 'ImageUtil'
        }
    },
    'sklearn': {
        'modules': ['sklearn'],
        'classes': {
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
    },
    'nvlabs': {
        'modules': ['tensorflow'],  # 'dnnlib'
        'classes': {
            'ImageGAN': ['StyleGAN', 'StyleGAN2']
        }
    },
    'nnabla': {
        'modules': ['nnabla'],  # 'nnabla_ext.cuda'
        'classes': {
            'ImageGAN': 'StyleGAN2'
        }
    }
}


def modules() -> Iterator[str]:
    """An iterator for names of known third-party modules.
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
        Name of the third-party module (as used in this toolbox, e.g.
        `opencv` for the OpenCV module).
    name: str
        The name of the abstract Deep Learning Toolbox class that
        should be implemented by the third-party module.
    """
    if module not in _MODULES:
        raise KeyError(f"Unknown module name: {module}")
    description = _MODULES[module]
    if 'classes' not in description:
        return False
    return name in description['classes']


def modules_with_class(cls: Union[type, str]) -> Iterator[str]:
    """Iterate names of all third-party modules that provide
    an implementation of a given class.

    Arguments
    ---------
    cls:
        Deep Learning ToolBox class or nome of such a class.
    """
    name = cls if isinstance(cls, str) else cls.__name__
    for module in _MODULES:
        if module_provides_class(module, name):
            yield module


def implementations(base: Union[str, type],
                    check_available: bool = False) -> Iterator[str]:
    """Iterate all implementations of a given deep learning toolbox
    base class.

    Arguments
    ---------
    base:
        The abstract base class, either its name or the type object.
    check_available:
        If `True`, only provide available implementations according
        to :py:func:`available`. If `False`, no availability check is
        perfomed, meaning that implementations may be available or
        unavailable.
    """
    def result(module, implementation):
        return __name__ + '.' + module + '.' + implementation

    base_name = base if isinstance(base, str) else base.__name__

    for module, description in _MODULES.items():
        if check_available and not available(module):
            continue
        try:
            classes = description['classes']
            implementations = classes[base_name]
            if isinstance(implementations, str):
                yield result(module, implementations)
                continue
            for implementation in implementations:
                yield result(module, implementation)
        except KeyError:
            continue


def available(module: str) -> bool:
    """Check if a third-party module is available. Availability in
    this context means, that all resources required for importing
    that module (usually some third-party packages) are available.

    Arguments
    ---------
    module: str
        The name of the thirdparty module, either absolute (with
        the `'dltb.thirdparty'` prefix) or relative to `dltb.thirdparty`.

    Result
    ------
    available:
        If a third-party module judged to be available, importing that
         module should not raise an exception.
    """
    if module.startswith(__name__):
        module = module[len(__name__) + 1:]

    # Check if we know that module
    if module not in _MODULES:
        return False
        # raise ValueError(f"Unsupported third-party module '{module}'. "
        #                  f"Valid values are: " +
        #                  ', '.join(f"'{name}'" for name in _MODULES))
    found = True

    # Check if the module was already imported
    if __name__ + '.' + module in sys.modules:
        return True  # module already loaded

    # Check if required third-party modules are installed
    description = _MODULES[module]
    if 'modules' in description:
        for name in description['modules']:
            if importlib.util.find_spec(name) is None:
                found = False
                break
    return found


def import_class(name: str, module: Union[str, List[str]] = None) -> type:
    """Import a class from a third-party module.  The class is specified
    as one of the abstract base classes of the deep learning toolbox,
    listed in the array `_BASE_CLASSES`.

    Arguments
    ---------
    name: str
        The name of the base class to import, or a fully qualified
        class name (including fully qualified module name).

    module:
        A single module or a list of modules. If no value is provided,
        all modules providing this class will be considered.
    """

    # FIXME[hack]: integrate with the rest of the function
    if name not in _BASE_CLASSES and '.' in name:
        module_full_name, class_name = name.rsplit('.', maxsplit=1)
        module = importlib.import_module(module_full_name)
        if not hasattr(module, class_name):
            raise AttributeError(f"Third-party module '{module.__name__}' "
                                 f"has no attribute '{class_name}'.")
        return getattr(module, class_name)

    # here starts the original implementation
    if isinstance(module, str):
        if not available(module):
            raise ImportError(f"Requested third-party module '{module}' "
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
            raise ImportError(f"No third-party module providing '{name}' "
                              f"is available. Checked: {module_list}")

    # check if module was already imported:
    module_full_name = f'{__name__}.{module_name}'
    if module_full_name in sys.modules:
        module = sys.modules[module_full_name]
    else:
        if not available(module_name):
            raise ImportError(f"Third-party module '{module_name}' "
                              "is not available.")
        module = importlib.import_module(module_full_name)

    # get class from module
    class_name = _MODULES[module_name]['classes'][name]
    if not isinstance(class_name, str):
        class_name = class_name[0]
    if not hasattr(module, class_name):
        raise AttributeError(f"Third-party module '{module.__name__}' "
                             f"has no attribute '{class_name}'.")
    return getattr(module, class_name)


def _check_module_config(module: str, attribute: str) -> None:
    """Check if the configuration for a third-party module contains
    the given attribute.

    Raises
    ------
    ValueError:
        If the name is not a valid third-party module, or if that
        module does not have the given attribute in its configuration.
    """
    if module not in _MODULES:
        raise ValueError(f"Unsupported third-party module '{module}'. "
                         f"Valid values are: " +
                         ', '.join(f"'{name}'" for name in _MODULES))
    description = _MODULES[module]
    if 'config' not in description:
        raise ValueError(f"Third-party module '{module}' "
                         "can not be configured.")

    module_config = description['config']
    if attribute not in module_config:
        raise ValueError(f"Unknown configuration attribute '{attribute}' "
                         f"for third-party module '{module}'.")


def configure_module(module: str, attribute: str, value: Any) -> None:
    """Configure a third-party module.

    Arguments
    ---------
    module: str
        Name of a third-party module.
    attribute: str
        Name of a configuration attribute.
    value: Any
        The new value for the attribute.
    """
    _check_module_config(module, attribute)
    _MODULES[module]['config'][attribute] = value


def module_configuration(module: str, attribute: str) -> Any:
    """Get a configuration value for a third-party module.

    Arguments
    ---------
    module: str
        Name of the third-party module
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
    into `sys.meta_path` before those modules are imported.
    """

    patch_keras: bool = True

    _post_imports = {
        'PIL': ('.pil', __name__),
        'torchvision': ('.pil', __name__),
        'torch': ('.torch', __name__),
        'sklearn': ('.sklearn', __name__)
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
            # use default path finder to get module loader (note that it
            # is important that we temporarily remove the fullname from
            # _post_imports to avoid infinite recursion)
            module_spec = importlib.util.find_spec(fullname)
            # now add the fullname again, as find_spec may be called
            # multiple times before the module actually gets loaded.
            self._post_imports[fullname] = args

            # Adapt the Loader of the module_spec
            module_spec.loader = \
                ImportInterceptor.LoaderWrapper(module_spec.loader, self)
            return module_spec

        # Proceed with the standard procedure ...
        return None

    def add_post_imports(self, fullname: str, args) -> None:
        """Add post import modules for a module.

        Arguments
        ---------
        fullname:
            The full name of the module for which a post import shall
            be performed.
        args:
            Arguments describing the additional module to be imported.
        """
        self._post_imports[fullname] = args

    def pop_post_imports(self, fullname: str):
        """Remove a post import.

        Argments
        --------
        fullname:
            Full name of the module for which post imports are registered.
        """
        return self._post_imports.pop(fullname)

    class LoaderWrapper(importlib.abc.Loader):
        """A wrapper around a :py:class:`importlib.abc.Loader`,
        perfoming additional imports after a module has been loaded.
        """
        def __init__(self, loader: importlib.abc.Loader, interceptor):
            LOG.debug("Creating post import LoaderWrapper")
            self._loader = loader
            self._interceptor = interceptor

        def create_module(self, spec):
            """A method that returns the module object to use when importing a
            module. This method may return None, indicating that
            default module creation semantics should take place.

            """
            LOG.debug("Performing create_module for module for spec: %s", spec)
            return self._loader.create_module(spec)

        def exec_module(self, module):
            """An abstract method that executes the module in its own namespace
            when a module is imported or reloaded. The module should
            already be initialized when exec_module() is called.

            """
            module_name = module.__name__
            LOG.debug("Performing exec_module for module '%s'", module_name)
            self._loader.exec_module(module)

            args = self._interceptor.pop_post_imports(module_name)
            LOG.debug("Performing post import for module '%s': %s",
                      module_name, args)
            importlib.import_module(*args)

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
    """Emit warnings concerning missing dependencies, i.e. third-party
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
    """List the state of registered third-party modules.
    """
    LOG.warning("Status of thirdparty modules:")
    for name in modules():
        LOG.warning("module '%s': %s", name, available(name))


def list_classes():
    """List classes that are provide by thirdparty modules.
    """
    LOG.warning("Classes provided by third-party modules:")
    for name in classes():
        LOG.warning("class '%s': %s", name,
                    ", ".join(modules_with_class(name)))


if config.warn_missing_dependencies:
    warn_missing_dependencies()

if config.thirdparty_info:
    list_modules()
    list_classes()
