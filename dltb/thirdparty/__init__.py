""".. moduleauthor:: Ulf Krumnack

.. module:: dltb.thirdparty

This package provides utility functions for checking dealing with
third-party libraries, ranging from an interface to check availability
and version matching, over package and data installation to abstract
interfaces that allow to access common functionionality (like images,
sound, video) using different libraries.

"""
# FIXME[old]: Some of the code below has been superseeded by the new
# module 'dltb.base.implementation'. Sort out, what could is still
# usefull and what can be removed.
#
# thirdparty.import_class(cls.__name__, module=module)


# standard imports
from typing import Union, List, Iterator, Any
import os
import sys
import logging
import importlib
import importlib.abc  # explicit import required for python >= 3.8

# toolbox imports
from ..config import config
from ..base.implementation import Implementable
from ..util.importer2 import import_interceptor

# logging
LOG = logging.getLogger(__name__)


_DLTB = 'dltb.'
_THIRDPARTY = 'dltb.thirdparty.'

Implementable.register_module_alias(_THIRDPARTY + 'imageio', 'imageio')
Implementable.register_implementation(_DLTB + 'base.image.ImageReader',
                                      _THIRDPARTY + 'imageio.ImageIO')
Implementable.register_implementation(_DLTB + 'base.image.ImageWriter',
                                      _THIRDPARTY + 'imageio.ImageIO')
Implementable.register_implementation(_DLTB + 'base.video.VideoReader',
                                      _THIRDPARTY + 'imageio.VideoReader')
Implementable.register_implementation(_DLTB + 'base.video.VideoWriter',
                                      _THIRDPARTY + 'imageio.VideoWriter')
Implementable.register_implementation(_DLTB + 'base.video.Webcam',
                                      _THIRDPARTY + 'imageio.Webcam')

Implementable.register_module_alias(_THIRDPARTY + 'opencv', 'cv2')
Implementable.register_module_alias(_THIRDPARTY + 'opencv', 'opencv')
Implementable.register_implementation(_DLTB + 'base.image.ImageReader',
                                      _THIRDPARTY + 'opencv.ImageIO')
Implementable.register_implementation(_DLTB + 'base.image.ImageWriter',
                                      _THIRDPARTY + 'opencv.ImageIO')
Implementable.register_implementation(_DLTB + 'base.image.ImageDisplay',
                                      _THIRDPARTY + 'opencv.ImageDisplay')
Implementable.register_implementation(_DLTB + 'base.image.ImageResizer',
                                      _THIRDPARTY + 'opencv.ImageUtils')
Implementable.register_implementation(_DLTB + 'base.image.ImageWarper',
                                      _THIRDPARTY + 'opencv.ImageUtils')
Implementable.register_implementation(_DLTB + 'base.video.VideoReader',
                                      _THIRDPARTY + 'opencv.VideoReader')
Implementable.register_implementation(_DLTB + 'base.video.VideoWriter',
                                      _THIRDPARTY + 'opencv.VideoWriter')
Implementable.register_implementation(_DLTB + 'base.video.Webcam',
                                      _THIRDPARTY + 'opencv.Webcam')
Implementable.register_implementation(_DLTB + 'tool.face.detector.Detector',
                                      _THIRDPARTY + 'opencv.face.DetectorHaar')
Implementable.register_implementation(_DLTB + 'tool.face.detector.Detector',
                                      _THIRDPARTY + 'opencv.face.DetectorSSD')

Implementable.register_module_alias(_THIRDPARTY + 'matplotlib', 'plt')
Implementable.register_module_alias(_THIRDPARTY + 'matplotlib', 'matplotlib')
Implementable.register_implementation(_DLTB + 'base.image.ImageReader',
                                      _THIRDPARTY + 'matplotlib.ImageIO')
Implementable.register_implementation(_DLTB + 'base.image.ImageWriter',
                                      _THIRDPARTY + 'matplotlib.ImageIO')
Implementable.register_implementation(_DLTB + 'base.image.Display',
                                      _THIRDPARTY + 'matplotlib.ImageDisplay')

Implementable.register_module_alias(_THIRDPARTY + 'skimage', 'skimage')
Implementable.register_module_alias(_THIRDPARTY + 'skimage', 'scikit-image')
Implementable.register_implementation(_DLTB + 'base.image.ImageResizer',
                                      _THIRDPARTY + 'skimage.ImageUtil')
Implementable.register_implementation(_DLTB + 'base.image.ImageWarper',
                                      _THIRDPARTY + 'skimage.ImageUtil')

Implementable.register_module_alias(_THIRDPARTY + 'qt', 'qt')
Implementable.register_implementation(_DLTB + 'base.image.ImageDisplay',
                                      _THIRDPARTY + 'qt.ImageDisplay')


Implementable.register_module_alias(_THIRDPARTY + 'arcface', 'arcface')
Implementable.register_implementation(_DLTB + 'tool.face.recognize.ArcFace',
                                      _THIRDPARTY + 'arcface.ArcFace')


Implementable.register_module_alias(_THIRDPARTY + 'mtcnn', 'mtcnn')
Implementable.register_module_alias(_THIRDPARTY + 'face_evolve.mtcnn',
                                    'mtcnn2')
Implementable.register_implementation(_DLTB + 'tool.face.mtcnn.Detector',
                                      _THIRDPARTY + 'mtcnn.Detector')
Implementable.register_implementation(_DLTB + 'tool.face.mtcnn.Detector',
                                      _THIRDPARTY +
                                      'face_evolve.mtcnn.Detector')


_BASE_CLASSES = {
    'ImageReader': _DLTB + 'base.image.ImageReader',
    'ImageWriter': _DLTB + 'base.image.ImageWriter',
    'ImageResizer': _DLTB + 'base.image.ImageResizer',
    'VideoReader': _DLTB + 'base.video.VideoReader',
    'VideoWriter': _DLTB + 'base.video.VideoWriter',
    'SoundReader': _DLTB + 'base.image.SoundReader',
    'SoundWriter': _DLTB + 'base.image.SoundWriter',
    'SoundPlayer': _DLTB + 'base.image.SoundPlayer',
    'SoundRecorder': _DLTB + 'base.image.SoundRecorder',
    'Webcam': _DLTB + 'base.video.Webcam',
    'FaceDetector': _DLTB + 'tool.face.detector.Detector',
    'ImageGAN': _DLTB + 'tool.gan.ImageGAN',
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
            'Webcam': 'Webcam',
            'FaceDetector': ['face.DetectorHaar', 'face.DetectorSSD']
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
    'mtcnn2': {
        'modules': ['torch'],
        'classes': {
            'FaceDetector': 'Detector'
        }
    },
    'dlib': {
        'modules': ['dlib', 'imutils'],
        'classes': {
            'FaceDetector': ['DetectorCNN', 'DetectorHOG']
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
    },
    'experiments': {
        'modules': ['torch'],
        'classes': {
            'ImageGAN': 'VGAN'
        }
    },
    'soundfile': {
        'modules': ['soundfile'],
        'classes': {
            'SoundReader': 'SoundReader',
            'SoundWriter': 'SoundWriter',
        }
    },
    'sounddevice': {
        'modules': ['soundfile'],
        'classes': {
            'SoundPlayer': 'SoundPlayer',
            'SoundRecorder': 'SoundRecorder',
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
        else:
            return True
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
    """Iterate all implementations of a given Deep Learning Toolbox
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

    if isinstance(base, str) and '.' not in base:
        # full_base_name = _BASE_CLASSES[base]
        short_base_name = base
    else:
        full_base_name = (base if isinstance(base, str) else
                          base.__module__ + '.' + base.__name__)
        try:
            short_base_name = \
                next(short for short, full in _BASE_CLASSES.items()
                     if full == full_base_name)
        except StopIteration:
            raise ValueError(f"No short name for '{full_base_name}'.")

    for module, description in _MODULES.items():
        if check_available and not available(module):
            continue
        try:
            classes = description['classes']
            implementations = classes[short_base_name]
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
        all modules providing this class will be considered. This
        argument will be ignored if `name` is a fully qualified class
        name.
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


# Should tensorflow.keras be preferred, even if Keras.IO is available?
prefer_tensorflow_keras = True


def patch_keras_import(fullname, path, target=None):
    """Patch the keras import process to use the TensorFlow implementation
    of Keras (`tensorflow.keras`) instead of the standard Keras.IO
    implementation (`keras`).

    """
    if (not prefer_tensorflow_keras and
            (importlib.util.find_spec('keras') is not None)):
        # The only way to configure the keras backend appears to be
        # via environment variable. We thus inject one for this
        # process. Keras must be loaded after this is done
        # os.environ['KERAS_BACKEND'] = 'theano'
        os.environ['KERAS_BACKEND'] = 'tensorflow'

        # Importing keras unconditionally outputs a message
        # "Using [...] backend." to sys.stderr (in keras/__init__.py).
        # There seems to be no sane way to avoid this.

        return  # nothing else to do ...

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


# FIXME[bug]: the current implementation of the preimport hokk imports
# 'tensorflow.keras' (as 'keras') at the first invocation of
# importlib.util.find_spec('keras').  It would be more appropriate to
# do the import only on loading the module, so that find_spec('keras')
# could be used to check if 'keras' is available.  Repairing this
# would probably need some adaptation of the ImportInterceptor class.
import_interceptor.add_preimport_hook('keras', patch_keras_import)

import_interceptor.add_postimport_depency('PIL', ('.pil', __name__))
import_interceptor.add_postimport_depency('torchvision', ('.pil', __name__))
import_interceptor.add_postimport_depency('torch', ('.torch', __name__))
import_interceptor.add_postimport_depency('sklearn', ('.sklearn', __name__))


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
