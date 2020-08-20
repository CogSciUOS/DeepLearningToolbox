
# standard imports
from typing import Union, List, Iterator, Any
import sys
import importlib

_modules = {
    'imageio': {
        'modules': ['imageio', 'imageio_ffmpeg', 'numpy'],
        'classes': {
            'ImageReader': 'ImageIO',
            'ImageWriter': 'ImageIO',
            'VideoReader': 'VideoReader',
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
            'VideoReader': 'VideoReader',
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
