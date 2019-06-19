"""Central module for managing add ons, i.e. additional modules,
models, datasets, etc. that can be used by the toolbox, but are
not part of the toolbox.

"""

from typing import List

import argparse
import importlib
from enum import Enum

# Taken from https://stackoverflow.com/a/19330461
class AutoEnum(Enum):
    """
    Automatically numbers enum members starting from 1.

    Includes support for a custom docstring per member.

    """
    __last_number__ = 0

    def __new__(cls, *args):
        """Ignores arguments (will be handled in __init__."""
        value = cls.__last_number__ + 1
        cls.__last_number__ = value
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, *args):
        """Can handle 0 or 1 argument; more requires a custom __init__.

        0  = auto-number w/o docstring
        1  = auto-number w/ docstring
        2+ = needs custom __init__

        """
        if len(args) == 1 and isinstance(args[0], str):
            self.__doc__ = args[0]
        elif args:
            raise TypeError('%s not dealt with -- need custom __init__' % (args,))

class Types(AutoEnum):
    """The different types of add ons avaliable for the Toolbox.
    """
    module = "a python module (with the same name)"
    tool = "a tool of the toolbox"
    data = "a dataset" 
    model = "a (pretrained) model"

# type: Types
#     the type of resource
# info: str
#     a human readable description of the resource
_infos = {
    'numpy': {
        'type': Types.module,
        'info': 'NumPy is the fundamental package for scientific computing'
                ' with Python.'
    },
    'tensorflow': {
        'type': Types.module,
        'info': 'tensorflow'
    },
    'keras': {
        'type': Types.module,
        'info': 'keras'
    },
    'appsdir': {
        'type': Types.module,
        'info': 'appsdir'
    },
    'matplotlib': {
        'type': Types.module,
        'info': 'matplotlib'
    },
    'keras': {
        'type': Types.module,
        'info': 'keras'
    },
    'cv2': {
        'type': Types.module,
        'info': 'OpenCV'
    },
    'caffe': {
        'type': Types.module,
        'info': 'caffe'
    },
    'PyQt5': {
        'type': Types.module,
        'info': 'PyQt5'
    },
    'pycuda': {
        'type': Types.module,
        'info': 'pycuda'
        # conda install -c lukepfister pycuda
    },
    'lucid': {
        'type': Types.module,
        'info': 'A collection of infrastructure and tools '
                'for research in neural network interpretability.'
    },
    'autoencoder': {
        'type': Types.tool,
        'info': 'A tool for inspecting autoencoders',
        'use': False
    },
    'advexample': {
        'type': Types.tool,
        'info': 'A tool for inspecting adversarial examples',
        'use': False
    },
    'imutils': {
        'type': Types.module,
        'info': 'A series of convenience functions to make '
                'basic image processing functions such as translation, '
                'rotation, resizing, skeletonization, displaying '
                'Matplotlib images, sorting contours, detecting edges, '
                'and much more easier with OpenCV and both '
                'Python 2.7 and Python 3.'
    },
    'dlib': {
        'type': Types.module,
        'info': 'Dlib is a modern C++ toolkit containing '
                'machine learning algorithms and tools '
                'for creating complex software to solve real world problems.'
    },
    'imagenet': {
        'type': Types.data,
        'info': 'The ILSVRC2012 dataset. A image classification dataset '
                'with 1K classes and more than 1M labeled training images.'
    },
    'alexnet': {
        'type': Types.model,
        'info': 'A classical deep network model for image classification.'
    },
    'ikkuna': {
        'type': Types.module,
        'info': 'A tool for monitoring neural network training.'
    }
}

def get_addons(type: Types):
    return [name for name, info in _infos.items()
            if info['type'] == Types.module]
    

def available(name: str, type: Types=None):
    """Check if an add on is available, i.e. installed on the system.

    Parameter
    ---------
    name:
        The name of the add on. If type is not given, this should
        be a valid key in the _infos dictionary.
    type:
        The Type of the add on.
    """
    if type is None:
        type = _infos[name]['type']
    if type == Types.module:
        spec = importlib.util.find_spec(name)
        return spec is not None
    if type == Types.tool:
        return True
    else:
        return False

def use(name: str, set: bool=None, type: Types=None):
    """Check if an add on should be used.

    This is not used yet, but it may be used for more fine grained
    customization of what resources should be at a given run of the
    toolbox, allowing to deactivate resource intense add ons that are
    not needed for the task at hand.

    Parameter
    ---------
    name:
        The name of the add on. If type is not given, this should
        be a valid key in the _infos dictionary.
    set:
    type:
    """
    if type is None:
        type = _infos[name]['type']
    if set is not None:
        _infos[name]['use'] = set
        return    
    return available(name) and _infos[name].get('use', True)


def install(name, force=False):
    if available(name):
        return

import argparse

class UseAddon(argparse.Action):
    """Auxiliary class for parsing command line options.
    Turn on use of given addon.

    """
    
    def __init__(self, option_strings: List[str], dest: str,
                 nargs: int=None, **kwargs) -> None:
        """Initialize the :py:class:`UseAddon` instance. This is intended to
        be invoked automaitcally by the argparse.ArgumentParser, when
        an argument is added with action=UseAddon.

        Parameter
        ---------
        option_strings: List[str]
            The list of command line options that will trigger this Action.
        dest: str
            The name of the attribute to be added to the object returned
            by parse_args().
        nargs: int
            The number of additional command line arguments to be consumed.
        default:
            The value produced if the argument is absent from the command
            line.
        help:
            A brief description of what the argument does.
        """
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser: argparse.ArgumentParser,
                 namespace: argparse.Namespace,
                 values: List[str], option_string: str=None) -> None:
        """This method is expected to be invoked by the
        argparse.ArgumentParser's parse_args() method in reply to
        detecting one of the relevant option strings on the command
        line.

        In addition to setting the respective destination value in the
        namespace, this will use the actual resource (named by the
        dest attribute of this :py:class:`UseAddon`)

        Parameter
        ---------
        parser:
            The ArgumentParser calling this method.
        namespace:
            The namespace for storing values in reply to an command
            line option.
        values:
            The list of additional command line arguments provided as
            values for this Option
        option_string:
            The actual option string found on the command line.

        """
        use(self.dest, True)
        setattr(namespace, self.dest, values)
