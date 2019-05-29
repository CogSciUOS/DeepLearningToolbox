"""Central module for managing add ons, i.e. additional modules,
models, datasets, etc. that can be used by the toolbox, but are
not part of the toolbox.

"""

import importlib

# type: the type of resource
#   module: a python module (with the same name)
#   tool: a tool of the toolbox
#   data: a dataset
#   model: a (pretrained) model
# info: a human readable description of the resource
_infos = {
    'lucid': {
        'type': 'module',
        'info': 'A collection of infrastructure and tools '
                'for research in neural network interpretability.'
    },
    'autoencoder': {
        'type': 'tool',
        'info': 'A tool for inspecting autoencoders',
        'use': False
    },
    'advexample': {
        'type': 'tool',
        'info': 'A tool for inspecting adversarial examples',
        'use': False
    },
    'imutils': {
        'type': 'module',
        'info': 'A series of convenience functions to make '
                'basic image processing functions such as translation, '
                'rotation, resizing, skeletonization, displaying '
                'Matplotlib images, sorting contours, detecting edges, '
                'and much more easier with OpenCV and both '
                'Python 2.7 and Python 3.'
    },
    'dlib': {
        'type': 'module',
        'info': 'Dlib is a modern C++ toolkit containing '
                'machine learning algorithms and tools '
                'for creating complex software to solve real world problems.'
    },
    'imagenet': {
        'type': 'data',
        'info': 'The ILSVRC2012 dataset. A image classification dataset '
                'with 1K classes and more than 1M labeled training images.'
    },
    'alexnet': {
        'type': 'model',
        'info': 'A classical deep network model for image classification.'
    },
    'ikkuna': {
        'type': 'module',
        'info': 'A tool for monitoring neural network training.'
    }
}


def available(name, type=None):
    """Check if an add on is available, i.e. installed on the system.
    """
    if type is None:
        type = _infos[name]['type']
    if type == 'module':
        spec = importlib.util.find_spec(name)
        return spec is not None
    if type == 'tool':
        return True
    else:
        return False

def use(name, set=None, type=None):
    """Check if an add on should be used.

    This is not used yet, but it may be used for more fine grained
    customization of what resources should be at a given run of the
    toolbox, allowing to deactivate resource intense add ons that are
    not needed for the task at hand.
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
    
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        addons.use(self.dest, True)
        setattr(namespace, self.dest, values)
