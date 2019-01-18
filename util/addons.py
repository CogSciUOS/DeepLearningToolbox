"""Central module for managing add ons, i.e. additional modules,
models, datasets, etc. that can be used by the toolbox, but are
not part of the toolbox.

"""

import importlib

_infos = {
    'lucid': {
        'type': 'module',
        'info': 'A collection of infrastructure and tools '
                'for research in neural network interpretability.'
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
    else:
        return False

def use(name):
    """Check if an add on should be used.

    This is not used yet, but it may be used for more fine grained
    customization of what resources should be at a given run of the
    toolbox, allowing to deactivate resource intense add ons that are
    not needed for the task at hand.
    """
    return available(name)


def install(name, force=False):
    if available(name):
        return

