"""Auxiliary module to provide uniform access to some Python type hints.
"""
__all__ = ['Protocol', 'get_origin', 'get_args']

# standard imports
import sys

try:
    from typing import Protocol, Literal
except ImportError:
    try:
        from typing_extensions import Protocol, Literal
    except ImportError:
        print("You need at least Python 3.8 "
              "or the 'typing_extensions' package "
              "to run the Deep Learning Toolbox.",
              file=sys.stderr)
        # pip install typing_extensions
        # conda install -c anaconda typing_extensions
        # conda install -c conda-forge typing_extensions
        sys.exit(1)

try:
    from typing import get_origin, get_args
except ImportError:
    try:
        from typing_inspect import get_origin, get_args
    except ImportError:
        print("You need at least Python 3.8 "
              "or the 'typing_inspect' package "
              "to run the Deep Learning Toolbox.",
              file=sys.stderr)
        # pip install typing_inspect
        # conda install -c anaconda typing_inspect
        # conda install -c conda-forge typing_inspect
        sys.exit(1)
