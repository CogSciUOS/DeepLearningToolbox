"""Auxiliary module to provide uniform access to some Python type hints.
"""
__all__ = ['Protocol']

# standard imports
import sys

try:
    from typing import Protocol
except ImportError:
    try:
        from typing_extensions import Protocol
    except ImportError:
        print("You need at least Python 3.8 "
              "or the 'typing_extensions' package "
              "to run the Deep Learning Toolbox.",
              file=sys.stderr)
        # pip install typing_extensions
        # conda install -c anaconda typing_extensions
        # conda install -c conda-forge typing_extensions
        sys.exit(1)
