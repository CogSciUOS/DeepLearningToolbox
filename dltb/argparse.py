"""Support for parsing of common Deep Learning Toolbox command line
options.

Intended usage:

```
from argparse import ArgumentParser
import dltb.argparse as ToolboxArgparse

# ...

parser = ArgumentParser(...)
# ... add specific arguments ...
ToolboxArgparse.add_arguments(parser)

args = parser.parse_args()
ToolboxArgparse.process_arguments(args)

# ...
```

"""

# standard imports
from argparse import ArgumentParser, Namespace
import sys
import logging
import importlib

# toolbox imports
from .config import config
from .util.logging import TerminalFormatter


def add_arguments(parser: ArgumentParser) -> None:
    """Add arguments to an :py:class:`ArgumentParser`, that
    allow to specify general options for the Deep Learning ToolBox.

    Parameters
    ----------
    parser: ArgumentParser
        The argument parser to which arguments are to be added.
    """
    group = parser.add_argument_group("General toolbox arguments")

    #
    # Debugging
    #
    group.add_argument('--info', default=[], action='append',
                       metavar='MODULE',
                       help='Show info messages from MODULE')
    group.add_argument('--debug', default=[], action='append',
                       metavar='MODULE',
                       help='Show debug messages from MODLE')

    #
    # Miscallenous
    #
    group.add_argument('--warn-missing-dependencies',
                       help="Issue warnings on missing software packages",
                       action='store_true', default=False)


def process_arguments(parser: ArgumentParser, args: Namespace = None) -> None:
    """Evaluate command line arguments for configuring the toolbox.

    Parameters
    ----------
    parser: ArgumentParser
        The argument parser (used for error proecessing).
    args: Namespace
        An `Namespace` from parsing the command line
        arguments with `parser.parse_args()`.
    """
    if args is None:
        args = parser.parse_args()

    if args.warn_missing_dependencies:
        config.warn_missing_dependencies = True

    #
    # Debugging
    #
    handler = None
    for what in ('info', 'debug'):
        modules = getattr(args, what)
        if not modules:
            continue  # no modules provided as 'info/debug' command line args

        if handler is None:
            handler = logging.StreamHandler(sys.stderr)
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(TerminalFormatter())
        for module in modules:
            logger = logging.getLogger(module)
            logger.addHandler(handler)
            logger.setLevel(getattr(logging, what.upper()))
            log = getattr(logger, what)
            log("Outputting %s messages from module %s", what, module)
            if (module != '__main__' and
                    importlib.util.find_spec(module) is None):
                logger.warning("Target module %s not found by importlib.",
                               module)
