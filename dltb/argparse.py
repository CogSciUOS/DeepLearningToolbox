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
from typing import Any, Optional
from argparse import ArgumentParser, Namespace, Action
import sys
import logging
import importlib

# toolbox imports
from .config import config
from .util.logging import TerminalFormatter


# argparse modules from additional deep learning toolbox components
# that want to do command line argument processing
_components = []


class NegateAction(Action):
    # pylint: disable=too-few-public-methods
    """An `Action` allowing to negate an option by prefixing
    it with `'no'`.
    """
    def __call__(self, parser: ArgumentParser, namespace: Namespace,
                 values: Any, option_string: Optional[str] = None) -> None:
        """
        Arguments
        ---------
        parser:
            The ArgumentParser object which contains this action.
        namespace:
            The Namespace object that will be returned by
            `parse_args()`.
        values:
            The associated command-line arguments. There should be no
            value associated with a `NegateAction`.
        option_string:
            The option string that was used to invoke this action.
            The `option_string` argument is only optional if the action
            is associated with a positional argument (which should not be
            the case for a `NegateAction`).
        """
        if option_string:
            setattr(namespace, self.dest, option_string[2:4] != 'no')


def add_arguments(parser: ArgumentParser, components=()) -> None:
    """Add arguments to an :py:class:`ArgumentParser`, that
    allow to specify general options for the Deep Learning ToolBox.

    Parameters
    ----------
    parser: ArgumentParser
        The argument parser to which arguments are to be added.
    """

    for component in components:
        if component == 'network':
            component_argparse = \
                importlib.import_module('..network.argparse', __name__)
        elif component == 'datasource':
            component_argparse = \
                importlib.import_module('..datasource.argparse', __name__)
        else:
            raise ValueError(f"Invalid component '{component}'. Known "
                             "components are 'network' and 'datasource'")

        component_argparse.prepare(parser)
        _components.append(component_argparse)

    #
    # Computation
    #
    group = parser.add_argument_group("Computation")
    group.add_argument('--cpu',
                       help="Force CPU usage (even if GPU is available)",
                       action='store_true', default=False)

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
        The argument parser (used for error processing).
    args: Namespace
        An `Namespace` from parsing the command line
        arguments with `parser.parse_args()`.
    """
    if args is None:
        args = parser.parse_args()

    if args.warn_missing_dependencies:
        config.warn_missing_dependencies = True

    if args.cpu:
        config.use_cpu = True

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

    for component in _components:
        component.process_arguments(parser, args)
