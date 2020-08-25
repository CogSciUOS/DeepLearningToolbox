# standard imports
from typing import Iterator
from argparse import ArgumentParser, Namespace

# toolbox imports
from .config import config


def prepare(parser: ArgumentParser) -> None:
    """Add arguments to an :py:class:`ArgumentParser`, that
    allow to specify general options for the Deep Learning ToolBox.

    Parameters
    ----------
    parser: ArgumentParser
        The argument parser to which arguments are to be added.
    """
    parser.add_argument('--warn-missing-dependencies',
                        help="Issue warnings on missing software packages",
                        action='store_true', default=False)

def evalute(args: Namespace) -> None:
    """Evaluate command line arguments for configuring the toolbox.

    Parameters
    ----------
    args: Namespace
        An `Namespace` from parsing the command line
        arguments with `parser.parse_args()`.
    """

    if args.warn_missing_dependencies:
        config.warn_missing_dependencies = True
