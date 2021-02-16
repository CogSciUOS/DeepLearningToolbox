"""Datasource specific argument parsing.
This module provides functions to add arguments to an
:py:class:`ArgumentParser` for specifying a :py:class:`Datasource`
on the command line.

"""

# standard imports
from typing import Iterator
from argparse import ArgumentParser, Namespace

# toolbox imports
from . import Datasource


def prepare(parser: ArgumentParser) -> None:
    """Add arguments to an :py:class:`ArgumentParser`, that
    allow to specify a datasource on the command line.

    Parameters
    ----------
    parser: ArgumentParser
        The argument parser to which arguments are to be added.
    """
    parser.add_argument('--datasource', help='Name of a datasource to use')
    parser.add_argument('--imagenet-val', help="Use Imagenet validation",
                        action='store_true', default=False)
    parser.add_argument('--list-datasources', help='List known datasources',
                        action='store_true', default=False)


def datasource(parser: ArgumentParser, args: Namespace) -> Datasource:
    """Evaluate command line arguments to create a
    :py:class:`Datasource`. If multiple datasources are specified only one
    will be returned. If multiple datasources are expected, use
    :py:func:`datasources`.

    Parameters
    ----------
    parser: ArgumentParser
        The argument parser (used for error handling).
    args: Namespace
        An `Namespace` from parsing the command line
        arguments with `parser.parse_args()`.

    Result
    ------
    datasource: Datasource
        The datasource obtained from the command line arguments. If no
        datasource was specified, `None` is returned.
    """
    if args is None:
        args = parser.parse_args()

    try:
        # get the first datasource specified on the command line
        datasource = next(datasources(args))
        datasource.prepare()
    except StopIteration:
        datasource = None

    return datasource


def datasources(args: Namespace) -> Iterator[Datasource]:
    """Evaluate command line arguments to create
    :py:class:`Datasource` objects.

    Parameters
    ----------
    args: Namespace
        An `Namespace` from parsing the command line
        arguments with `parser.parse_args()`.

    Result
    ------
    datasources: Iterator[Datasource]
        The datasource obtained from the command line arguments.
    """

    if args.list_datasources:
        print("Known datasources:", ", ".
              join(Datasource.instance_register.keys()))

    if args.datasource:
        yield Datasource[args.datasource]
    if args.imagenet_val:
        yield Datasource['imagenet-val']

    #raise StopIteration("No more datasource specified on the command line")
