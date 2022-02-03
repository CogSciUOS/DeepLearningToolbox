"""Integration of the torch library. Torch provides its
own array types, similar to numpy.

"""

# standard imports
from argparse import ArgumentParser, Namespace
import sys
import importlib

# toolbox imports
from dltb.network.argparse import extend, list_networks, select_network


# Default github to use with torch hub.
HUB_DEFAULT_GITHUB = 'pytorch/vision:v0.6.0'


def prepare_argument_parser(group, name=None) -> None:
    """Add torch related command line arguments to an `ArgumentParser`.
    """
    if name == 'framework':
        group.add_argument('--torch', metavar='GITHUB', nargs='?',
                           const=True, help="use the torch backend")
        group.add_argument('--torch-hub', metavar='GITHUB', nargs='?',
                           const=HUB_DEFAULT_GITHUB,
                           help='Use torch hub from GITHUB repository')


def process_arguments(args: Namespace) -> None:
    """Process torch related command line arguments.

    Arguments
    ---------
    """
    if args.torch_hub:
        hub = importlib.import_module('.hub', __name__)

        if args.list_networks:
            list_networks(hub.Network.models(args.torch_hub))

        if args.network and isinstance(args.network, str):
            name = select_network(args.network,
                                  hub.Network.models(args.torch_hub))
            args.network = hub.Network(repository=args.torch_hub,
                                       model=name)

    if args.torch:
        vision = importlib.import_module('.vision', __name__)
        if args.list_networks:
            list_networks(vision.Network.pretrained())

        arg = None
        if args.network and isinstance(args.network, str):
            arg = args.network
        elif isinstance(args.torch, str):
            arg = args.torch

        if arg is not None:
            name = select_network(arg, vision.Network.pretrained())
            args.network = vision.Network(model=name)


extend(prepare_argument_parser, process_arguments)
