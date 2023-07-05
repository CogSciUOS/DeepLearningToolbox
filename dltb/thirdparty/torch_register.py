"""Register torch implementations.

This module should only be imported if the `torch` package is installed.
"""

# standard imports
from argparse import Namespace

# toolbox imports
from dltb.config import config
from dltb.util.importer import importable, import_module
from dltb.util.importer import add_postimport_depency
from dltb.network import Network
from dltb.network.argparse import extend, list_networks, select_network
from . import THIRDPARTY


if not importable('torch'):
    raise ImportError("Registering torch definitions failed "
                      "as module 'torch' is not importable.")

add_postimport_depency('torch', ('.torch', THIRDPARTY))
add_postimport_depency('torchvision', ('.pil', THIRDPARTY))

Network.register_instance('resnet-torch', THIRDPARTY + '.torch.examples',
                          'DemoResnetNetwork')


config.add_property('torch_hub_github', default='pytorch/vision:v0.6.0',
                    description='Default github to use with torch hub.')


def prepare_argument_parser(group, name=None) -> None:
    """Add torch related command line arguments to an `ArgumentParser`.
    """
    if name == 'framework':
        group.add_argument('--torch', metavar='GITHUB', nargs='?',
                           const=True, help="use the torch backend")
        group.add_argument('--torch-hub', metavar='GITHUB', nargs='?',
                           const=config.torch_hub_github,
                           help='Use torch hub from GITHUB repository')


def process_arguments(args: Namespace) -> None:
    """Process torch related command line arguments.

    Arguments
    ---------
    """
    if args.torch_hub:
        hub = import_module('.hub', 'dltb.thirdparty.torch')

        if args.list_networks:
            list_networks(hub.Network.models(args.torch_hub))

        if args.network and isinstance(args.network, str):
            name = select_network(args.network,
                                  hub.Network.models(args.torch_hub))
            args.network = hub.Network(repository=args.torch_hub,
                                       model=name)

    if args.torch:
        vision = import_module('.vision', 'dltb.thirdparty.torch')
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
