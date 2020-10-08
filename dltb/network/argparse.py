"""Network specific argument parsing.
This module provides functions to add arguments to an
:py:class:`ArgumentParser` for specifying a :py:class:`Network`
on the command line.

"""

# standard imports
from typing import Iterator
from argparse import ArgumentParser, Namespace

# toolbox imports
from . import Network


def prepare(parser: ArgumentParser) -> None:
    """Add arguments to an :py:class:`ArgumentParser`, that
    allow to specify a network on the command line.

    Parameters
    ----------
    parser: ArgumentParser
        The argument parser to which arguments are to be added.
    """
    parser.add_argument('--alexnet', help="Use AlexNet (Tensorflow)",
                        action='store_true', default=False)
    parser.add_argument('--resnet', help="Use ResNet (Torch)",
                        action='store_true', default=False)
    parser.add_argument('--network', help='Name of a network to use')
    parser.add_argument('--model', help='Filename of model to use',
                        default='models/example_keras_mnist_model.h5')
    parser.add_argument('--framework', help="The framework to use "
                        "(torch/keras/tensorflow)")
    parser.add_argument('--list-networks', help='List known networks',
                        action='store_true', default=False)


def network(args: Namespace) -> Network:
    """Evaluate command line arguments to create a
    :py:class:`Network`. If multiple networks are specified only one
    will be returned. If multiple networks are expected, use
    :py:func:`networks`.

    Parameters
    ----------
    args: Namespace
        An `Namespace` from parsing the command line
        arguments with `parser.parse_args()`.

    Result
    ------
    network: Network
        The network obtained from the command line arguments. If no
        network was specified, `None` is returned.
    """

    try:
        # get the first network specified on the command line
        network = next(networks(args))
        network.prepare()
    except StopIteration:
        network = None

    return network


def networks(args: Namespace) -> Iterator[Network]:
    """Evaluate command line arguments to create
    :py:class:`Network` objects.

    Parameters
    ----------
    args: Namespace
        An `Namespace` from parsing the command line
        arguments with `parser.parse_args()`.

    Result
    ------
    networks: Iterator[Network]
        The network obtained from the command line arguments.
    """

    frameworks = ['torch', 'keras', 'tensorflow', 'caffe']
    # FIXME[old]: "keras-tensorflow" or "keras-theano"

    if args.framework in frameworks:
        frameworks.remove(args.framework)
        frameworks.insert(0, args.framework)

    # FIXME[old]
    # if args.framework.startswith('keras'):
    #     networks.append('keras-network')
    # elif args.framework == 'torch':
    #     networks.append('torch-network')

    if args.list_networks:
        print("Known networks:", ", ".join(Network.instance_register.keys()))

    if args.network:
        yield Network[args.network]
    if args.alexnet:
        yield Network['alexnet-tf']
    if args.resnet:
        for framework in frameworks:
            if framework == 'torch':
                yield Network['resnet-torch']
            elif framework == 'keras':
                yield Network['resnet-keras']
            else:
                continue
            break

    # raise StopIteration("No more network specified on the command line")

    name = "FIXME[old]"
    if name == 'keras-network':
        # FIXME[concept]:
        #   here we really need the command line arguments!
        # dash_idx = args.framework.find('-')
        # backend = args.framework[dash_idx + 1:]
        # network = keras(backend, args.cpu, model_file=args.model)
        network = None
    elif name == 'torch-network':
        # FIXME[hack]: provide these parameters on the command line ...
        # net_file = 'models/example_torch_mnist_net.py'
        # net_class = 'Net'
        # parameter_file = 'models/example_torch_mnist_model.pth'
        # input_shape = (28, 28)
        # network = torch(args.cpu, net_file, net_class,
        #                 parameter_file, input_shape)
        network = None
