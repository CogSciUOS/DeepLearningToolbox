"""Network specific argument parsing.
This module provides functions to add arguments to an
:py:class:`ArgumentParser` for specifying a :py:class:`Network`
on the command line.

"""

# standard imports
from typing import Iterator, Union
from argparse import ArgumentParser, Namespace

# toolbox imports
from . import Network

def int_or_str(argument: str) -> Union[int, str]:
    """Return the argument either as integer (if it can be converted
    to int) or as string.
    """
    try:
        return int(argument)
    except ValueError:
        return argument


def prepare(parser: ArgumentParser, layers: bool = False) -> None:
    """Add arguments to an :py:class:`ArgumentParser`, that
    allow to specify a network on the command line.

    Parameters
    ----------
    parser: ArgumentParser
        The argument parser to which arguments are to be added.
    """
    group = parser.add_argument_group('Network')

    network_group = group.add_mutually_exclusive_group()
    network_group.add_argument('--alexnet', help="Use AlexNet (Tensorflow)",
                               action='store_true', default=False)
    network_group.add_argument('--resnet', help="Use ResNet (Torch)",
                               action='store_true', default=False)
    network_group.add_argument('--network', help='Name of a network to use')
    network_group.add_argument('--model', help='Filename of model to use',
                               default='models/example_keras_mnist_model.h5')
    network_group.add_argument('--framework', help="The framework to use "
                               "(torch/keras/tensorflow)")
    network_group.add_argument('--list-networks', help='List known networks',
                               action='store_true', default=False)

    if layers:
        group.add_argument('--layer', metavar='LAYER', type=int_or_str,
                           nargs='+', help='Specify specific layer(s)')


def network(parser: ArgumentParser, args: Namespace = None,
            layers: bool = False) -> Network:
    """Evaluate command line arguments to create a
    :py:class:`Network`. If multiple networks are specified only one
    will be returned. If multiple networks are expected, use
    :py:func:`networks`.

    Parameters
    ----------
    parser: ArgumentParser
        The argument parser (used for error proecessing).
    args: Namespace
        An `Namespace` from parsing the command line
        arguments with `parser.parse_args()`.
    layers:
        If not False, the function will return as a second value a
        list of layers. The argument value will be used as a default
        value, if no layers are specified on the command line. The
        value `True` means all layers of the network, the value
        [] means no layers.

    Result
    ------
    network: Network
        The network obtained from the command line arguments. If no
        network was specified, `None` is returned.
    layers: List[Layer]
        A list of layers. This value is only present if the `layers`
        argument is not `False`.
    """
    if args is None:
        args = parser.parse_args()

    try:
        # get the first network specified on the command line
        network = next(networks(args))
        network.prepare()
    except StopIteration:
        parser.error("No network was specified.")

    if layers is False:
        return network

    try:
        layers = list(network.layers(args.layer or
                                     (None if layers is True else layers)))
    except KeyError as error:
        parser.error(f"Invalid layer: {error.args[0]}")

    return network, layers


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
