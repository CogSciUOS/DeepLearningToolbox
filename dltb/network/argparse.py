"""Network specific argument parsing.
This module provides functions to add arguments to an
:py:class:`ArgumentParser` for specifying a :py:class:`Network`
on the command line.

"""

# standard imports
from typing import Iterator, Union, Iterable
from argparse import ArgumentParser, Namespace
import sys
import itertools

# toolbox imports
from . import Network


_argument_extenders = []
_argument_processors = []


def extend(extender, processor) -> None:
    """Extend nework command line argument processing.

    Arguments
    ---------
    extender:
        A function that can extend an `ArgumentParser` with additional
        arguments.
    processor:
        A function that can processes the additional command line arguments
        added by the `extender`.
    """
    _argument_extenders.append(extender)
    _argument_processors.append(processor)


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

    framework_group = group.add_mutually_exclusive_group()
    framework_group.add_argument('--framework', help="the framework to use "
                                 "(torch/keras/tensorflow)")
    for argument_extender in _argument_extenders:
        argument_extender(framework_group, 'framework')

    network_group = group.add_mutually_exclusive_group()
    network_group.add_argument('--alexnet', help="use AlexNet (Tensorflow)",
                               action='store_true', default=False)
    network_group.add_argument('--resnet', help="use ResNet (Torch)",
                               action='store_true', default=False)
    network_group.add_argument('--densenet', help='use densenet (Keras)',
                               action='store_true', default=False)
    network_group.add_argument('--network', help='name of a network to use')
    network_group.add_argument('--model', help='filename of model to use',
                               default='models/example_keras_mnist_model.h5')
    network_group.add_argument('--list-networks', help='list known networks',
                               action='store_true', default=False)
    for argument_extender in _argument_extenders:
        argument_extender(network_group, 'network')

    for argument_extender in _argument_extenders:
        argument_extender(group)

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
        the_network = next(networks(args))
        the_network.prepare()
    except StopIteration:
        parser.error("No network was specified.")

    if layers is False:
        return the_network

    try:
        layers = list(the_network.layers(args.layer or
                                         (None if layers is True else layers)))
    except KeyError as error:
        parser.error(f"Invalid layer: {error.args[0]}")

    return the_network, layers


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
        # place selected framework at the beginning of the frameworks list
        frameworks.remove(args.framework)
        frameworks.insert(0, args.framework)

    # FIXME[old]
    # if args.framework.startswith('keras'):
    #     networks.append('keras-network')
    # elif args.framework == 'torch':
    #     networks.append('torch-network')


    if args.network:
        if isinstance(args.network, Network):
            yield args.network
        else:
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
    if args.densenet:
        # FIXME[hack]: DenseNet should be properly integrated into
        # dltb.thirdparty.keras
        import dltb.thirdparty.tensorflow
        from experiments.densenet import DenseNet
        yield DenseNet()

    # raise StopIteration("No more network specified on the command line")

    name = "FIXME[old]"
    if name == 'keras-network':
        # FIXME[concept]:
        #   here we really need the command line arguments!
        # dash_idx = args.framework.find('-')
        # backend = args.framework[dash_idx + 1:]
        # network = keras(backend, args.cpu, model_file=args.model)
        _network = None
    elif name == 'torch-network':
        # FIXME[hack]: provide these parameters on the command line ...
        # net_file = 'models/example_torch_mnist_net.py'
        # net_class = 'Net'
        # parameter_file = 'models/example_torch_mnist_model.pth'
        # input_shape = (28, 28)
        # network = torch(args.cpu, net_file, net_class,
        #                 parameter_file, input_shape)
        _network = None


def process_arguments(parser: ArgumentParser, args: Namespace = None) -> None:
    """Evaluate command line arguments for configuring networks.

    Parameters
    ----------
    parser: ArgumentParser
        The argument parser (used for error processing).
    args: Namespace
        An `Namespace` from parsing the command line
        arguments with `parser.parse_args()`.
    """
    for process in _argument_processors:
        process(args)

    if args.list_networks:
        list_networks(Network.instance_register.keys())


def list_networks(network_names: Iterable[str]) -> None:
    """Ouput a list of networks and exit.

    Arguments
    ---------
    network_names:
        The network names.
    """
    print("Available models:")
    for i, name in enumerate(network_names, 1):
        print(f" {i:2}) {name}")
    sys.exit(0)


def select_network(arg: str, network_names: Iterable[str]) -> str:
    """Select a network name from an list of network names, either
    based on its (1-based!) index, or its name.

    Argument
    --------
    arg:
        The command line argument based on which a network should
        be choosen.
    networks:
    """
    try:
        return next(itertools.islice(network_names, int(arg)-1, None))
    except StopIteration as exception:
        raise ValueError(f"{arg} is an invalid network index.") from exception
    except ValueError as exception:
        if arg not in network_names:
            raise ValueError(f"'{arg}' is not a valid network name.") \
                from exception
    return arg
