#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A command line interface for running and testing networks.

.. moduleauthor:: Ulf Krumnack


"""
# FIXME[todo]:
#  - actviation: extract actviation from network
#  - eval: evaluate network on dataset
#  - train: train on dataset

# standard imports
import sys
import logging
import argparse

# toolbox imports
from dltb.network import argparse as NetworkArgparse
from util.error import print_exception

# logging
LOG = logging.getLogger(__name__)


def main():

    parser = \
        argparse.ArgumentParser(description='Deep-learning based classifiers')
    parser.add_argument('--architecture', action='store_true', default=False,
                        help="display the network's architecture")
    parser.add_argument('--summary', action='store_true', default=False,
                        help="display the network's architecture")
    parser.add_argument('--backend-summary', action='store_true',
                        default=False,
                        help="display backend specific summary of the network")
    parser.add_argument('--layer', help='describe the specified layer')

    NetworkArgparse.prepare(parser)
    args = parser.parse_args()

    network = NetworkArgparse.network(parser, args)
    if network is None:
        print("No network was specified.")
        return

    if args.summary:
        print(f"Network: {network}")
        print(f"  input: '{network.input_layer_id()}', "
              f"shape: {network.get_input_shape()}, "
              f"{network._get_input_shape()}")
        print(f"  output: '{network.output_layer_id()}',"
              f" shape: {network.get_output_shape()}")

    if args.architecture:
        for index, layer in enumerate(network.layers()):
            print(f"{index:3} {str(layer):20}  "
                  f"{str(layer.input_shape):20} -> "
                  f"{str(layer.output_shape):20}")

    if args.backend_summary:
        if 'network.keras' in sys.modules:
            KerasNetwork = sys.modules['network.keras'].Network
            if isinstance(network, KerasNetwork):
                network.model.summary()
        else:
            print("Don't know how to obtain a backend specific summary "
                  "for this network, sorry!")

    if args.layer:
        layer = network[args.layer]
        print(f"Layer: {layer}")
        print(f"  input: '{layer.input_shape}'")
        print(f"  output: '{layer.output_shape}'")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exception:
        print_exception(exception)
        sys.exit(1)
