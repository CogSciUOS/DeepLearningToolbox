#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A command line interface for extracting activation values from a network.

.. moduleauthor:: Ulf Krumnack

"""

# standard imports
import logging
import argparse

# toolbox imports
from dltb.base.data import Data
from dltb.util.image import imread
from dltb.network import argparse as NetworkArgparse
from dltb.tool.activation import ActivationTool, ActivationWorker

# logging
LOG = logging.getLogger(__name__)


def main():
    """The main program.
    """

    parser = \
        argparse.ArgumentParser(description="Activation extraction from "
                                "layers of a neural network")
    NetworkArgparse.prepare(parser)
    args = parser.parse_args()

    network = NetworkArgparse.network(args)
    if network is None:
        logging.error("No network was specified.")
        return

    network.summary()

    image = imread('images/elephant.jpg')
    image = network.image_to_internal(image)
    activations1 = network.get_activations(image)
    print('1: network.get_activations(image):')
    for index, activation in enumerate(activations1):
        print(f" ({index}) {activation.shape}")

    tool = ActivationTool(network)
    activations2 = tool(image)
    print(f'2: ActivationTool(network)(image): {type(activations2)}')
    for layer_id, activation in activations2.items():
        print(f" ({layer_id}) {activation.shape}")

    data = Data(image)
    activations3 = tool(data)
    print('3: ActivationTool(network)(data):')
    for layer_id, activation in activations3.items():
        print(f" ({layer_id}) {activation.shape}")

    data = Data(image)
    tool.apply(data)
    activations4 = tool.data_activations(data)
    print('4: ActivationTool(network).apply(data):')
    for layer_id, activation in activations4.items():
        print(f" ({layer_id}) {activation.shape}")

    worker = ActivationWorker(tool=tool)
    data = Data(image)
    worker.work(data, busy_async=False)
    activations5 = tool.data_activations(data)
    print('5: ActivationWorker(tool).work(data):')
    for layer_id, activation in activations5.items():
        print(f" ({layer_id}) {activation.shape}")


if __name__ == "__main__":
    main()
