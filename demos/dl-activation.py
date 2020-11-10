#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A command line interface for extracting activation values from a network.

.. moduleauthor:: Ulf Krumnack

"""

# standard imports
import logging
import argparse

# third party imports
import numpy as np

# toolbox imports
from dltb.base.data import Data
from dltb.util.image import imread
from dltb.network import Network, argparse as NetworkArgparse
from dltb.datasource import Datasource, Datafetcher
from dltb.datasource import argparse as DatasourceArgparse
from dltb.tool.activation import ActivationTool, ActivationWorker

# logging
LOG = logging.getLogger(__name__)


def extract_activations(network: Network,
                        datasource: Datasource, layers,
                        batch_size: int = 128) -> None:
    """Extract activation values for a dataset from a Network.
    """
    tool = ActivationTool(network)
    try:
        samples = len(datasource)
        # Here we could:
        #  np.memmap(filename, dtype='float32', mode='w+',
        #            shape=(samples,) + network[layer].output_shape[1:])
        results = {
            layer: np.ndarray((samples,) + network[layer].output_shape[1:])
            for layer in layers
        }

        fetcher = Datafetcher(datasource, batch_size=batch_size)
        index = 0
        for batch in fetcher:
            print("dl-activation: batch:", type(batch.array), len(batch.array))
            print("dl-activation: indices:", batch[0].index)  # , batch[-1].index
            activations = network.get_activations(batch, layers)
            #print(type(activations), len(activations))
            print("dl-activation: activations:", type(activations[0]))
            for index, values in enumerate(activations):
                results[layers[index]][index:index+len(batch)] = values
            print("dl-activation: batch finished.")
    except InterruptedError:
        print("Interrupted.")


def main():
    """The main program.
    """

    parser = \
        argparse.ArgumentParser(description="Activation extraction from "
                                "layers of a neural network")
    NetworkArgparse.prepare(parser)
    DatasourceArgparse.prepare(parser)
    args = parser.parse_args()

    network = NetworkArgparse.network(args)
    if network is None:
        logging.error("No network was specified.")
        return

    network.summary()

    datasource = DatasourceArgparse.datasource(args)
    if datasource is None:
        logging.error("No datasource was specified.")
        return

    #image = 'images/elephant.jpg'
    image = imread('images/elephant.jpg')
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

    #
    # Now loop over the dataset
    #
    layers = list(network.layer_names())
    print("Layers:", layers)
    extract_activations(network, datasource, layers[-1:])


if __name__ == "__main__":
    main()
