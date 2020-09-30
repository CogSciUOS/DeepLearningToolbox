#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A command line interface for extracting activation values from a network.

.. moduleauthor:: Ulf Krumnack

"""

# standard imports
import os
import sys
import logging
import argparse

# third party imports

# toolbox imports
from network import Network
from network import argparse as NetworkArgparse
from datasource.imagenet import ImageNet
from dltb.base.data import Data
from dltb.tool.classifier import Classifier

# logging
LOG = logging.getLogger(__name__)


def main():

    parser = \
        argparse.ArgumentParser(description="Activation extraction from "
                                "layers of a neural network")
    NetworkArgparse.prepare(parser)
    args = parser.parse_args()

    network = NetworkArgparse.network(args)
    if network is None:
        logging.error("No network was specified.")
        return

    print(network)
    print(type(network))
    # print("len:", len(network))
    print("layer_dict:", network.layer_dict)

    import network.torch
    if isinstance(network, network.torch.Network):
        print("Torch network!")
    return

    datasource = ImageNet()
    datasource.prepare()
    if datasource is None:
        logging.error("No datasource was specified.")
        return

    print(datasource)

if __name__ == "__main__":
    main()
