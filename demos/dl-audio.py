#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A command line interface for running audio models.
"""
# standard imports
from typing import List, Union
import logging
import argparse
import signal
import time


def main():
    """The main program.
    """

    parser = \
        argparse.ArgumentParser(description="Activation extraction from "
                                "layers of a neural network")
    parser.add_argument('--gui', action='store_true',
                        help='display activations in graphical user interface')
    parser.add_argument('--iterate', action='store_true',
                        help='iterate over activation values')
    parser.add_argument('--top', type=int,
                        help='obtain top n activation values')
    parser.add_argument('--store', action='store_true',
                        help='store activation values')
    parser.add_argument('--archive', action='store_true',
                        help='use activation values from archive')
    parser.add_argument('--store-top', action='store_true',
                        help='store top activation values')
    parser.add_argument('image', metavar='IMAGE', nargs='*',
                        help='input image(s)')

    ToolboxArgparse.add_arguments(parser)
    NetworkArgparse.prepare(parser, layers=True)
    DatasourceArgparse.prepare(parser)

    args = parser.parse_args()
    ToolboxArgparse.process_arguments(parser)


if __name__ == "__main__":
    main()

