#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''A framework-agnostic visualisation tool for deep neural networks

.. moduleauthor:: Rüdiger Busche, Petr Byvshev, Ulf Krumnack, Rasmus
Diederichsen

'''
import sys
import argparse

import util
from util import addons
from toolbox import Toolbox

import logging

# FIXME[hack]: integrate into Toolbox
logging.debug("importing datasources")

from datasources import Predefined
datasets = Predefined.get_data_source_ids()

logging.debug(f"got datesets: {datasets}")



def main():
    '''Start the program.'''

    parser = argparse.ArgumentParser(
        description='Visual neural network analysis.')
    
    parser.add_argument('--model', help='Filename of model to use',
                        default='models/example_keras_mnist_model.h5')
    parser.add_argument('--data', help='filename of dataset to visualize')
    parser.add_argument('--datadir', help='directory containing input images')
    if (len(datasets) > 0):
        parser.add_argument('--dataset', help='name of a dataset',
                            choices=datasets, default=datasets[0])  # 'mnist'
    parser.add_argument('--framework', help='The framework to use.',
                        choices=['keras-tensorflow', 'keras-theano', 'torch'],
                        default='keras-tensorflow')
    parser.add_argument('--cpu', help='Do not attempt to use GPUs',
                        action='store_true', default=False)
    parser.add_argument('--alexnet', help='Load the AlexNet model',
                        action='store_true', default=False)
    parser.add_argument('--autoencoder',
                        help='Load the autoencoder module (experimental!)',
                        action=addons.UseAddon, default=False)
    parser.add_argument('--advexample',
                        help='Load the adversarial example module (experimental!)',
                        action=addons.UseAddon, default=False)
    args = parser.parse_args()

    util.use_cpu = args.cpu

    toolbox = Toolbox(args)
    rc = toolbox.run()
    sys.exit(rc)


if __name__ == '__main__':
    main()
