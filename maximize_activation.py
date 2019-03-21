#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A script to run activation maximization

.. moduleauthor:: Antonia Hain, Ulf Krumnack
"""

import os
import sys
import argparse
import signal
import util

import logging
#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.debug(f"Logger[debug]: {logger.getEffectiveLevel()}")

from tools.am import Engine, Config

def main():
    '''Start the program.'''

    parser = argparse.ArgumentParser(
        description='Activation Maximization.')
    parser.add_argument('--model', help='Filename of model to use',
                        default='models/example_keras_mnist_model.h5')
    parser.add_argument('--framework', help='The framework to use.',
                        choices=['tensorflow', 'torch',
                                 'keras-tensorflow', 'keras-theano'],
                        default='tensorflow')
    parser.add_argument('--cpu', help='Do not attempt to use GPUs',
                        action='store_true',
                        default=False)
    args = parser.parse_args()

    util.use_cpu = args.cpu

    from network.tensorflow import Network as TensorFlowNetwork
    checkpoint = os.path.join('models', 'example_tf_alexnet',
                              'bvlc_alexnet.ckpt')
    network = TensorFlowNetwork(checkpoint=checkpoint, id='AlexNet')
    network._online()

    engine = Engine(config=Config())
    engine.add_observer(MatplotlibObserver())
    signal.signal(signal.SIGINT, lambda sig, frame: engine.stop())
    engine.maximize_activation(network)


import matplotlib.pyplot as plt
plt.ion()
plt.show()

class MatplotlibObserver(Engine.Observer):
    im = None
    
    def maximization_changed(self, engine: Engine, info: Engine.Change) -> None:
        """Respond to change in the activation maximization Engine.

        Parameters
        ----------
        engine: Engine
            Engine which changed (since we could observe multiple ones)
        info: ConfigChange
            Object for communicating which aspect of the engine changed.
        """
        if info.image_changed:
            image = engine.get_snapshot(normalize=True)
            if self.im is None:
                self.im = plt.imshow(image)
            else:
                self.im.set_data(image)
            plt.title(f'Iteration: {engine.iteration}')
            plt.draw()
            plt.pause(0.001)

import cv2
class CvObserver(Engine.Observer):
    def maximization_changed(self, engine: Engine, info: Engine.Change) -> None:
        if info.image_changed:
            cv2.imshow(f'{engine.iteration}', engine.image)

if __name__ == '__main__':
    main()
