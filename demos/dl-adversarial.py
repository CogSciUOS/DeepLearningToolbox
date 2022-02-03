#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SGE run parameters
#
# We need a cuda card
#$ -l cuda=1
#
# Use the environment
#$ -V
#
# Execute in the current working directory
#$ -cwd

# python demos/dl-adversarial.py --torch --network inception_v3 --adversarial --targets=131 examples/cat.jpg



# pylint --extension-pkg-whitelist=torch dltb/thirdparty/torch/model.py


"""A command line interface for creating and testing adversarial examples.

.. moduleauthor:: Ulf Krumnack


Examples
--------

  python demos/dl-adversarial.py --torch --network inception_v3 --classify examples/cat.jpg

"""
# standard imports
from datetime import datetime
import os
import sys
import logging
import argparse
import socket

# thirdparty imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio

# toolbox imports
import dltb.argparse as ToolboxArgparse
from dltb.base.image import Image
from dltb.util.itertools import Selection
from dltb.thirdparty.torch.vision import Network as TorchNetwork
from dltb.thirdparty.torch.adversarial import IterativeGradientSignAttacker
from dltb.thirdparty.torch.adversarial import Example

# logging
LOG = logging.getLogger(__name__)


__version__ = '0.1'


# output_directory = os.path.join(os.getenv('HOME'), 'scratch', 'images')
# output_directory = '.'
output_directory = 'outputs'

# -----------------------------------------------------------------------------


# FIXME[todo]: this may be especiall interesting for distribute/grid
# computation


def setup_logging(args) -> None:
    """Set up the logging depending on the command line arguments.
    """
    # if args.debug:
    #     logging.basicConfig(level=logging.DEBUG)
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info("Running on %s, at %s.",
                 socket.gethostname(), datetime.today().strftime('%Y-%m-%d'))
    logging.info("GPU is %savailable.",
                 '' if TorchNetwork.gpu_is_available() else 'not ')
    logging.info(TorchNetwork.framework_info())
    if 'SGE_O_HOST' in os.environ:
        logging.info("Running as grid job, submitted from %s.",
                     os.getenv('SGE_O_HOST'))

    cwd = os.getcwd()
    logging.info("Current working directory is '%s'.", cwd)

    if cwd not in sys.path:
        logging.info("Inserting current working directory "
                     "into the Python path.")
        sys.path.insert(0, cwd)

    logging.debug("Python path:")
    for i, directory in enumerate(sys.path):
        logging.debug("  [%d] '%s'", i, directory)

    logging.info("TORCH_HOME=%s", os.getenv('TORCH_HOME', 'None'))
    logging.info("XDG_CACHE_HOME=%s", os.getenv('XDG_CACHE_HOME', 'None'))

    logging.info("Command line options:")
    logging.info("  classify=%s", args.classify)
    logging.info("  framework=%s", args.framework)
    logging.info("  verbose=%s", args.verbose)
    logging.info("  model=%s", args.model)
    logging.info("  targets=%s", args.targets)

# -----------------------------------------------------------------------------



def visualize(original: Example,
              adversarial: Example,
              epsilon: float) -> None:
    """Visualize an adversarial example.

    Arguments
    ---------
    original:
        The original example

    adversarial:
        The adversarial example

    epsilon:
    """

    diff = original.inputs - adversarial.inputs
    min_diff, max_diff = diff.min(), diff.max()

    _, axes = plt.subplots(1, 3, figsize=(18, 8))
    axes[0].imshow(original.inputs)
    axes[0].set_title('Clean Example', fontsize=20)

    # axes[1].imshow(x_grad)
    axes[1].hist(diff.flat)
    axes[1].set_title('Perturbation', fontsize=20)
    axes[1].set_yticklabels([])
    axes[1].set_xticklabels([])
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    axes[2].imshow(adversarial.inputs)
    axes[2].set_title('Adversarial Example', fontsize=20)

    axes[0].axis('off')
    axes[2].axis('off')

    axes[0].text(1.1, 0.5, "+{}*".format(round(epsilon, 3)),
                 size=15, ha="center", transform=axes[0].transAxes)

    axes[0].text(0.5, -0.13,
                 f"Prediction: {original.label}\n"
                 f"Probability: {original.score:.4f}",
                 size=15, ha="center",
                 transform=axes[0].transAxes)

    axes[1].text(1.1, 0.5, " = ", size=15, ha="center",
                 transform=axes[1].transAxes)

    axes[1].text(0.5, -0.13,
                 f"Adversarial Image: min={adversarial.inputs.min():.4f}, "
                 f"max={adversarial.inputs.max():.4f}\n"
                 f"Perturbation: min={min_diff:.4f}, max={max_diff:.4f}",
                 size=15, ha="center",
                 transform=axes[1].transAxes)

    axes[2].text(0.5, -0.13,
                 f"Prediction: {adversarial.label}\n"
                 f"Probability: {adversarial.score:.4f}", size=15, ha="center",
                 transform=axes[2].transAxes)

    plt.show()

# -----------------------------------------------------------------------------


def main():
    """The main program.
    """

    parser = \
        argparse.ArgumentParser(description='Deep-learning based adversarial '
                                'examples')
    parser.add_argument('--classify', action='store_true',
                        help="classify given input files")
    parser.add_argument('--adversarial', action='store_true',
                        help="perform adversarial attack")
    parser.add_argument('--target', action='store_true', default=False,
                        help='create targeted adversarial example')
    parser.add_argument('--targets', type=str,
                        help="class(es) for targeted adversarial attack")
    parser.add_argument('--show', action='store_true',
                        help='show the images that are processed')
    parser.add_argument('--save', action='store_true',
                        help='store adversarial examples')
    parser.add_argument('--reload', action='store_true',
                        help='reload saved image')
    parser.add_argument('--verbose', action='store_true',
                        help="increase output verbosity")
    parser.add_argument('--version', action='version',
                        version=f"%(prog)s {__version__}")
    ToolboxArgparse.add_arguments(parser, components=('network', ))
    parser.add_argument('files', metavar='FILE', nargs='*',
                        help="files to process")
    args = parser.parse_args()
    ToolboxArgparse.process_arguments(parser, args)

    #
    # Process the command line options
    #

    # output some information
    setup_logging(args)

    network = args.network

    logging.info("Network: %s", network.name)
    # network.range_info()

    if args.adversarial:
        attacker = \
            IterativeGradientSignAttacker(max_steps=20, min_confidence=.99)
        logging.info("Targets=%s", args.targets)

    #
    # main loop of the program: process command line arguments
    #

    for filename in args.files:

        print(f"Loading image '{filename}'")

        # image = imread(input_filename)
        image_tensor = network.image_to_internal(filename)
        print("image_tensor:", type(image_tensor), image_tensor.shape,
              image_tensor.dtype,
              image_tensor.min(), image_tensor.max())

        # Classifiy the original image
        if args.classify:
            network.classify_image(image_tensor)

        # Create the adversarial example
        if args.adversarial:

            # perform targeted adversarial attack
            for target in Selection(args.targets):
                output_filename = \
                    os.path.join(output_directory, 'adversarial_examples',
                                 f"image-{network.name}-{target}.png")

                original, adversarial = \
                    attacker(network, image_tensor, target=target,
                             result=('original', 'adversarial'))

                diff = original.inputs - adversarial.inputs

                print("Maximal difference between image and "
                      "adversarial example: "
                      f"{np.abs(diff).max():.4f} "
                      f"(real={diff.min()}/{diff.max()}, "
                      f"levels={np.round(np.abs(diff).max()*255)})")

                #
                # Visualization
                #
                if args.show:
                    eps = .123  # FIXME[hack]
                    visualize(original, adversarial, eps)

                #
                # Storing the image
                #

                adversarial_uint8 = \
                    (adversarial.inputs*255).astype(np.uint8)
                if os.path.exists(output_filename):
                    logging.warning("File '%s' already exists. "
                                    "Will not overwrite.", output_filename)
                else:
                    if (filename.endswith('.jpeg') or
                        filename.endswith('.jpg')):
                        #
                        imageio.imwrite(output_filename, adversarial_uint8,
                                        quality=100)
                    else:
                        imageio.imwrite(output_filename, adversarial_uint8)

                #
                # Reloading the image
                #
                if args.reload:
                    relaoded_image_uint8 = imageio.imread(output_filename)
                    relaoded_image_float = \
                        relaoded_image_uint8.astype(np.float32)/255
                    reload_diffb = \
                        adversarial_uint8 - relaoded_image_uint8
                    reload_diff = adversarial.inputs - relaoded_image_float
                    print("Maximal image difference after reload "
                          f"(image='{output_filename}'): "
                          f"uint8={np.abs(reload_diffb).max()}, "
                          f"float={np.abs(reload_diff).max():.4f}, "
                          f"threshold={1./255:.4f}")

                    adv_example3 = \
                        network.image_to_internal(relaoded_image_float)
                    print("Predictions for the reloaded image:")
                    network.classify_image(adv_example3, preprocess=False)


if __name__ == "__main__":
    main()
