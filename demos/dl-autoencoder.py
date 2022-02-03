#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A command line interface for running and testing autoencoder.

.. moduleauthor:: Ulf Krumnack


Examples
--------

Train an autoencoder on a dataset

   python demos/dl-autoencoder.py --mnist --train --display


----------------------------
The TensorFlow autoencoder training and plot


# training with plotting
from dltb.datasource import Datasource
from dltb.thirdparty.tensorflow.ae import Autoencoder
from dltb.tool.train import Trainer
from tqdm import tqdm

datasource_train = Datasource(module='mnist', one_hot=True)
ae = Autoencoder(shape=datasource_train.shape, code_dim=2)
trainer = Trainer(trainee=ae, training_data=datasource_train)

from dltb.thirdparty.matplotlib import MplDisplay, MplScatter2dPlotter
display = MplDisplay()
plotter = MplScatter2dPlotter(display=display)
def plotit(trainer, plotter):
    data, labels = datasource_train[:10000, ('array', 'label')]
    trainer.trainee.plot_data_codes_2d(data, labels=labels, plotter=plotter)
plotit(trainer, plotter)
trainer.at_end_of_epoch(plotit, trainer, plotter)

# run the trainer in a background thread
trainer.train(epochs=5, restore=False, progress=tqdm, run=True)

# run the plotter in the main thread
display.show(on_close=trainer.stop)


"""
# standard imports
from typing import Optional
import argparse
import logging

# third party imports
from tqdm import tqdm

# toolbox imports
import dltb.argparse as ToolboxArgparse
from dltb.datasource import Datasource
from dltb.tool.autoencoder import Autoencoder
from dltb.tool.train import Trainer
from dltb.util.plot import Display, Scatter2dPlotter, TilingPlotter

# logging
LOG = logging.getLogger(__name__)


def demp_plot_codes_2d(autoencoder: Autoencoder, plotter: Scatter2dPlotter,
                       datasource: Datasource) -> None:
    """Plot the code distribution for a 2D code space.
    """
    data, labels = datasource[:10000, ('array', 'label')]
    autoencoder.plot_data_codes_2d(data, labels=labels, plotter=plotter)


def demp_plot_recoded_images(autoencoder: Autoencoder, plotter: TilingPlotter,
                             datasource: Datasource) -> None:
    """Plot reconstructed images.
    """
    data = datasource[:100, ('array')]
    autoencoder.plot_recoded_images(data, plotter=plotter)


def demo_training(autoencoder: Autoencoder, datasource: Datasource,
                  progress: Optional = None,
                  display: Optional[Display] = None,
                  plot: str = 'codes') -> None:
    """Train an autoencoder.
    """
    print("Training an autoencoder")
    trainer = Trainer(trainee=autoencoder, training_data=datasource)

    if display is not None:
        if plot == 'codes':
            plot_function = demp_plot_codes_2d
            plotter = Scatter2dPlotter(display=display)
        else:
            plot_function = demp_plot_recoded_images
            plotter = TilingPlotter(display=display)

        # plot the initial state
        plot_function(autoencoder, plotter, datasource)
        # register the plot function to be done at every epoch
        # (trainer.at_end_of_epoch) or at every batch
        # (trainer.at_end_of_batch).
        trainer.at_end_of_batch(plot_function, autoencoder, plotter, datasource)

    # run the trainer: when using a display, training is run in a background
    # thread while the main thread is doing the GUI event loop
    #
    # pylint - disable false positive for keyword arguments introduced
    # by decorator (should be fixed in pylint 2.13), see
    # https://github.com/PyCQA/pylint/issues/258
    # pylint: disable=unexpected-keyword-arg
    trainer.train(epochs=5, restore=False, progress=progress,
                  run=display is not None)

    if display is not None:
        # run the display in the main thread
        display.show(on_close=trainer.stop)
        plotter.axes = None

        # once training is done
        display.axes.clear()
        display.axes.plot(trainer.loss_history())
        display.show()


def demo_inference(autoencoder: Autoencoder, datasource: Datasource) -> None:
    """Perform inference with the autoencoder.
    """
    data, labels = datasource[:1000, ('array', 'label')]
    # codes = autoencoder.encode(data)
    autoencoder.plot_data_codes_2d(data, labels=labels)


def main():
    """The main program.
    """

    parser = \
        argparse.ArgumentParser(description='Deep-learning based classifiers')
    parser.add_argument('--train', action='store_true', default=False,
                        help='train the autoencoder on the given datasource')
    parser.add_argument('--inference', action='store_true', default=False,
                        help='perform inference with the autoencoder')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='plot codes and/or results')

    ToolboxArgparse.add_arguments(parser, ('network',))
    args = parser.parse_args()
    ToolboxArgparse.process_arguments(parser, args)

    #
    # Create an autoencoder
    #

    data_shape = (28, 28, 1)
    # autoencoder = NetworkArgparse.network(parser, args)
    autoencoder = Autoencoder(shape=data_shape, code_dim=2, module='tf')


    # Further command line arguments
    display = Display() if args.plot else None

    #
    # Use the autoencoder
    #
    if args.train:
        datasource_train = Datasource(module='mnist', one_hot=True)
        demo_training(autoencoder, datasource_train, progress=tqdm,
                      display=display)

    if args.inference:
        test_data = Datasource(module='mnist', section='test', one_hot=True)
        demo_inference(autoencoder, test_data)


if __name__ == "__main__":
    main()
