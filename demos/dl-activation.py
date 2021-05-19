#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A command line interface for extracting activation values from a network.

.. moduleauthor:: Ulf Krumnack


Invocation:

  dl-activation.py


Iterate over a Datasource, compute activation values and store them in an
archive:

  dl-activation.py --alexnet --imagenet-val --store

Determine top n activation valus for a Network on a Datasource:

  dl-activation.py
"""

# standard imports
from typing import List, Union
import logging
import argparse
import signal
import time

# third-party imports
import numpy as np
from tqdm import tqdm

# GUI imports
from PyQt5.QtWidgets import QApplication, QMainWindow
from qtgui.widgets.activationview import QActivationView

# toolbox imports
import dltb.argparse as ToolboxArgparse
from dltb.base.data import Data
from dltb.util.image import imread, get_display
from dltb.network import Network, Layer, argparse as NetworkArgparse
from dltb.datasource import Datasource, Datafetcher
from dltb.datasource import argparse as DatasourceArgparse
from dltb.tool.activation import ActivationTool, ActivationWorker
from dltb.tool.activation import ActivationsArchiveNumpy
from dltb.tool.activation import TopActivations

# logging
LOG = logging.getLogger(__name__)


def gui_show_activations(activations, title: str = None) -> None:
    first_activations = next(iter(activations if isinstance(activations, list)
                                  else activations.values()))

    print(f"first_activations: {first_activations.shape}")
    first_activations = first_activations.transpose((2, 0, 1))
    print(f"first_activations: {first_activations.shape}")

    activationview.setActivations(first_activations)
    window.setCentralWidget(activationview)
    window.show()

    # Start the event loop.
    app.exec_()


# FIXME[todo]: handling (keyboard) interrupts when running TensorFlow shows
# some delay - the signal is only handled once the tensorflow invocation
# returns, which may take same time, especially when working with large
# batches (at least a second). This results in a unresponsive behaviour.
# To do: check how to deal with interrupts/signals while Tensorflow
# is running.

def signal_handler(signum, frame):
    """Handle KeyboardInterrupt: quit application."""
    print(f"dl-activation: Keyboard interrupt: signum={signum}, frame={frame}")
    if signum == signal.SIGINT:
        print("SIGINT")
    elif signum == signal.SIGQUIT:
        print("SIGQUIT")

def extract_activations1(network: Network,
                         datasource: Datasource, layers: List[Layer] = None,
                         batch_size: int = 128) -> None:
    """Extract activations from a :py:class:`Datasource`.
    """
    tool = ActivationTool(network)
    worker = ActivationWorker(tool=tool)
    worker.set_layers(layers)
    # FIXME[bug]: error in ActivationWorker
    worker.extract_activations(datasource, batch_size=batch_size)


def extract_activations2(network: Network,
                         datasource: Datasource, layers: List[Layer] = None,
                         batch_size: int = 128) -> None:
    """Extract activation values for a dataset from a Network.
    """
    # Setup handling of KeyboardInterrupt (Ctrl-C)
    # original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)
    # Setup handling of Quit (Ctrl-\)
    # original_sigquit_handler = signal.signal(signal.SIGQUIT, signal_handler)

    print(f"dl-activations: extracting activations of layers {layers} "
          f"of network {network}.")

    # tool = ActivationTool(network)
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
            print(f"dl-activation: processing batch of length {len(batch)} "
                  f"with elements given as {type(batch.array)}, "
                  f"first element having index {batch[0].index} "
                  f"and shape {batch[0].array.shape} [{batch[0].array.dtype}]")

            batch_start = time.time()
            activations = network.get_activations(batch, layers)
            batch_end = time.time()
            batch_duration = batch_end - batch_start
            # print(type(activations), len(activations))
            print("dl-activation: "
                  f"activations are of type {type(activations)}, "
                  f"first element is {type(activations[0])} "
                  f"with shape {activations[0].shape} "
                  f"[{activations[0].dtype}]")
            for index, values in enumerate(activations):
                print(f"dl-activation:  [{index}]: {values.shape}")
                results[layers[index]][index:index+len(batch)] = values
            print("dl-activation: "
                  f"batch finished in {batch_duration*1000:.0f} ms.")
    except KeyboardInterrupt:
        # print(f"error procesing {data.filename} {data.shape}")
        print("Keyboard interrupt")
        # self.output_status(top, end='\n')
    except InterruptedError:
        print("Interrupted.")
    finally:
        print("dl-activation: finished processing")
        # signal.signal(signal.SIGINT, original_sigint_handler)
        # signal.signal(signal.SIGQUIT, original_sigquit_handler)


def console_show_activations(activations, title: str=None) -> None:
    if title is not None:
        print(title)
    if isinstance(activations, list):
        for index, activation in enumerate(activations):
            print(f" ({index}) {activation.shape}")
    elif isinstance(activations, dict):
        for layer_id, activation in activations.items():
            print(f" ({layer_id}) {activation.shape}")

#
# Different options to get activations for an image:
#

def demo_image_activations1(network, image) -> None:
    activations1 = network.get_activations(image)  # list
    show_activations(activations1, title="Demo 1: "
                     "network.get_activations(image): {type(activations1)}")


def demo_image_activations2(network, image) -> None:
    tool = ActivationTool(network)
    activations2 = tool(image)  # dict
    show_activations(activations2, title="Demo 2: "
                     f"ActivationTool(network)(image): {type(activations2)}")


def demo_image_activations3(network, image) -> None:
    tool = ActivationTool(network)
    data = Data(image)
    activations3 = tool(data)  # dict
    show_activations(activations3, title="Demo 3: "
                     "ActivationTool(network)(data):")


def demo_image_activations4(network, image) -> None:
    tool = ActivationTool(network)
    data = Data(image)
    tool.apply(data)
    activations4 = tool.data_activations(data)  # dict
    show_activations(activations4, title="Demo 4: "
                     "ActivationTool(network).apply(data):")


def demo_image_activations5(network, image) -> None:
    tool = ActivationTool(network)
    worker = ActivationWorker(tool=tool)
    data = Data(image)
    worker.work(data, busy_async=False)
    activations5 = tool.data_activations(data)  # dict
    show_activations(activations5, title="Demo 5: "
                     "ActivationWorker(tool).work(data):")


def demo_iterate_activations(network: Network, datasource: Datasource,
                             action=None) -> None:
    """Iterate over a :py:class:`Datasource` and compute network activations.

    Arguments
    ---------
    network:
        The network from which activation values are obtained.
    datasource:
        The :py:class:`Datasource` providing data.
    action:
        A function to execute for every iteration step. The function is
        fed with two arguments: the numerical index of the datasource
        and (a dictionary containing) the activation values.
    """
    def print_activations(index, activations) -> None:
        print(f"Activations for index={idx}:",
              ", ".join(f"{l}: {a.shape}" for l, a in activations.items()))
        
    if action is None:
        action = print_activations

    worker = ActivationWorker(network=network)
    for idx, activations in enumerate(worker.iterate_activations(datasource)):
        action(index, activations)


def demo_store_activations(network: Network, datasource: Datasource) -> None:
    """Store activation values observed in a :py:class:Network when
    applied to a :py:class:`Datasource`.

    This method uses the `ActivationsArchiveNumpy` class which stores
    activation values using the numpy memmap mechanism.
    """
    archive = ActivationsArchiveNumpy(network=network, datasource=datasource,
                                      store=True)
    with archive:
        archive.info()
        try:
            display = get_display()
            display.blocking = None
            for index in tqdm(range(archive.valid, archive.total),
                              initial=archive.valid, total=archive.total):
                data = datasource[index]
                display.show(data)
                archive += network.get_activations(data)
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
        finally:
            print(f"End: {archive.valid}/{archive.total}")

def demo_load_activations(network: Union[Network, str],
                          datasource: Union[Datasource, str]) -> None:
    """Iterate over activation values stored in an
    :py:class:`ActivationsArchive`.

    For using such an archive, neither the `Network`, nor the
    `Datasource` has to be instantiated - it is sufficient to provide
    their keys, from which archive name is derived (remark: it should
    also be also possible to provide the path to the archive directly,
    then even those keys are not required, but that is not implemented
    yet).

    """
    with ActivationsArchiveNumpy(network=network,
                                 datasource=datasource) as archive:
        for activations in archive:
            print(type(activations))

    
def demo_top_activations(network: Network, datasource: Datasource) -> None:
    worker = ActivationWorker(network=network)
    # FIXME[old]
    # worker._init_layer_top_activations()
    for idx, activations in enumerate(worker.iterate_activations(datasource)):
        print(f"({idx})")
        # worker._update_layer_top_activations()

def demo_store_top_activations(network: Network, datasource: Datasource,
                               top: int = 9) -> None:
    """Store top activation values in a :py:class:`TopActivations` archive.
    Such an archive does not store all activation values but just
    the `top` activations for 
    """
    archive = TopActivations(network=network, datasource=datasource,
                             top=top, store=True)
    with archive:
        archive.info()
        try:
            for index in tqdm(range(archive.valid, archive.total),
                              initial=archive.valid, total=archive.total):
                archive += network.get_activations(datasource[index])
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
        finally:
            print(f"End: {archive.valid}/{archive.total}")




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

    network, layers = NetworkArgparse.network(parser, args, layers=True)
    network.summary(layers=layers)

    datasource = DatasourceArgparse.datasource(parser, args)

    # FIXME[hack]: GUI display
    global show_activations
    if args.gui:
        global app, window, activationview
        app = QApplication([])

        window = QMainWindow()
        activationview = QActivationView()
        show_activations = gui_show_activations
    else:
        show_activations = console_show_activations

    if args.iterate:
        demo_iterate_activations(network, datasource)
    elif args.store:
        demo_store_activations(network, datasource)
    elif args.archive:
        demo_load_activations(network, datasource)
    elif args.top:
        demo_top_activations(network, datasource)
    elif args.store_top:
        demo_store_top_activations(network, datasource)

    elif datasource is not None:
        #
        # loop over the dataset
        #

        # FIXME[bug]: error in ActivationWorker
        extract_activations1(network, datasource, layers=layers)

    else:
        # image_file = 'images/elephant.jpg'
        for image_file in args.image:
            image = imread(image_file)
            # FIXME[bug]: error in ActivationWorker
            demo_image_activations1(network, image)
            demo_image_activations2(network, image)
            demo_image_activations3(network, image)
            demo_image_activations4(network, image)
            demo_image_activations5(network, image)


if __name__ == "__main__":
    main()
