#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''A framework-agnostic visualisation tool for deep neural networks

.. moduleauthor:: Rüdiger Busche, Petr Byvshev, Ulf Krumnack, Rasmus
Diederichsen

'''

import sys
import os
import logging

# Begin checking for modules

# FIXME[todo]: check the packages that we actually need - This is not
# a good place to do here, and we need a more general concept to do
# this in a consistent way.

# Python
#
# We need at least python 3.6, as we make (heavy) use of the following
# features:
#  - PEP 498: Formatted string literals
#    [https://docs.python.org/3/whatsnew/3.6.html#whatsnew36-pep498]
#  - PEP 526: Syntax for variable annotations
#    [https://docs.python.org/3/whatsnew/3.6.html#whatsnew36-pep526]
if sys.version_info < (3, 6):
    logging.fatal("You need at least python 3.6, but you are running python "
                  + ".".join([str(_) for _ in sys.version_info]))
    sys.exit(1)

try:
    
    # Version checking.
    # We use the 'packaging.version' module, which is a popular
    # third-party module (e.g. used by setuptools) and which
    # is conformant to PEP 404.
    # An alternative would be to use the nowadays outdated
    # 'distutils.version' module, which is build in, but undocumented
    # and conformant only to the superseeded PEP 386.
    import packaging.version
    
    # new versions of tensorflow require recent versions of protobuf:
    #  - I experienced problems with tensorflow 1.12.0 with protobuf 3.6.1:
    #    tensorflow/python/keras/backend/__init__.py", line 22,
    #      from tensorflow.python.keras._impl.keras.backend import abs
    #    -> ImportError: cannot import name 'abs'
    #    However reinstalling both packages resolved the problem ...
    #    maybe not a version issue?
    
except ModuleNotFoundError as e:
    logging.fatal(str(e))
    sys.exit(1)

# End checking for modules


# FIXME[hack]: avoid a lot of debug output from matplotlib ...
import matplotlib

import util
import toolbox

#
# Changing global logging Handler
#
print("!!!!!!!!!!!!!!!! Changing global logging Handler !!!!!!!!!!!!!!!!!!!!")
logging.basicConfig(level=logging.DEBUG)

root_logger = logging.getLogger()
root_logger.handlers = []
logRecorder = util.RecorderHandler()
root_logger.addHandler(logRecorder)

# local loggger
logger = logging.getLogger(__name__)
logger.debug(f"Effective debug level: {logger.getEffectiveLevel()}")

    

from util import addons

from PyQt5.QtWidgets import QApplication

from qtgui.mainwindow import DeepVisMainWindow


# FIXME[todo]: speed up initialization by only loading frameworks
# that actually needed
# FIXME[todo]: also provide code to check if the framework is available

from network import Network
# from network.keras import Network as KerasNetwork
# from network.torch import Network as TorchNetwork


def keras(backend: str, cpu: bool,
          model_file: str='models/example_keras_mnist_model.h5') -> Network:
    # actually: KerasNetwork
    '''
    Visualise a Keras-based network

    Parameters
    ----------
    backend     :   str
                    Name of the Keras backend
                    (currently only "tensorflow" or "theano")
    cpu         :   bool
                    Whether to use only cpu, not gpu
    model_file       :   str
                    Filename where the model_file is located (in hdf5)

    Returns
    -------
    network.network.Network
        The concrete network instance to visualise

    Raises
    ------
    RuntimeError
        In case of unknown backend

    '''
    if backend == 'tensorflow':
        return Network.load('network.keras_tensorflow',
                            model_file=model_file)
    elif backend == 'theano':
        return Network.load('network.keras_theano',
                            model_file=model_file)
    else:
        raise RuntimeError('Unknown backend {backend}')


def torch(cpu: bool, model_file: str, net_class: str, parameter_file: str,
          input_shape: tuple) -> Network:  # actually: TorchNetwork
    '''
    Visualise a Torch-based network

    .. error:: Torch network currently does not work.

    Parameters
    ----------
    cpu : bool
        Whether to use only cpu, not gpu

    model_file : str
        Filename where the model is defined (a Python file with a
        :py:class:`torch.nn.Module` sublcass)

    net_class : str
        Name of the model_file class (see ``model_file``)

    parameter_file : str
        Name of the file storing the model weights (pickled torch weights)

    input_shape : tuple
        Shape of the input images

    Returns
    -------
    network: TorchNetwork
        The concrete network instance to visualize.

    '''
    # FIXME[todo]: Fix errors when running torch network
    return Network.load('network.torch',
                        model_file, parameter_file, net_class=net_class,
                        input_shape=input_shape, use_cuda=not cpu)


def initializeToolbox(args, gui):
    try:
        from datasources import DataDirectory
        from model import Model
        from tools.am import Engine as AMEngine
        from tools.am import Config as AMConfig

        model = Model()
        gui.setModel(model)

        am_engine = AMEngine(model, AMConfig())
        gui.setMaximizationEngine(am_engine)

        if addons.use('lucid'):
            from tools.lucid import Engine as LucidEngine

            lucid_engine = LucidEngine()
            # FIXME[hack]
            lucid_engine.load_model('InceptionV1')
            lucid_engine.set_layer('mixed4a', 476)
            gui.setLucidEngine(lucid_engine)

        from datasources import Predefined
        if args.data:
            source = Predefined.get_data_source(args.data)
        elif args.dataset:
            source = Predefined.get_data_source(args.dataset)
        elif args.datadir:
            source = DataDirectory(args.datadir)

        gui.setDataSource(source)

        #
        # network: dependes on the selected framework
        #
        # FIXME[hack]: two networks/models seem to cause problems!
        if args.alexnet:
            #
            # AlexNet trained on ImageNet data (TensorFlow)
            #
            logger.debug("alexnet: import tensorflow")
            from network.tensorflow import Network as TensorFlowNetwork
            checkpoint = os.path.join('models', 'example_tf_alexnet',
                                      'bvlc_alexnet.ckpt')
            logger.debug("alexnet: TensorFlowNetwork")
            network = TensorFlowNetwork(checkpoint=checkpoint, id='AlexNet')
            logger.debug("alexnet: prepare")
            network._online()
            logger.debug("alexnet: Load Class Names")
            from datasources.imagenet_classes import class_names2
            network.set_output_labels(class_names2)
            logger.debug("alexnet: Done")

        elif args.framework.startswith('keras'):
            # "keras-tensorflow" or "keras-theaono""
            dash_idx = args.framework.find('-')
            backend = args.framework[dash_idx + 1:]
            network = keras(backend, args.cpu, model_file=args.model)

        elif args.framework == 'torch':
            # FIXME[hack]: provide these parameters on the command line ...
            net_file = 'models/example_torch_mnist_net.py'
            net_class = 'Net'
            parameter_file = 'models/example_torch_mnist_model.pth'
            input_shape = (28, 28)
            network = torch(args.cpu, net_file, net_class,
            parameter_file, input_shape)
        else:
            network = None

        # FIXME[hack]: the @change decorator does not work in different thread
        #model.add_network(network)
        m,change = model.add_network(network)
        m.notifyObservers(change)

    except Exception as e:
        # FIXME[hack]: rethink error handling in threads!
        import traceback
        print(e)
        traceback.print_tb(e.__traceback__)

import argparse

def main():
    '''Start the program.'''


    logger.debug("importing datasources")

    from datasources import Predefined
    datasets = Predefined.get_data_source_ids()

    logger.debug(f"got datesets: {datasets}")

    class UseAddon(argparse.Action):
        """Turn on use of given addon."""
        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            if nargs is not None:
                raise ValueError("nargs not allowed")
            super().__init__(option_strings, dest, nargs=0, **kwargs)
        def __call__(self, parser, namespace, values, option_string=None):
            addons.use(self.dest, True)
            setattr(namespace, self.dest, values)

    parser = argparse.ArgumentParser(
        description='Visual neural network analysis.')
    
    parser.add_argument('--model', help='Filename of model to use',
                        default='models/example_keras_mnist_model.h5')
    parser.add_argument('--data', help='filename of dataset to visualize')
    parser.add_argument('--datadir', help='directory containing input images')
    if (len(datasets) > 0):
        parser.add_argument('--dataset', help='name of a dataset',
                            choices=datasets,
                            default=datasets[0])  # 'mnist'
    parser.add_argument('--framework', help='The framework to use.',
                        choices=['keras-tensorflow', 'keras-theano', 'torch'],
                        default='keras-tensorflow')
    parser.add_argument('--cpu', help='Do not attempt to use GPUs',
                        action='store_true',
                        default=False)
    parser.add_argument('--alexnet', help='Load the AlexNet model',
                        action='store_true',
                        default=False)
    parser.add_argument('--autoencoder',
                        help='Load the autoencoder module (experimental!)',
                        action=UseAddon,
                        default=False)
    parser.add_argument('--advexample',
                        help='Load the adversarial example module (experimental!)',
                        action=UseAddon,
                        default=False)
    args = parser.parse_args()

    util.use_cpu = args.cpu

    #
    # create the actual application
    #
    app = QApplication(sys.argv)

    mainWindow = DeepVisMainWindow()
    mainWindow.show()
    mainWindow.activateLogging(root_logger, logRecorder, True)

    # FIXME[hack]
    mainWindow._runner.runTask(initializeToolbox, args, mainWindow)
    #initializeToolbox(args, mainWindow)

    util.start_timer(mainWindow.showStatusResources)
    rc = app.exec_()
    util.stop_timer()
    sys.exit(rc)


if __name__ == '__main__':
    main()
