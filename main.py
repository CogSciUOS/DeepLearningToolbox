#!/usr/bin/env python3
'''
A framework-agnostic visualisation tool for deep neural networks

.. moduleauthor:: Rüdiger Busche, Petr Byvshev, Ulf Krumnack, Rasmus Diederichsen

'''

import sys
import argparse
import os

from PyQt5.QtWidgets import QApplication

from qtgui.mainwindow import DeepVisMainWindow


# FIXME[todo]: speed up initialization by only loading frameworks
# that actually needed
# FIXME[todo]: also provide code to check if the framework is available
#from network.keras import Network as KerasNetwork
#from network.torch import Network as TorchNetwork
from network import Network


def keras(backend: str, cpu: bool, model_file: str='models/example_keras_mnist_model.h5') -> Network: # KerasNetwork
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


def torch(cpu: bool, model_file: str, net_class: str, parameter_file: str, input_shape: tuple) -> Network: # TorchNetwork
    '''
    Visualise a Torch-based network

    .. error:: Torch network currently does not work.

    Parameters
    ----------
    cpu         :   bool
                    Whether to use only cpu, not gpu
    model_file  :   str
                    Filename where the model is defined (a Python file with a
                    :py:class:`torch.nn.Module` sublcass)
    net_class   :   str
                    Name of the model_file class (see ``model_file``)

    parameter_file  :   str
                        Name of the file storing the model weights (pickled torch weights)
    input_shape     :   tuple
                        Shape of the input images

    Returns
    -------
    TorchNetwork
        The concrete network instance to visualise

    '''
    # FIXME[todo]: Fix errors when running torch network
    return Network.load('network.torch',
                        model_file, parameter_file, net_class=net_class,
                        input_shape=input_shape, use_cuda=not cpu)


def main():
    '''Start the program.'''

    from datasources import Predefined
    datasets = Predefined.get_data_source_ids()

    parser = argparse.ArgumentParser(
        description='Visual neural network analysis.')
    parser.add_argument('--model', help='Filename of model to use',
                        default='models/example_keras_mnist_model.h5')
    parser.add_argument('--data', help='filename of dataset to visualize')
    parser.add_argument('--datadir', help='directory containing input images')
    if (len(datasets) > 0):
        parser.add_argument('--dataset', help='name of a dataset',
                            choices=datasets,
                            default=datasets[0]) # 'mnist'
    parser.add_argument('--framework', help='The framework to use.',
                        choices=['keras-tensorflow', 'keras-theano', 'torch'],
                        default='keras-tensorflow')
    parser.add_argument('--cpu', help='Do not attempt to use GPUs',
                        action='store_true',
                        default=False)
    args = parser.parse_args()

    #
    # network2: AlexNet trained on ImageNet data (TensorFlow)
    #
    from network.tensorflow import Network as TensorFlowNetwork
    checkpoint = os.path.join('models', 'example_tf_alexnet',
                              'bvlc_alexnet.ckpt')
    network2 = TensorFlowNetwork(checkpoint=checkpoint)
    from datasources.imagenet_classes import class_names2
    network2.set_output_labels(class_names2)

    #
    # network: dependes on the selected framework
    #
    if True:
        network = None
    elif args.framework.startswith('keras'):
        dash_idx = args.framework.find('-')
        backend  = args.framework[dash_idx + 1:]
        network  = keras(backend, args.cpu, model_file=args.model)

    elif args.framework == 'torch':
        # FIXME[hack]: provide these parameters on the command line ...
        net_file       = 'models/example_torch_mnist_net.py'
        net_class      = 'Net'
        parameter_file = 'models/example_torch_mnist_model.pth'
        input_shape    = (28, 28)
        network = torch(args.cpu, net_file, net_class,
                        parameter_file, input_shape)
    else:
        network = None
        
    from datasources import DataDirectory
    from model import Model
    app        = QApplication(sys.argv)
    model      = Model(network2)
    #model.addNetwork(network2)

    if args.data:
        source = Predefined.get_data_source(args.data)
    elif args.dataset:
        source = Predefined.get_data_source(args.dataset)
    elif args.datadir:
        source = DataDirectory(args.datadir)
    model.setDataSource(source)

    mainWindow = DeepVisMainWindow(model)
    mainWindow.show()

    rc = app.exec_()
    sys.exit(rc)


if __name__ == '__main__':
    main()
