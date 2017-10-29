#!/usr/bin/env python
"""
A framework-agnostic visualisation tool for deep neural networks

.. moduleauthor Rüdiger Busche, Petr Byvshev, Ulf Krumnack, Rasmus Diederichsen

"""

import sys
import argparse
import os

from PyQt5.QtWidgets import QApplication

from qtgui.main import DeepVisMainWindow


def keras(backend, cpu, model='models/example_keras_mnist_model.h5'):
    """
    Visualise a Keras-based network

    Parameters
    ----------
    backend     :   str
                    Name of the Keras backend
    cpu         :   bool
                    Whether to use only cpu, not gpu
    model       :   str
                    Filename where the model is located (in hdf5)

    Returns
    -------
    network.network.Network
                The concrete network instance to visualise

    Raises
    ------
    RuntimeError
                In case of unknown backend

    """
    # the only way to configure the keras backend appears to be via env vars we
    # thus inject one for this process. Keras must be loaded after this is done
    if backend == 'tensorflow':
        os.environ['KERAS_BACKEND'] = 'tensorflow'
        if cpu:
            # TODO: I don't know if this actually works
            print('Running in CPU-only mode.')
            import tensorflow as tf
            from multiprocessing import cpu_count
            from keras import backend as K
            num_cpus = cpu_count()
            config = tf.ConfigProto(intra_op_parallelism_threads=num_cpus,
                                    inter_op_parallelism_threads=num_cpus,
                                    allow_soft_placement=True,
                                    device_count={'CPU': num_cpus, 'GPU': 0}
                                    )
            session = tf.Session(config=config)
            K.set_session(session)
        from network.keras_tensorflow import Network as KerasTensorFlowNetwork
        network = KerasTensorFlowNetwork(model_file=model)
        return network
    elif backend == 'theano':
        os.environ['KERAS_BACKEND'] = 'theano'
        print("Currently, only TF backend is supported", file=sys.stderr)
        return None
    else:
        raise RuntimeError("Unknown backend %s" % backend)


def torch(cpu, model, net_class, parameter_file, input_shape):
    """
    Visualise a Torch-based network

    Parameters
    ----------
    cpu         :   bool
                    Whether to use only cpu, not gpu
    model       :   str
                    Filename where the model is defined (a Python file with a
                    torch.nn.Module sublcass)
    net_class   :   str
                    Name of the model class (see ``model``)

    parameter_file  :   str
                        Name of the file storing the model weights (pickled
                        torch weights)
    input_shape     :   tuple
                        Shape of the input images

    Returns
    -------
    network.torch.TorchNetwork
                The concrete network instance to visualise

    """
    # TODO Fix errors when running torch network
    from network.torch import Network as TorchNetwork
    return TorchNetwork(model, parameter_file, net_class=net_class,
                        input_shape=input_shape, use_cuda=not cpu)


def main():
    parser = argparse.ArgumentParser(
        description='Visual neural network analysis.')
    parser.add_argument("--model", help='Filename of model to use',
                        default='models/example_keras_mnist_model.h5')
    parser.add_argument("--data", help='filename of dataset to visualize')
    parser.add_argument("--datadir", help='directory containing input images')
    parser.add_argument("--dataset", help='name of a dataset',
                        choices=['mnist'],
                        default='mnist')
    parser.add_argument("--framework", help='The framework to use.',
                        choices=['keras-tensorflow', 'keras-theano', 'torch'],
                        default='keras-tensorflow')
    parser.add_argument("--cpu", help='Do not attempt to use GPUs',
                        action='store_true',
                        default=False)
    args = parser.parse_args()

    if args.framework.startswith('keras'):
        dash_idx = args.framework.find('-')
        backend = args.framework[dash_idx + 1:]
        network = keras(backend, args.cpu, model=args.model)
    elif args.framework == 'torch':
        # FIXME[hack]: provide these parameter on the command line ...
        net_file = "models/example_torch_mnist_net.py"
        net_class = "Net"
        parameter_file = "models/example_torch_mnist_model.pth"
        input_shape = (28, 28)
        network = torch(args.cpu, net_file, net_class,
                        parameter_file, input_shape)

    if network:
        app = QApplication(sys.argv)
        mainWindow = DeepVisMainWindow()
        mainWindow.setNetwork(network)
        if args.data:
            mainWindow.setInputDataFile(args.data)
        elif args.dataset:
            mainWindow.setInputDataSet(args.dataset)
        elif args.datadir:
            mainWindow.setInputDataDirectory(args.datadir)

        mainWindow.show()

        rc = app.exec_()
        sys.exit(rc)


if __name__ == '__main__':
    main()
