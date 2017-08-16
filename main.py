#!/usr/bin/env python

import sys
import argparse
import numpy as np

from PyQt5.QtWidgets import QApplication

from qtgui.main import DeepVisMainWindow
from list_of_funnctions import KerasNetwork

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural network analysis.')
    parser.add_argument("--model", help = 'filename of model to use',
                        default = 'models/example_keras_mnist_model.h5')
    parser.add_argument("--data", help = 'filename of dataset to visualize',
                        default = 'mnist')
    args = parser.parse_args()

    network = KerasNetwork(args.model)
    if args.data=='mnist':
        from keras.datasets import mnist
        data = mnist.load_data()[0][0]
    else:
        data = np.load(args.data)
    
    data = data.reshape(data.shape[0],data.shape[1],data.shape[2],1)

    app = QApplication(sys.argv)
    mainWindow = DeepVisMainWindow()
    mainWindow.setNetwork(network, data)
    mainWindow.show()

    sys.exit(app.exec_())
