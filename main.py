#!/usr/bin/env python

import sys
import argparse

from PyQt5.QtWidgets import QApplication
import numpy as np

from gui import App
from list_of_funnctions import KerasNetwork

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural network analysation.')
    parser.add_argument("model", help = 'filename of model to use',
                        default = 'my2c2d_model.h5')
    parser.add_argument("data", help = 'filename of dataset to visualize',
                        default = 'data.npy')
    args = parser.parse_args()

    network = KerasNetwork(args.model)
    data = np.load(args.data)
    data=data.reshape(data.shape[0],data.shape[1],data.shape[2],1)

    app = QApplication(sys.argv)
    ex = App(network, data)
    sys.exit(app.exec_())
