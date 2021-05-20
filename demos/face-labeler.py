#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

from PyQt5.QtWidgets import QApplication, QLabel

from dltb.datasource import Datasource
from qtgui.widgets.data import QDataView


def main():
    """Start the program.
    """

    app = QApplication([])

    data_view = QDataView()
    data_view._multiImageView.setGrid((None, 10))
    data_view.show()

    datasource = Datasource['widerface']
    datasource.prepare()
    data_view.setData(datasource[45])

    # This will also run the graphical interface
    rc = app.exec()

    print(f"Main: exiting gracefully (rc={rc}).")
    sys.exit(rc)


if __name__ == '__main__':
    main()
