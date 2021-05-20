#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

from PyQt5.QtWidgets import QApplication, QLabel

from dltb.datasource import Datasource
from qtgui.widgets.data import QDataInspector


def main():
    """Start the program.
    """

    app = QApplication([])

    data_inspector = QDataInspector(datasource_inspector=False)
    rec = QApplication.desktop().screenGeometry();
    data_inspector.setMaximumSize(rec.width()-100, rec.height()-100)
    data_view = data_inspector.dataView()
    data_view.multiImageView().setGrid((None, 10))
    data_view.setDataInfoVisible(False)

    datasource = Datasource['widerface']
    datasource.prepare()
    # data_view.setData(datasource[45])
    data_inspector.setDatasource(datasource)

    # This will also run the graphical interface
    data_inspector.show()
    rc = app.exec()

    print(f"Main: exiting gracefully (rc={rc}).")
    sys.exit(rc)


if __name__ == '__main__':
    main()

    
