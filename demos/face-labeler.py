#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# standard imports
from argparse import ArgumentParser
import sys

# third-party imports
from PyQt5.QtWidgets import QApplication

# toolbox imports
import dltb.argparse as ToolboxArgparse
from dltb.datasource import Datasource
from qtgui.widgets.data import QDataInspector


def main():
    """Start the program.
    """

    app = QApplication([])

    data_inspector = QDataInspector(datasource_inspector=False)
    rec = QApplication.desktop().screenGeometry()
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


# standard imports
from argparse import ArgumentParser
import sys

# third-party imports
from PyQt5.QtWidgets import QApplication

# toolbox imports
import dltb.argparse as ToolboxArgparse
from dltb.datasource import Datasource
from experiments.childface import ChildFaces
from qtgui.widgets.data import QDataInspector
from qtgui.widgets.image import QMultiImageView


def main2():
    parser = ArgumentParser(description='Face labelint tool')

    ToolboxArgparse.add_arguments(parser)

    args = parser.parse_args()
    ToolboxArgparse.process_arguments(parser, args)

    datasource = ChildFaces(prepare=True)
    print(datasource)
    print(len(datasource._filenames), "filenames")
    print(len(list(datasource.labels())), "labels")

    data = datasource[0]
    print(data)

    datasource.load_metadata(data)

    data.add_attribute('valid', True)
    datasource.write_metadata(data)

    labels = list(datasource.labels())
    label = labels[0]

    faces = [datasource.get_data(filename=filename)
             for filename in datasource.faces(label)]
    print(faces)

    #
    # Run the graphical user interface
    #
    app = QApplication([])
    screensize = QApplication.desktop().screenGeometry()

    image_view = QMultiImageView(grid=(3, None))
    image_view.setImages(faces)

    # This will run the graphical interface
    gui = image_view
    gui.setMaximumSize(screensize.width() - 100, screensize.height() - 100)

    gui.show()
    rc = app.exec()

    print(f"Main: exiting gracefully (rc={rc}).")
    sys.exit(rc)


if __name__ == "__main__":
    main2()
