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

# Requirements
#  - numpy
#  - pyqt
#  - pillow=8 (to support .webp files)
#
# Deep learning toolbox (dltb)
#  git clone https://github.com/CogSciUOS/DeepLearningToolbox.git
#  export PYTHONPATH=${PWD}/DeepLearningToolbox:${PYTHONPATH}
#
# Running:
#   python demos/face-labeler.py --directory=/net/projects/scratch/summer/valid_until_31_January_2022/krumnack/childface/clean4/UnifiedFunneled

# standard imports
from argparse import ArgumentParser
import sys

# third-party imports
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QSpinBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout

# toolbox imports
import dltb.argparse as ToolboxArgparse
from dltb.datasource import Datasource
from experiments.childface import ChildFaces
from qtgui.widgets.data import QDataInfoBox
from qtgui.widgets.image import QImageView, QMultiImageView
from qtgui.widgets.scroll import QOrientedScrollArea
from qtgui.utils import protect


class QFaceLabeler(QWidget):

    def __init__(self, datasource: Datasource = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._datasource = datasource
        self._labels = list(datasource.labels())
        self._faces = []

        self.dataView = QDataInfoBox()
        self.imageView = QImageView()
        self.multiImageView = QMultiImageView(grid=(None, 5))
        self.multiImageView.currentImageChanged.connect(self.onImageChanged)

        self.spinBox = QSpinBox()
        self.spinBox.setMinimum(1)
        self.spinBox.setMaximum(len(self._labels))
        self.spinBox.valueChanged.connect(self.onSpinboxChanged)
        self.label = QLabel()

        self.help = QLabel("Help: A=All are valid; N=none is valid")
        self.help.setWordWrap(True)

        layout = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(self.spinBox)
        row.addWidget(self.label)
        row.addStretch()
        layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.multiImageView)
        column = QVBoxLayout()
        column.addWidget(self.dataView)
        column.addWidget(self.imageView)
        row.addLayout(column)
        layout.addLayout(row)
        layout.addStretch()
        self.setLayout(layout)

    @protect
    def onSpinboxChanged(self, value: int) -> None:
        index = value - 1
        label = self._labels[index]
        self.label.setText(self._labels[index])
        self._faces = [self._datasource.get_data(filename=filename)
                       for filename in self._datasource.faces(label)]
        self.multiImageView.setImages(self._faces)

    @protect
    def onImageChanged(self, index: int) -> None:
        print(f"selected face {index}")
        if 0 <= index < len(self._faces):
            self.dataView.setData(self._faces[index])
        else:
            self.dataView.setData(None)
            self.imageView.setData(None)


def main2():
    parser = ArgumentParser(description='Face labeling tool')
    parser.add_argument('--directory', type=str,
                        help="path to the data directory "
                        "(including 'UnifiedFunneled')")

    ToolboxArgparse.add_arguments(parser)

    args = parser.parse_args()
    ToolboxArgparse.process_arguments(parser, args)

    datasource = ChildFaces(directory=args.directory, prepare=True)
    print(datasource)
    from dltb.util.image import imread
    image = imread('/net/projects/scratch/summer/valid_until_31_January_2022/krumnack/childface/clean4/UnifiedFunneled/AaronWolff/New/1091847.jpg', module='imageio')

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

    face_labeler = QFaceLabeler(datasource)
    image_view = face_labeler.multiImageView
    image_view.setImages(faces)

    # This will run the graphical interface
    gui = face_labeler
    gui.setMaximumSize(screensize.width() - 100, screensize.height() - 100)

    gui.show()
    rc = app.exec()

    print(f"Main: exiting gracefully (rc={rc}).")
    sys.exit(rc)


if __name__ == "__main__":
    main2()
