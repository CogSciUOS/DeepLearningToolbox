#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python demos/face-labeler.py --directory='/net/projects/scratch/summer/valid_until_31_January_2022/krumnack/childface/clean4/UnifiedFunneled2' --debug __main__

# FIXME[todo]:
# - move face-label out auf demos/ split in separate files ...
# - read in original metadata (including age)
# - nice display for metadata
# - store new metadata
# - non hardcoded filename handling



"""The Labeled Children Faces in the Wild dataset.

A dataset compiled by the Computer Vision Group of the IKW.

from dltb.thirdparty.datasource.lfw import LabeledFacesInTheWild
lfw = LabeledFacesInTheWild()
lfw.prepare()
lfw.sklearn is None

"""
# standard imports
from typing import Iterable
import os
import json
import logging

# toolbox imports
from dltb.base.image import Image
from dltb.datasource import ImageDirectory

# logging
LOG = logging.getLogger(__name__)

print(__name__)

# FIXME[hack]
DIRECTORY = '/space/home/ulf/data/children/clean4/UnifiedFunneled'
DIRECTORY = '/net/projects/scratch/summer/valid_until_31_January_2022/krumnack/childface/clean4/UnifiedFunneled2'

class ChildFaces(ImageDirectory):
    """The directory clean4 contains the 4th clean stage.

    Unified/NAME/new

    UnifiedFunneled/NAME/new-img-ID-faceN.jpg
    UnifiedFunneled/NAME/new-img-ID-faceN.json
    """
    def __init__(self, key: str = None, directory: str = None,
                 **kwargs) -> None:
        """Initialize the Labeled Faces in the Wild (LFW) dataset.

        Parameters
        ----------
        direcotry: str
            The path to the children faces root directory. This directory
            should contain the subdirectories, each holding images
            of one person.
        """
        key = key or "childface4"
        if directory is None:
            directory = DIRECTORY
        suffix = ('gif', 'jpeg', 'jpg', 'png', 'webp')
        description = "Children Faces in the Wild"
        super().__init__(key=key, directory=directory, suffix=suffix,
                         description=description,
                         label_from_directory='name',
                         **kwargs)

    def __str__(self) -> str:
        return super().__str__() + f" with {len(self._labels)} labels"

    def _prepare(self) -> None:
        super()._prepare()

        # extract the set of labels from filenames
        # (alternative approach: read in the directory names)
        labels = set()
        for name in self._filenames:
            labels.add(name.split('/', 1)[0])
        self._labels = sorted(labels)

    def labels(self) -> Iterable[str]:
        return self._labels

    def faces(self, label: str) -> Iterable[str]:
        """Iterate the face images for a given identity.
        """
        prefix = label + '/'
        for filename in self._filenames:
            if filename.startswith(prefix):
                yield filename

    def _get_data(self, data: Image, **kwargs) -> None:
        """Get data from this :py:class:`Datasource`.
        """
        super()._get_data(data, **kwargs)
        self.load_metadata(data)

    def load_metadata(self, data: Image) -> None:
        """
        """
        filename_meta = data.filename.rsplit('.', maxsplit=1)[0] + '.json'
        suffix = '2' if os.path.isfile(filename_meta + '2') else ''

        filename = filename_meta + suffix
        if os.path.isfile(filename):
            LOG.debug("Loading meta file '%s'", filename)
            with open(filename) as infile:
                meta = json.load(infile)
                # image:        path to the image file
                # source:       path to the source file
                # dataset:      the dataset from which this images was taken
                # boundingbox:  bounding bos of the face in the original image
                # id:           the class label
                data.add_attribute('image', meta['image'])
                data.add_attribute('source', meta['source'])
                data.add_attribute('dataset', meta['dataset'])
                data.add_attribute('boundingbox', meta['boundingbox'])
                data.add_attribute('id', meta['id'])
            data.add_attribute('metafile', filename_meta)
        else:
            LOG.debug("No meta file for data (tried '%s')", filename)

    def write_metadata(self, data: Image) -> None:
        """
        """
        if hasattr(data, 'metafile') and hasattr(data, 'valid'):
            suffix = '2'
            meta = {
                'image': data.filename,
                'source': data.source,
                'dataset': data.dataset,
                'boundingbox': data.boundingbox,
                'id': data.id,
                'valid': data.valid,
                'age': data.age
            }
            filename = data.metafile + suffix
            LOG.debug("Writing new meta file '%s'", filename)
            with open(filename, 'w') as outfile:
                json.dump(meta, outfile)
        else:
            LOG.debug("Not writing new meta file (metafile: %s, valid: %s).",
                      hasattr(data, 'metafile'), hasattr(data, 'valid'))


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
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSlot
from PyQt5.QtGui import QKeyEvent, QImage, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QSpinBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QSizePolicy

# toolbox imports
import dltb.argparse as ToolboxArgparse
from dltb.base.image import Imagelike, BoundingBox, Region
from dltb.datasource import Datasource
from dltb.util.image import imread
#from experiments.childface import ChildFaces
from qtgui.widgets.data import QDataInfoBox
from qtgui.widgets.image import QImageView, QMultiImageView
from qtgui.widgets.scroll import QOrientedScrollArea
from qtgui.utils import protect

class QMultiFaceView(QMultiImageView):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        line_width = 2
        self._bluePen = QPen(Qt.blue)
        self._bluePen.setWidth(line_width)
        self._redPen = QPen(Qt.red)
        self._redPen.setWidth(2 * line_width)
        self._greenPen = QPen(Qt.green)
        self._greenPen.setWidth(line_width)

    def setImages(self, images: Iterable[Imagelike]) -> None:
        # FIXME[hack]: we would need a better mechanism to incorporate
        # metadata
        images = list(images)
        super().setImages(images)
        for index, (qimage, image) in enumerate(zip(self._qimages, images)):
            qimage.dataset = getattr(image, 'dataset', None)
            if hasattr(image, 'age'):
                self.setAge(index, image.age)

    def age(self, index: int = None) -> int:
        """The age label for an image in this :py:class:`QMultiFaceView`.

        Arguments
        ---------
        index:
            An index identifying the image to be checked. If no index is
            provided, the currently selected image is used. If no index
            can be determined, the method will return `None`.

        Result
        ------
        age:
            The age (as int or tuple of ints) if an age label exists for
            the image, otherwise `None`.
        """
        if index is None:
            index = self.currentImage()
        if not 0 <= index < len(self._qimages):
            return False  # no index could be determined

        if self._regions is None:
            return getattr(self._qimages[index], 'age', None)

        region = self._regions[index]
        return getattr(region, 'age', None)

    def setAge(self, age: int, index: int = None) -> None:
        """Set the age of an image an image in this :py:class:`QMultiFaceView`.

        Arguments
        ---------
        index:
            An index identifying the image for which the age is to be set.
            If no index is provided, the currently selected image is used.
            If there is no such index, the method will do nothing.
        """
        if index is None:
            index = self.currentImage()
        if not self._qimages or not 0 <= index < len(self._qimages):
            return

        if self._regions is None:
            setattr(self._qimages[index], 'age', age)
        else:
            region = self._regions[index]
            region.age = age
        self.annotationsChanged.emit(index)
        self.update()

    @protect
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Process `QKeyEvent`s. Use cursor keys to change the currently
        selected image.

        Arguments
        ---------
        event:
            The key event to process.
        """
        key = event.key()

        if key in (Qt.Key_A, Qt.Key_1):
            self.setAge((0,4))
        elif key in (Qt.Key_B, Qt.Key_2):
            self.setAge((3,7))
        elif key in (Qt.Key_C, Qt.Key_3):
            self.setAge((6,12))
        elif key in (Qt.Key_D, Qt.Key_4):
            self.setAge((10,20))
        elif key in (Qt.Key_E, Qt.Key_5):
            self.setAge((18,100))
        elif key == Qt.Key_Backspace:
            self.setAge(None)
        else:
            super().keyPressEvent(event)
            
    def _paintImage(self, painter: QPainter, index: int,
                    qimage: QImage, rect: QRect) -> None:
        super()._paintImage(painter, index, qimage, rect)

        # add the age label
        age = self.age(index)
        ageText = "None" if age is None else f"{age}"
        pen = self._bluePen if age is None else self._greenPen
        painter.setPen(pen)
        painter.drawText(rect, Qt.AlignHCenter | Qt.AlignBottom, ageText)

        dataset = qimage.dataset
        metaText = "No Meta" if dataset is None else dataset
        pen = self._redPen if dataset is None else self._greenPen
        painter.setPen(pen)
        painter.drawText(rect, Qt.AlignHCenter | Qt.AlignTop, metaText)


class QFaceLabeler(QWidget):

    def __init__(self, datasource: Datasource = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._datasource = datasource
        self._labels = list(datasource.labels())
        self._faces = []
        self._index = None  # index of the currently selected image

        self.dataView = QDataInfoBox()
        self.imageView = QImageView()
        self.multiImageView = QMultiFaceView(grid=(None, 5))
        self.multiImageView.currentImageChanged.connect(self.onImageChanged)

        self.multiImageScroller = \
            QOrientedScrollArea(orientation=Qt.Vertical)
        self.multiImageScroller.setWidget(self.multiImageView)
        self.multiImageScroller.setWidgetResizable(True)
        self.multiImageScroller.setSizePolicy(QSizePolicy.Fixed,
                                              QSizePolicy.Expanding)
        
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
        row.addWidget(self.multiImageScroller)
        column = QVBoxLayout()
        column.addWidget(self.dataView)
        column.addStretch()
        column.addWidget(self.imageView)
        row.addLayout(column)
        layout.addLayout(row)
        #layout.addStretch()
        self.setLayout(layout)

        self.showPerson()

    def storeMetadata(self) -> None:
        if self._index is not None and 0 <= self._index < len(self._faces):
            data = self._faces[self._index]
            age = self.multiImageView.age(self._index)
            invalid = self.multiImageView.invalid(self._index)
            data.add_attribute('age', age)
            data.add_attribute('valid', not invalid)
            self._datasource.write_metadata(data)

    def showPerson(self, index: int = 0) -> None:
        label = self._labels[index]
        if self._index is not None:
            self.storeMetadata()

        self.label.setText(self._labels[index])
        self._faces = [self._datasource.get_data(filename=filename)
                       for filename in self._datasource.faces(label)]
        self.multiImageView.setImages(self._faces)

    @protect
    def onSpinboxChanged(self, value: int) -> None:
        self.showPerson(value - 1)

    @protect
    def onImageChanged(self, index: int) -> None:
        print(f"selected face {index} of {len(self._faces)}")
        self.storeMetadata()
        if 0 <= index < len(self._faces):
            self._index = index
            data = self._faces[index]
            self.dataView.setData(data)
            if hasattr(data, 'source'):
                filename = data.source
                filename = filename.replace('E:\\clean4', '/net/projects/scratch/summer/valid_until_31_January_2022/krumnack/childface/clean4')
                filename = filename.replace('\\', '/')
                image = Image(image=filename)
                bbox = data.boundingbox
                bbox = BoundingBox(x1=bbox[0], y1=bbox[1],
                                   x2=bbox[2], y2=bbox[3])
                self.imageView.setData(image)
                self.imageView.addRegion(Region(bbox))
            else:
                self.imageView.setData(None)

            position = self.multiImageView.imagePosition(index)
            if position is not None:
                imageSize = self.multiImageView.imageSize()
                spacing = self.multiImageView.spacing()
                xmargin = (imageSize.width() + spacing) // 2
                ymargin = (imageSize.height() + spacing) // 2
                self.multiImageScroller.ensureVisible(position.x(),
                                                      position.y(),
                                                      xmargin, ymargin)
        else:
            self._index = None
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
    #image_view = face_labeler.multiImageView
    #image_view.setImages(faces)

    # This will run the graphical interface
    gui = face_labeler
    gui.setMaximumSize(screensize.width() - 300, screensize.height() - 300)

    gui.show()
    rc = app.exec()

    print(f"Main: exiting gracefully (rc={rc}).")
    sys.exit(rc)


if __name__ == "__main__":
    main2()
