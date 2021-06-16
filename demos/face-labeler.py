#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python demos/face-labeler.py --directory='/net/projects/scratch/summer/valid_until_31_January_2022/krumnack/childface/clean4/UnifiedFunneled2' --debug __main__

# FIXME[todo]:
# - move face-label out auf demos/ split in separate files ...
# - read in original metadata (including age)
# - nice display for metadata
# - make sure to clear data/image if no image is selected
# - non hardcoded filename handling

# FIXME[hack]: DIRECTORY: a directory holding the different stages of the
# dataset (clean0, clean2 and clean4)
DIRECTORY = '/space/home/ulf/data/children/'
DIRECTORY = '/net/projects/scratch/summer/valid_until_31_January_2022/krumnack/childface/'

DIRECTORY_CLEAN0 = None  # defaults to DIRECTORY/clean0
DIRECTORY_CLEAN1 = None  # defaults to DIRECTORY/clean1
DIRECTORY_CLEAN2 = None  # defaults to DIRECTORY/clean2
DIRECTORY_CLEAN4 = None  # defaults to DIRECTORY/clean4


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
LOG = logging.getLogger(__name__)  # __name__ is usually '__main__'

class ChildFaces(ImageDirectory):
    """The directory clean4 contains the 4th clean stage.

    Unified/NAME/new

    UnifiedFunneled/NAME/new-img-ID-faceN.jpg
    UnifiedFunneled/NAME/new-img-ID-faceN.json

    There are stages of the dataset:

    clean1/clean0:
        images from AgeDB, FGNET, LargeAgeGap, and IMDB, with
        json files.
    clean2/
        newly collected images
        ExtracedCleaned
        ExtracedCleaned2
        ExtractedFaces
        Leila
        Mats
        Mats_New_Images.zip
        Mats.rar
        Patricia
    clean4/
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
        self._labels = None
        super().__init__(key=key, directory=directory, suffix=suffix,
                         description=description,
                         label_from_directory='name',
                         **kwargs)
        self._clean4 = os.path.dirname(self.directory)
        self._basedir = os.path.dirname(self._clean4)
        self._clean2 = os.path.join(self._basedir, 'clean2')
        self._clean0 = os.path.join(self._basedir, 'clean0')
        print("basedir", self._basedir)
        print("clean4", self._clean4)

    def __str__(self) -> str:
        return (super().__str__() +
                (f" with {len(self._labels)} labels"
                 if self.prepared else " (unprepard)"))

    def _prepared(self) -> bool:
        return (self._labels is not None) and super()._prepared()

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
        filename_meta += '2' if os.path.isfile(filename_meta + '2') else ''
        if os.path.isfile(filename_meta):
            LOG.debug("Loading meta file '%s'", filename_meta)
            with open(filename_meta) as infile:
                meta = json.load(infile)
                # image:        path to the image file
                # source:       path to the source file
                # dataset:      the dataset from which this images was taken
                # boundingbox:  bounding bos of the face in the original image
                # id:           the class label
                data.add_attribute('image', meta['image'])

                source_filename = meta['source']
                source_filename = source_filename.replace('\\', '/')
                source_filename = source_filename.replace('E:', self._basedir)
                data.add_attribute('source', source_filename)
                data.add_attribute('dataset', meta['dataset'])
                if 'boundingbox' in meta:
                    data.add_attribute('boundingbox', meta['boundingbox'])
                if 'id' in meta:
                    data.add_attribute('id', meta['id'])
                data.add_attribute('age', meta.get('age', None))
                data.add_attribute('valid', meta.get('valid', True))
            data.add_attribute('metafile', filename_meta)
        else:
            LOG.debug("No meta file for data (tried '%s')", filename_meta)
            if data.filename.startswith(self.directory):
                filename = data.filename[len(self.directory)+1:]
                parts = filename.split('/')
                label, imagename = parts[0], parts[-1]
                if imagename.startswith('imdb_wiki'):
                    data.add_attribute('dataset', 'imdb_wiki')
                    source_filename = os.path.join(self._clean4, 'Unified',
                                                   filename)
                    if os.path.isfile(source_filename):
                        data.add_attribute('source', source_filename)
                elif len(parts) > 2 and parts[1] == 'New':
                    LOG.warning("New image without meta data: '%s'", filename)
                    data.add_attribute('dataset')
                elif os.path.isfile(os.path.join(self._clean2, 'Patricia',
                                                 filename)):
                    data.add_attribute('dataset', 'Patricia')
                    data.add_attribute('source',
                                       os.path.join(self._clean2, 'Patricia',
                                                    filename))
                else:
                    LOG.warning("Unknown source dataset for '%s'", filename)
                    data.add_attribute('dataset')
            else:
                LOG.warning("Bad filename: '%s' (not in directory '%s')",
                            data.filename, self.directory)
                data.add_attribute('dataset')
            if not filename_meta.endswith('json2'):
                filename_meta += '2'
            data.add_attribute('metafile', filename_meta)
            if not data.has_attribute('age'):
                data.add_attribute('age')
            if not data.has_attribute('valid'):
                data.add_attribute('valid', True)

    def write_metadata(self, data: Image) -> None:
        """
        """
        if hasattr(data, 'metafile') and hasattr(data, 'valid'):
            suffix = '' if data.metafile.endswith('json2') else '2'
            meta = {
                'image': data.filename,
                'source': data.source,
                'dataset': data.dataset,
                'valid': data.valid,
                'age': data.age
            }
            if data.has_attribute('boundingbox'):
                meta['boundingbox'] = data.boundingbox
            if data.has_attribute('id'):
                meta['id'] = data.id
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
#   python demos/face-labeler.py --directory=/net/projects/scratch/summer/valid_until_31_January_2022/krumnack/childface/clean4/UnifiedFunneled2

# standard imports
from argparse import ArgumentParser
import sys

# third-party imports
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSlot
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QImage, QPainter, QPen
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

    AGES = {
        '1': (0,4),
        '2': (3,7),
        '3': (6,12),
        '4': (10,20),
        '5': (18,100)
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        line_width = 2
        self._bluePen = QPen(Qt.blue)
        self._bluePen.setWidth(line_width)
        self._redPen = QPen(Qt.red)
        self._redPen.setWidth(2 * line_width)
        self._greenPen = QPen(Qt.green)
        self._greenPen.setWidth(line_width)

    def invalid(self, index: int = None) -> bool:
        """Check if an image in this :py:class:`QMultiImageView` is invalid.
        Images can have a flag, marking them as valid or invalid.

        Arguments
        ---------
        index:
            An index identifying the image to be checked. If no index is
            provided, the currently selected image is used. If no index
            can be determined, the method will return `False`.

        Result
        ------
        invalid:
            `True`, if the image has been explicitly marked as invalid.
            Otherwise `False`.
        """
        if index is None:
            index = self.currentImage()
        if not 0 <= index < len(self._images):
            return False  # no index could be determined

        if self._regions is None:
            return not getattr(self._images[index], 'valid', True)

        region = self._regions[index]
        return not getattr(region, 'valid', True)

    def invalidate(self, index: int = None) -> None:
        """Invalidate an image in this :py:class:`QMultiImageView`.
        Invalid images will be displayed in a different way as
        valid images.

        Arguments
        ---------
        index:
            An index identifying the image to be invalidated. If no index is
            provided, the currently selected image is used. If there is
            no such index, the method will do nothing.
        """
        if index is None:
            index = self.currentImage()
        if not self._images or not 0 <= index < len(self._images):
            return

        if self._regions is None:
            setattr(self._images[index], 'valid',
                    not getattr(self._images[index], 'valid', False))
        else:
            region = self._regions[index]
            region.valid = not getattr(region, 'valid', False)
        self.annotationsChanged.emit(index)
        self.update()

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
        if not 0 <= index < len(self._images):
            return False  # no index could be determined

        if self._regions is None:
            return getattr(self._images[index], 'age', None)

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
        if not self._images or not 0 <= index < len(self._images):
            return

        if self._regions is None:
            setattr(self._images[index], 'age', age)
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
        elif key == Qt.Key_Delete:
            self.invalidate()
        else:
            super().keyPressEvent(event)
            
    @protect
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Right button press invalidates an image.
        """
        super().mousePressEvent(event)  # selects the current image

        if event.button() == Qt.RightButton:
            self.invalidate()

    @protect
    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """A mouse double click invalidates the current image.
        """
        self.invalidate()

    def _paintImage(self, painter: QPainter, index: int,
                    image: Image, rect: QRect) -> None:
        super()._paintImage(painter, index, image, rect)

        # draw specific decorations
        if self.invalid(index):
            painter.setPen(self._redPen)
            painter.drawLine(rect.topLeft(), rect.bottomRight())
            painter.drawLine(rect.topRight(), rect.bottomLeft())

        # add the age label
        age = self.age(index)
        ageText = "None" if age is None else f"{age}"
        pen = self._bluePen if age is None else self._greenPen
        painter.setPen(pen)
        painter.drawText(rect, Qt.AlignHCenter | Qt.AlignBottom, ageText)

        dataset = image.dataset
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

        self.help = QLabel("Help: " +
                           "; ".join(f"{key}={age}" for key, age in
                                     QMultiFaceView.AGES.items()) + 
                           "; A=All are valid; N=none is valid")
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
        layout.addWidget(self.help)
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
        self.storeMetadata()
        if 0 <= index < len(self._faces):
            self._index = index
            data = self._faces[index]
            self.dataView.setData(data)
            if hasattr(data, 'source'):
                image = Image(image=data.source)
                self.imageView.setData(image)

                if data.has_attribute('boundingbox'):
                    bbox = data.boundingbox
                    bbox = BoundingBox(x1=bbox[0], y1=bbox[1],
                                       x2=bbox[2], y2=bbox[3])
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

def main():
    parser = ArgumentParser(description='Face labeling tool')
    parser.add_argument('--directory', type=str, default=DIRECTORY,
                        help="path to the base directory "
                        "(containing clean* subdirectories)")
    parser.add_argument('--clean0', type=str,
                        help="path to the clean0 directory (optional)")
    parser.add_argument('--clean2', type=str,
                        help="path to the clean2 directory (optional)")
    parser.add_argument('--clean4', type=str,
                        help="path to the clean4 directory "
                        "(including 'UnifiedFunneled2')")

    ToolboxArgparse.add_arguments(parser)

    args = parser.parse_args()
    ToolboxArgparse.process_arguments(parser, args)

    if args.clean0:
        directory_clean0 = args.clean0
    elif DIRECTORY_CLEAN0 is None:
        directory_clean0 = os.path.join(args.directory, 'clean0')
    else:
        directory_clean0 = DIRECTORY_CLEAN0

    if args.clean2:
        directory_clean2 = args.clean2
    elif DIRECTORY_CLEAN2 is None:
        directory_clean2 = os.path.join(args.directory, 'clean2')
    else:
        directory_clean2 = DIRECTORY_CLEAN2

    if args.clean4:
        directory_clean4 = args.clean4
    elif DIRECTORY_CLEAN4 is None:
        directory_clean4 = os.path.join(args.directory, 'clean4')
    else:
        directory_clean4 = DIRECTORY_CLEAN4
    directory_funneled = os.path.join(directory_clean4, 'UnifiedFunneled2')

        
    #
    # Some chanity checks
    #
    if not os.path.isdir(directory_clean4):
        logging.warning("Clean4 directory '%s' does not exist "
                        "- no data to label.",
                        directory_clean4)
        sys.exit(1)

    if not os.path.isdir(directory_clean0):
        logging.warning("Clean0 directory '%s' does not exist "
                        "- some images may not be available.",
                        directory_clean0)

    try:
        # RIFF (little-endian) data, Web/P image, VP8 encoding, 230x230
        problematic_image = os.path.join(directory_clean4, 'UnifiedFunneled',
                                         'AaronWolff', 'New', '1091847.jpg')
        import imageio
        from dltb.util.image import imread
        image = imread(problematic_image, module='imageio')
    except ValueError as ex:
        # imageio: ValueError: Could not find a format to read the
        # specified file in single-image mode
        logging.error("Problematic image file: '%s' (imageio version %s)",
                      problematic_image, imageio.__version__)
        # error: imageio 2.9.0 [conda: pyhd3eb1b0_0 default] (Ubuntu 20.04)
        # error: imageio 2.9.0 [conda: py_0 conda-forge] (Ubuntu 20.04)
        # ok:    imageio 2.6.1 [conda: py36_0 default] (Ubuntu 16.04)
        # print(ex, file=sys.stderr)
        #sys.exit(1)

    #
    # open the data set
    #

    try:
        datasource = ChildFaces(directory=directory_funneled, prepare=True)
    except Exception as ex:
        print(ex, file=sys.stderr)
        sys.exit(1)
    print(datasource)

    data = datasource[0]
    print(data)

    datasource.load_metadata(data)

    # FIXME[test]
    data.add_attribute('valid', True)
    datasource.write_metadata(data)

    labels = list(datasource.labels())
    label = labels[0]

    faces = [datasource.get_data(filename=filename)
             for filename in datasource.faces(label)]
    print(faces)

    #
    # run the graphical user interface
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

    logging.info(f"Main: exiting gracefully (rc=%d).", rc)
    sys.exit(rc)


if __name__ == "__main__":
    main()
