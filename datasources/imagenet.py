from . import (DataSource, Labeled, DataDirectory, InputData, Predefined)

# FIXME[todo]: this module datasources.imagenet_classes provides textual
# class names for the 1000 ImageNet classes. The same information is
# is extracted from the ImageNet database by the class ImageNet defined
# in this module.
# We may want to integrate this into the class to
# avoid redundancies or even inconsistencies
from datasources.imagenet_classes import class_names

import os
import re

import logging
logger = logging.getLogger(__name__)

import random
import pickle
from scipy.misc import imread, imresize



# FIXME[todo]: this is currently just a hack, not a full-fleshed
# DataSource!
#  - implement the standard DataSource API
#  - allow for all "sections" (train/valid/test)
#  - automatically provide teh ImageNet class labels

# FIXME[hack]
try:
    from appdirs import AppDirs
    appname = "deepvis"  # FIXME: not the right place to define here!
    appauthor = "krumnack"
    appdirs = AppDirs(appname, appauthor)
except ImportError:
    appdirs = None
    logger.warning(
        "---------------------------------------------------------------\n"
        "info: module 'appdirs' is not installed. We can live without it,\n"
        "but having it around will provide additional features.\n"
        "See: https://github.com/ActiveState/appdirs\n"
        "---------------------------------------------------------------\n")


class ImageNet(DataDirectory, Labeled, Predefined):
    """

    Labels
    ------
    The label information is obtained in different ways, depending on
    the section (train/val/test) of the dataset.

    In the 'train' section, files are grouped in subdirectories,
    one subdirectory per class. Directories are named by WordNet 3.0
    synset identifier. To obtain the actual (numeric) label
    (in the range 0-999), the class stores a mapping
    :py:attr:`_wn_table` mapping synset names to labels.

    In the 'train' section, all 50.000 files are store in one directory,
    with filenames providing a numeric identifier (in the range 1-50,000).
    A mapping from these numeric identifiers to 1-based labels (1-1000)
    is provided by the file 'val_labels.txt'. This information is hold in
    the attribute :py:attr:`_val_labels` in a slightly modified form:
    image identifiers and labels are both shifted to be 0-based, i.e.,
    this list contains at index i the information for image id=i+1,
    and the value obtained is the 0-based numerical label (0-999) 
    """
    _section_ids = ['train', 'test', 'val']

    _section = None

    # list of class labels (class_number -1 ) -> str
    _classes: list = None
    # FIXME[design]: there is also an array _target[], mapping
    # indices to labels. Make this more coherent!
    
    # list of val_labels: (number_from_filename - 1) -> (class_number - 1)
    _val_labels: list = None

    # _appsdir: an Appdirs object
    _appsdir = None

    def __init__(self, prefix=None, section='val', **kwargs):
        super().__init__(id=f"imagenet-{section}",
                         description=f"ImageNet {section}", **kwargs)
        self._imagenet_data = os.getenv('IMAGENET_DATA', '.')
        self._section = section
        self._prefix = os.path.join(self._imagenet_data, self._section)
        self._available = os.path.isdir(self._prefix)
        self._image = None

    def _prepare_data(self):
        logger.info(f"PREPARING ImageNet : {self._prefix}: {self._available}")
        if self._section == 'train':
            # get the list of directory names, i.e.
            # the WordNet 3.0 synsets ('n0......')
            self._synsets = os.listdir(self._prefix)

            # FIXME[hack]: just randomly choose one subdir
            # self._category = random.randint(0, len(self._synsets))
            # self.directory = os.path.join(self._prefix,
            #                               self._synsets[self._category]))

        #
        # get a list of filenames
        #
        if self._appdirs is not None:
            filename = f"imagenet_{self._section}_filelist.p"
            imagenet_filelist = os.path.join(self._appdirs.user_cache_dir,
                                             filename)
            logger.info(f"ImageNet: trying to load filelist from "
                        f"'{imagenet_filelist}")
            if os.path.isfile(imagenet_filelist):
                self._filenames = pickle.load(open(imagenet_filelist, 'rb'))
        else:
            imagenet_filelist = None

        # No filist available yet - read it in
        if self._filenames is None:
            super()._prepare()
            if self._appdirs is not None:
                logger.info(f"Writing filenames to {imagenet_filelist}")
                if not os.path.isdir(self._appdirs.user_cache_dir):
                    os.makedirs(self._appdirs.user_cache_dir)
                pickle.dump(self._filenames, open(imagenet_filelist, 'wb'))

        self.directory = self._prefix

    def _prepare_labels(self):
        if self._section == 'train':
            # get a mapping of synsets to labels
            from .imagenet_classes import wn_table
            self._wn_table = wn_table
            logger.info(f"Length of wn_table: {len(self._wn_table)}")

        elif self._section == 'val':
            #
            # load the image to label mapping
            #
            val_labels_filename = os.path.join(self._imagenet_data,
                                               'val_labels.txt')
            with open(val_labels_filename) as f:
                val_labels = f.readlines()
            self._val_labels = [int(l.strip())-1 for l in val_labels] 

            # load the labels file (classes.txt). 
            # The file contains mapping of labels to synset names
            # and text representation for labels,
            # e.g. "1000:n03255030:dumbbell"
            classes_filename = os.path.join(self._imagenet_data, 'classes.txt')
            with open(classes_filename) as f:
                classes = f.readlines()
            self.set_label_texts([c.strip().split(':')[2] for c in classes])

        logger.info(f"ImageNet is now prepared: {len(self)}")

    def _label_for_filename(self, filename):
        label = None
        if self._section == 'train':
            match_object = re.search('(n[0-9]*)_', filename)
            if match_object is not None:
                synset = self._wn_table.get(match_object.group(1))
        elif (self._section == 'val' and self._val_labels is not None):
            match_object = re.search('ILSVRC2012_val_([0-9]*).JPEG', filename)
            if match_object is not None:
                label = self._val_labels[int(match_object.group(1))-1]
        return label

    def __getitem__(self, index):  # FIXME[old]
        if not self.prepared:
            return InputData(None, None)

        filename = self._filenames[index]
        data = imread(os.path.join(self._dirname, filename))
        category = self._category_for_filename(filename)
        return InputData(data, category)

    def _fetch(self):
        if self._section == 'train':
            category = random.randint(0, len(self._synsets)-1)
            img_dir = os.path.join(self._prefix, self._synsets[category])
            filename = random.choice(os.listdir(img_dir))
            img_file = os.path.join(img_dir, filename)
        elif self._section == 'val':
            filename = random.choice(self._filenames)
            img_file = os.path.join(self._prefix, filename)

        self._image = imread(img_file)

    @property
    def fetched(self):
        return self._image is not None

    def _get_data(self):
        """The actual implementation of the :py:meth:`data` property
        to be overwritten by subclasses.

        It can be assumed that a data point has been fetched when this
        method is invoked.
        """
        return self._image

    def oldrandom(self):  # FIXME[old]
        category = random.randint(0, len(self._synsets)-1)
        img_dir = os.path.join(self._prefix, self._synsets[category])
        self._image = random.choice(os.listdir(img_dir))
        img_file = os.path.join(img_dir, self._image)
        im = imread(img_file)
        return InputData(im, category)

    def _description_for_index(self, index: int) -> str:
        if not self.prepared:
            return ''
        if self._section == 'val' and self._val_labels is not None:
            filename = self._filenames[index]
            match_object = re.search('ILSVRC2012_val_([0-9]*).JPEG', filename)
            if match_object is not None:
                label = self._val_labels[int(match_object.group(1))-1]
                return self._classes[label]
        return ''

    def get_label(self, target: int) -> str:
        label = self._val_labels[target]
        return self._classes[target]

    def check_availability():
        '''Check if this Datasource is available.

        Returns
        -------
        True if the DataSource can be instantiated, False otherwise.
        '''
        return self._prefix

    def download():
        raise NotImplementedError("Downloading ImageNet is "
                                  "not implemented yet.")

    def __str__(self):
        return f"ImageNet ({self._section})"
