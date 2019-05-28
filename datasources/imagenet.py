from . import DataSource, Labeled, Random, DataDirectory, InputData, Predefined

# FIXME[todo]: this module datasources.imagenet_classes provides textual
# class names for the 1000 ImageNet classes. The same information is
# is extracted from the ImageNet database by the class ImageNet defined
# in this module.
# We may want to integrate this into the class to
# avoid redundancies or even inconsistencies
from datasources.imagenet_classes import class_names

import os
import re
import numpy as np

import logging
logger = logging.getLogger(__name__)

import random
import pickle
from scipy.misc import imread



# FIXME[todo]: this is currently just a hack, not a full-fleshed
# DataSource!
#  - implement the standard DataSource API
#  - allow for all "sections" (train/valid/test)
#  - automatically provide teh ImageNet class labels



class ImageNet(DataDirectory, Random, Labeled, Predefined):
    """An interface to the ILSVRC2012 dataset.

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

    Attributes
    ----------

    _section_ids: list
        The available sections of the ILSVRC2012 dataset: 'train',
        'val', and 'test'. Notice that 'test' will provide no labels.

    _section_id: str
        The actual section of the Datasource.


    Labels
    ------
    There are different convention on how to label the ILSVRC2012 dataset.
    The natural convention is to use Synsets, either in form of the
    Synset ID (format='synset', e.g., "n01440764"), a long textual form
    listing all wordforms of the synset (format='text_long',
    e.g. "tench, Tinca tinca"), a short textual form containing just
    the first wordform (format='text', e.g. "tench",
    possibly containing spaces for multiword expressions like "bell pepper").
    However, for neural networks some numerical form is needed.
    Here different forms are in used. The ILSVRC2012 dataset comes with
    a 1-based indexing (format="ilsvrc", e.g. ).
    Caffe simply sorts the synset IDs and indexes them starting from 0
    (format='caffe', maps 0 to n01440764 "tench, Tinca tincan").


    See
    * http://caffe.berkeleyvision.org/gathered/examples/imagenet.html
    * https://github.com/pytorch/vision/issues/484
    """
    _section_ids = ['train', 'val', 'test']

    _section = None

    # list of class labels (class_number -1 ) -> str
    _classes: list = None
    # FIXME[design]: there is also an array _target[], mapping
    # indices to labels. Make this more coherent!
    
    # list of val_labels: (number_from_filename - 1) -> (class_number - 1)
    _val_labels: list = None

    _image: np.ndarray = None
    _label: int = None

    # FIXME[hack]
    try:
        from appdirs import AppDirs
        appname = "deepvis"  # FIXME: not the right place to define here!
        appauthor = "krumnack"
        _appdirs = AppDirs(appname, appauthor)
    except ImportError:
        _appdirs = None
        logger.warning(
            "--------------------------------------------------------------\n"
            "info: module 'appdirs' is not installed.\n"
            "We can live without it, but having it around will provide\n"
            "additional features.\n"
            "See: https://github.com/ActiveState/appdirs\n"
            "--------------------------------------------------------------\n")

    def __init__(self, prefix=None, section='val', **kwargs):
        super().__init__(id=f"imagenet-{section}",
                         description=f"ImageNet {section}", **kwargs)
        self._imagenet_data = os.getenv('IMAGENET_DATA', '.')
        self._section = section
        self.directory = os.path.join(self._imagenet_data, self._section)
        self._available = os.path.isdir(self.directory)
        self._image = None

    @property
    def number_of_labels(self) -> int:
        """The ImageNet dataset has 1000 classes.
        """
        return 1000

    def _prepare_data(self):
        logger.info(f"PREPARING ImageNet: {self.directory}: {self._available}")
        if self._section == 'train':
            # get the list of directory names, i.e.
            # the WordNet 3.0 synsets ('n0......')
            self._synsets = os.listdir(self.directory)

        #
        # get a list of filenames (possibly stored in a cache file)
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

        #
        # No filelist available yet - read it in (and possible store in cache)
        #
        if self._filenames is None:
            super()._prepare_data()
            if self._appdirs is not None:
                logger.info(f"Writing filenames to {imagenet_filelist}")
                if not os.path.isdir(self._appdirs.user_cache_dir):
                    os.makedirs(self._appdirs.user_cache_dir)
                pickle.dump(self._filenames, open(imagenet_filelist, 'wb'))

    @property
    def labels_prepared(self) -> bool:
        """Check if labels for this dataset have been prepared.
        """
        return ((self._section == 'train' and self._wn_table is not None) or
                (self._section == 'val' and self._val_labels is not None))

    def _prepare_labels(self):
        imagenet_labels_filename = \
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'imagenet_labels.txt')
        ids = []
        caffe_ids = []
        torch_ids = []
        synsets = []
        wordforms = []
        wordforms_long = []

        try:
            with open(imagenet_labels_filename) as f:
                for line in f.readlines():
                    id, torch_id, caffe_id, synset, wordform, wordform_long = \
                        line.strip().split(':')
                    ids.append(int(id))
                    torch_ids.append(int(torch_id))
                    caffe_ids.append(int(caffe_id))
                    synsets.append(synset)
                    wordforms.append(wordform)
                    wordforms_long.append(wordform_long)
            
            self.add_label_format('imagenet', np.asarray(ids))
            self.add_label_format('caffe', np.asarray(caffe_ids))
            self.add_label_format('torch', np.asarray(torch_ids))
            self.add_label_format('synset', synsets)
            self.add_label_format('text', wordforms)
            self.add_label_format('text_long', wordforms_long)

        except FileNotFoundError:
            logger.warning("ImageNet labels files not found. "
                           f"Make sure that {imagenet_labels_filename} "
                           "is available.")

        # FIXME[old]
        # load the labels file (classes.txt). 
        # The file contains mapping of labels to synset names
        # and text representation for labels,
        # e.g. "1000:n03255030:dumbbell"
        #try:
        #    classes_filename = os.path.join(self._imagenet_data,
        #                                    'classes.txt')
        #    with open(classes_filename) as f:
        #        classes = f.readlines()
        #    texts = [c.strip().split(':')[2] for c in classes]
        #    self.add_label_format('text', texts)
        #except FileNotFoundError:
        #    logger.warning("ImageNet class names not found. "
        #                   f"Make sure that {classes_filename} "
        #                   "is available.")
        
        if self._section == 'train':
            self._prepare_labels_train()

        elif self._section == 'val':
            self._prepare_labels_val()
        
        
        logger.info(f"ImageNet is now prepared: {len(self)}")

    def _prepare_labels_train(self):
        """Prepare the image to label mapping for the ILSVRC2012
        training set. This mapping can be achieved as the training
        files are grouped in directories named by the respective
        synset.
        """
        # FIXME[old]
        # get a mapping of synsets to labels
        #from .imagenet_classes import wn_table
        # self._wn_table = wn_table
        self._wn_table = { synset: index for index, synset
                           in enumerate(self._label_formats['synset']) }
        logger.info(f"Length of wn_table: {len(self._wn_table)}")

    def _prepare_labels_val(self):
        """Load the image to label mapping for the ILSVRC2012
        validation set. This mapping is provided as part of the
        'ILSVRC2012_devkit_t12'.
        """

        #val_labels_filename = os.path.join(self._imagenet_data,
        #                                   'val_labels.txt')
        val_labels_filename = \
            os.path.join(self._imagenet_data, 'ILSVRC2012_devkit_t12',
                         'data', 'ILSVRC2012_validation_ground_truth.txt')
        try:
            with open(val_labels_filename) as f:
                self._val_labels = [int(l.strip())-1 for l in f.readlines()] 
        except FileNotFoundError:
            logger.warning("ImageNet validation labels not found. "
                           f"Make sure that {val_labels_filename} "
                           "is available.")

    def _label_for_filename(self, filename):
        label = None
        if self._section == 'train':
            match_object = re.search('(n[0-9]*)_', filename)
            if match_object is not None:
                synset = self._wn_table.get(match_object.group(1))
        elif self._section == 'val' and self._val_labels is not None:
            match_object = re.search('ILSVRC2012_val_([0-9]*).JPEG', filename)
            if match_object is not None:
                label = self._val_labels[int(match_object.group(1))-1]
        return label

    def __getitem__(self, index):  # FIXME[old]
        if not self.prepared:
            return InputData(None, None)

        filename = self._filenames[index]
        data = imread(os.path.join(self.directory, filename))
        category = self._category_for_filename(filename)
        return InputData(data, category)

    def _fetch(self, **kwargs):
        self._fetch_random(**kwargs)

    def _fetch_random(self, **kwargs):
        if self._section == 'train':
            category = random.randint(0, len(self._synsets)-1)
            img_dir = os.path.join(self.directory, self._synsets[category])
            filename = random.choice(os.listdir(img_dir))
            img_file = os.path.join(img_dir, filename)
        elif self._section == 'val':
            filename = random.choice(self._filenames)
            img_file = os.path.join(self.directory, filename)

        self._image = imread(img_file)
        self._label = self._label_for_filename(filename)

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

    def _get_label(self):
        return self._label


    def oldrandom(self):  # FIXME[old]
        category = random.randint(0, len(self._synsets)-1)
        img_dir = os.path.join(self.directory, self._synsets[category])
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

    def get_label_old(self, target: int) -> str:
        label = self._val_labels[target]
        return self._classes[target]

    def check_availability():
        '''Check if this Datasource is available.

        Returns
        -------
        True if the DataSource can be instantiated, False otherwise.
        '''
        return self.directory

    def download():
        raise NotImplementedError("Downloading ImageNet is "
                                  "not implemented yet.")

    def __str__(self):
        return f"ImageNet ({self._section})"
