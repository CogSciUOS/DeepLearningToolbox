import os
import re

import logging
logger = logging.getLogger(__name__)

import random
import pickle
from scipy.misc import imread, imresize
from datasources.imagenet_classes import class_names

from datasources import DataSource, DataDirectory, InputData, Predefined

# FIXME[todo]: this is currently just a hack, not a full-fleshed
# DataSource!
#  - implement the standard DataSource API
#  - allow for all "sections" (train/valid/test)
#  - automatically provide teh ImageNet class labels

try:
    from appdirs import AppDirs
except ImportError:
    AppDirs = None
    logger.warning(
        "---------------------------------------------------------------\n"
        "info: module 'appdirs' is not installed. We can live without it,\n"
        "but having it around will provide additional features.\n"
        "See: https://github.com/ActiveState/appdirs\n"
        "---------------------------------------------------------------\n")


class ImageNet(DataDirectory, Predefined):

    _section_ids = ['train', 'test', 'val']

    def __init__(self, prefix=None, section='train'):
        self._section = section
        self._imagenet_data = os.getenv('IMAGENET_DATA', '.')
        super(ImageNet, self).__init__()
        Predefined.__init__(self, 'imagenet')

    def prepare(self):
        if self.prepared:
            return  # nothing to do ...

        self._prefix = os.path.join(self._imagenet_data, self._section)
        self._available = os.path.isdir(self._prefix)

        logger.info(f"PREPARING ImageNet : {self._prefix}: {self._available}")
        self._categories = os.listdir(self._prefix)

        from .imagenet_classes import wn_table
        self._wn_table = wn_table
        logger.info(f"Length of wn_table: {len(self._wn_table)}")

        self.directory = self._prefix

        # FIXME[hack]: just randomly choose one subdir
        self._category = random.randint(0, len(self._categories))
        # self.directory = os.path.join(self._prefix,
        #                               self._categories[self._category]))

        if AppDirs is not None:
            appname = "deepvis"  # FIXME: not the right place to define here!
            appauthor = "krumnack"
            dirs = AppDirs(appname, appauthor)
            imagenet_filelist = os.path.join(dirs.user_cache_dir,
                                             "imagenet_filelist.p")
            logger.info(f"ImageNet: trying to load filelist from "
                        f"'{imagenet_filelist}")
            if os.path.isfile(imagenet_filelist):
                self._filenames = pickle.load(open(imagenet_filelist, 'rb'))
        else:
            imagenet_filelist = None

        if self._filenames is None:
            super().prepare()
            if imagenet_filelist is not None:
                logger.info(f"Writing filenames to {imagenet_filelist}")
                if not os.path.isdir(dirs.user_cache_dir):
                    os.makedirs(dirs.user_cache_dir)
                pickle.dump(self._filenames, open(imagenet_filelist, 'wb'))
        else:
            self.change('state_changed')
        logger.info(f"ImageNet is now prepared: {len(self)}")

    def __getitem__(self, index):
        if not self.prepared:
            return InputData(None, None)

        filename = self._filenames[index]
        data = imread(os.path.join(self._dirname, filename))
        category = self._category_for_filename(filename)
        return InputData(data, category)

    def _category_for_filename(self, filename):
        match_object = re.search('(n[0-9]*)_', filename)
        if match_object is not None:
            category = self._wn_table.get(match_object.group(1))
        else:
            category = None
        return category

    def get_section_ids(self):
        '''Get a list of sections provided by this data source.  A data source
        may for example be divided into training and test data.
        '''
        return self._section_ids

    def get_section(self):
        '''Get the current section in the data source.
        '''
        return self._section

    def random(self):
        category = random.randint(0, len(self._categories)-1)
        img_dir = os.path.join(self._prefix, self._categories[category])
        self._image = random.choice(os.listdir(img_dir))
        img_file = os.path.join(img_dir, self._image)
        im = imread(img_file)
        return InputData(im, category)

    def set_section(self, section_id):
        '''Set the current section in the data source.
        '''
        self.section = section_id
        self.prepare()

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
