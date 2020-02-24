from . import Datasource, DataDirectory, Labeled, Random, Predefined

from util.image import imread


import os
import random
import numpy as np
import pickle

import logging
logger = logging.getLogger(__name__)


class WiderFace(DataDirectory, Random, Labeled, Predefined):
    """
    http://shuoyang1213.me/WIDERFACE/

    Class attributes
    ----------------
    
    """

    blur = ('clear', 'normal blur', 'heavy blur')
    expression = ('typical expression', 'exaggerate expression')
    illumination = ('normal illumination', 'extreme illumination')
    occlusion = ('no occlusion', 'partial occlusion', 'heavy occlusion')
    pose = ('typical pose', 'atypical pose')
    invalid = ('valid image', 'invalid image')

    # FIXME[hack]: we should put this in the util package!
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

    def __init__(self, prefix=None, section='train', **kwargs):
        """Initialize the WIDER Face Datasource.
        """
        super().__init__(id=f"wider-faces-{section}",
                         description=f"WIDER Faces", **kwargs)
        self._widerface_data = os.getenv('WIDERFACE_DATA', '.')
        self._section = section
        self.directory = os.path.join(self._widerface_data,
                                      'WIDER_' + self._section, 'images')
        self._available = os.path.isdir(self.directory)
        self._image = None
        self._initialize_filenames()

    def _initialize_filenames(self, widerface_data: str=None) ->None:
        if widerface_data is not None:
            self._widerface_data = widerface_data

        self._annotations_filename = None
        if self._widerface_data is not None:
            annotations = os.path.join(self._widerface_data, 'wider_face_split',
                                       'wider_face_train_bbx_gt.txt')
            if os.path.isfile(annotations):
                self._annotations_filename = annotations
        
    @property
    def number_of_labels(self) -> int:
        """The WIDER face dataset has 62 classes.
        """
        return 62

    def _prepare_data(self):
        """Prepare the WIDER Face dataset. This will provide in a list of
        all images provided by the dataset, either by reading in a
        prepared file, or by traversing the directory.
        """
        logger.info(f"PREPARING WIDER Face: {self.directory}: {self._available}")

        #
        # get a list of filenames (possibly stored in a cache file)
        #
        if self._appdirs is not None:
            filename = f"widerface_{self._section}_filelist.p"
            widerface_filelist = os.path.join(self._appdirs.user_cache_dir,
                                             filename)
            logger.info(f"WIDER Face: trying to load filelist from "
                        f"'{widerface_filelist}")
            if os.path.isfile(widerface_filelist):
                self._filenames = pickle.load(open(widerface_filelist, 'rb'))
        else:
            widerface_filelist = None

        #
        # No filelist available yet - read it in (and possible store in cache)
        #
        if self._filenames is None:
            super()._prepare_data()
            if self._appdirs is not None:
                logger.info(f"Writing filenames to {widerface_filelist}")
                if not os.path.isdir(self._appdirs.user_cache_dir):
                    os.makedirs(self._appdirs.user_cache_dir)
                pickle.dump(self._filenames, open(widerface_filelist, 'wb'))

        self.load_annotations()

    def load_annotations(self):
        """Load the annotations for the training images.

        The annotations are stored in a single large text file
        ('wider_face_train_bbx_gt.txtX'), with a multi-line entry per file.
        An entry has the following structure: The first line contains
        the filename of the training image. The second line contains
        the number of faces in that image. Then follows one line for
        each face, consisting of a bounding box (x,y,w,h) and attributes
        (blur, expression, illumination, invalid, occlusion, pose)
        encoded numerically. In these lines, all numbers are separated
        by spaces. Example:

        0--Parade/0_Parade_marchingband_1_95.jpg
        5
        828 209 56 76 0 0 0 0 0 0 
        661 258 49 65 0 0 0 0 0 0 
        503 253 48 66 0 0 1 0 0 0 
        366 181 51 74 0 0 1 0 0 0 
        148 176 54 68 0 0 1 0 0 0 

        """
        self._annotations = {}
        try:
            with open(self._annotations_filename, "r") as f:
                for filename in f:
                    filename = filename.rstrip()
                    lines = int(f.readline())
                    faces = []
                    for l in range(lines):
                        # x1, y1, w, h, blur, expression, illumination,
                        #    invalid, occlusion, pose
                        faces.append(map(int, f.readline().split()))
                    if lines == 0:
                        # images with 0 faces nevertheless have one
                        # line with dummy attributes -> just ignore that line
                        f.readline()
                    # Store all faces for the current file
                    self._annotations[filename] = faces
        except FileNotFoundError as error:
            self._annotations = {}

    def _fetch(self, **kwargs):
        self._fetch_random(**kwargs)

    def _fetch_random(self, **kwargs):
        img_file = random.choice(list(self._annotations.keys()))
        self._image = imread(os.path.join(self.directory, img_file))
        for x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose in self._annotations[img_file]:
            self._image[y1:y1+h,x1:x1+4,:] = 0
            self._image[y1:y1+h,x1+w-4:x1+w,:] = 0
            self._image[y1:y1+4,x1:x1+w,:] = 0
            self._image[y1+h-4:y1+h,x1:x1+w,:] = 0
        self._label = "f{img_file}"

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

    def __str__(self):
        return f'WIDER Faces'

    def check_availability() -> bool:
        """Check if this Datasource is available.

        Returns
        -------
        available: bool
            True if the DataSource can be prepared, False otherwise.
        """
        return self._annotations_filename is not None
