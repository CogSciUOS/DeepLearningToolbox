import os
import random
from glob import glob
import numpy as np

from . import Datasource, DataDirectory, Predefined, Imagesource
from util.image import imread, Landmarks

import logging
logger = logging.getLogger(__name__)

class Helen(DataDirectory, Predefined, Imagesource):
    """A face landmarking dataset consisting of 2330 higher resolution
    face images, annotated with 194 points facial landmarks.

    http://www.ifp.illinois.edu/~vuongle2/helen/
    
    Class attributes
    ----------------
    _annotations: dict
        A dictionary mapping filenames to annotation files.
    _load_annotations: bool
        A flag indicating if all annotations should be loaded into
        memory (True), or if only the current annotation should
        be loaded.
    """
    _annotations: dict = None
    _load_annotations: bool = False

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

    def __init__(self, prefix=None, **kwargs):
        """Initialize the HELEN Facial Landmark Datasource.
        """
        helen_data = os.getenv('HELEN_DATA', '.')
        super().__init__(id=f"Helen", dirname=helen_data,
                         description=f"HELEN Faces", **kwargs)

    def _prepare_data(self):
        """Prepare the HELEN Face dataset. This will provide in a list of
        all images provided by the dataset, either by reading in a
        prepared file, or by traversing the directory.
        It will also prepare the mapping of filenames to annotations
        (landmarks).
        """
        # This will perform sanity checks and also call
        # self._prepare_filenames().
        super()._prepare_data()

        # Now We will also prepare the annotations (landmarks)
        self._prepare_annotations()

    def _prepare_filenames(self):
        """Prepare the list of filenames maintained by the
        :py:class:`Helen` datasource.

        The data are distributed over five directories
        ('helen_1' to 'helen_5')

        Raises
        ------
        FileNotFoundError:
            One of the subdirectories does not exist.
        """
        self._filenames = []
        subdirs = ('helen_1', 'helen_2', 'helen_3', 'helen_4', 'helen_5')
        # subdirs = ('train_1', 'train_2', 'train_3', 'train_4', 'test')
        for d in subdirs:
            subdir = os.path.join(self.directory, d)
            if not os.path.isdir(subdir):
                FileNotFoundError(f"No such directory: {subdir}")
            self._filenames += glob(os.path.join(subdir, "*.*"))

    def _prepare_annotations(self, filename: str='annotations.txt',
                             load: bool=None) -> None:
        """Prepare the annotations dictionary mapping file names to annotation
        files.

        Arguments
        ---------
        filename: str
            Name of a text file containing the mapping information in
            two columns, separated by a space. The first column is
            the image filename (basename without directory and suffix)
            and the second column contains the name of the annotation file
            in the "annotations" subdirectory (again basename
            without directory).        

        Notice: the file 'annotations.txt' is not part of the HELEN dataset,
        but has to be manually constructed.

        Raises
        ------
        FileNotFoundError:
            The 'annotation' subdirectory does not exist.
        """
        abs_annotation_dir = os.path.join(self.directory, 'annotation')
        if not os.path.isdir(abs_annotation_dir):
            raise FileNotFoundError("The 'annotation' directory of the HELEN "
                                    "facial landmark dataset is missing.")

        if load is not None:
            self._load_annotations = load

        self._annotations = {}
        abs_filename = os.path.join(self.directory, filename)
        if os.path.isfile(abs_filename):
            # we have a file mapping image names to annotation files
            with open(abs_filename) as file:
                for line in file:
                    image, annotation_file = line.rstrip().split()
                    self._annotations[image] = annotation_file
            if self._load_annotations:
                for image, name in self._annotations.items():
                    abs_annotation = os.path.join(abs_annotation_dir, name)
                    _, self._annotations[image] = \
                        self._load_annotation(abs_annotation)
        else:
            # we have no file mapping image names to annotation files
            # -> create the mapping by iterating over the annotation directory
            for name in os.listdir(abs_annotation_dir):
                abs_annotation = os.path.join(abs_annotation_dir, name)
                image_name, landmarks = self._load_annotation(abs_annotation)
                self._annotations[image_name] = \
                    landmarks if self._load_annotations else name
            # FIXME[todo]:
            # 1. in case we created the annotations mapping,
            #    we may store them in a cache file ...
            # 2. in case we loaded all annotations, we may
            #    store them in a cache file

    def _load_annotation(self, filename: str) -> None:
        """Load annotation (facial landmark information) from a file.

        The annotation file consist of 195 lines. The first line
        is the name of the image, while the following 194 lines contain
        landmark coordinates, each coordinate being a comma separated
        pair of floats.

        Arguments
        ---------
        filename: str
            The absolute name of the annotation file.
        """
        with open(filename) as file:
            # Ignore the first line (image name)
            image_name = file.readline().rstrip()
            points = np.ndarray((194,2))
            for i, line in enumerate(file):
                x, y = line.split(' , ')
                points[i] = float(x), float(y)
        return image_name, Landmarks(points)

    def _fetch(self, **kwargs):
        # Step 1: provide the image
        super()._fetch(**kwargs)

        # Step 2: provide the metadata (landmarks)
        if self._metadata is not None:
            image_base = self._metadata.get_attribute('basename')
            if image_base.endswith('.jpg'):
                image_base = image_base[:-len('.jpg')]
            if self._load_annotations:
                landmarks = self._annotations[image_base]
            else:
                filename = os.path.join(self.directory, 'annotation',
                                    self._annotations[image_base])
                _, landmarks = self._load_annotation(filename)

            self._metadata.set_attribute('name', image_base)
            self._metadata.add_region(landmarks)

    def __str__(self):
        return 'Helen'


