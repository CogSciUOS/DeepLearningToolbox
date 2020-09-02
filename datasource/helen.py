"""The HELEN face dataset.
"""

# standard imports
from typing import Tuple
import os
import errno
from glob import glob

# third party imports
import numpy as np

# toolbox imports
from util.image import Landmarks, Region
from .data import Data
from .datasource import Imagesource
from .directory import DataDirectory


class Helen(DataDirectory, Imagesource):
    # pylint: disable=too-many-ancestors
    """A face landmarking dataset consisting of 2330 higher resolution
    face images, annotated with 194 points facial landmarks (one face
    per image), but no bounding box for the face.

    http://www.ifp.illinois.edu/~vuongle2/helen/

    The 2330 images of this datasource are stored in individual image
    files, distributed over the directories helen_1 to helen_5 (or
    alternatively in train_1 to train_4 and test). Landmark annotations
    for each image are in a separate file in the directory `annotation`.
    Each file contains a reference to the image file and coordinates
    of the 194 facial landmark points.

    Attributes
    ----------
    Class attributes:

    _annotations: dict
        A dictionary mapping filenames to annotation files or
        landmark objects (depending on the value of the flag
        _load_annotations).
    _load_annotations: bool
        A flag indicating if all annotations should be loaded into
        memory (True), or if only the current annotation should
        be loaded (False).

    """

    def __init__(self, key: str = "Helen", **kwargs) -> None:
        """Initialize the HELEN Facial Landmark Datasource.
        """
        helen_data = os.getenv('HELEN_DATA', '.')
        super().__init__(key=key, directory=helen_data,
                         description=f"HELEN Faces", **kwargs)
        self._annotations = None
        self._load_annotations = False

    def __str__(self):
        return 'Helen'

    #
    # Preparation
    #

    def _prepare(self) -> None:
        # pylint: disable=arguments-differ
        """Prepare the HELEN Face dataset. This will provide in a list of
        all images provided by the dataset, either by reading in a
        prepared file, or by traversing the directory.
        It will also prepare the mapping of filenames to annotations
        (landmarks).
        """
        # This will perform sanity checks and also call
        # self._prepare_filenames().
        super()._prepare()

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
        for subdir in subdirs:
            subdir = os.path.join(self.directory, subdir)
            if not os.path.isdir(subdir):
                FileNotFoundError(errno.ENOTDIR, os.strerror(errno.ENOTDIR),
                                  subdir)
            self._filenames += glob(os.path.join(subdir, "*.*"))

    def _prepare_annotations(self, filename: str = 'annotations.txt',
                             load: bool = None) -> None:
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
            raise FileNotFoundError(errno.ENOTDIR,
                                    "The 'annotation' directory of the HELEN "
                                    "facial landmark dataset is missing.",
                                    abs_annotation_dir)

        if load is not None:
            self._load_annotations = load

        cache_file = ('helen_annotations_'
                      f'{str(self._load_annotations).lower()}.p')
        self._annotations = self._read_cache(cache_file)
        if self._annotations is not None:
            return  # we have loaded the annotations from the cache file

        self._annotations = {}
        abs_filename = os.path.join(self.directory, filename)
        if os.path.isfile(abs_filename):
            # we have a file mapping image names to annotation files
            # -> parse that file
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
        self._write_cache(cache_file, self._annotations)

    @staticmethod
    def _load_annotation(filename: str) -> Tuple[str, Landmarks]:
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
            points = np.ndarray((194, 2))
            for i, line in enumerate(file):
                x_pos, y_pos = line.split(' , ')
                points[i] = float(x_pos), float(y_pos)
        return image_name, Landmarks(points)

    #
    # Data
    #

    def _get_meta(self, data: Data, **kwargs) -> None:
        data.add_attribute('label', batch=True)
        super()._get_meta(data, **kwargs)

    def _get_data_from_file(self, data: Data, filename: str) -> None:
        super()._get_data_from_file(data, filename)

        # provide the metadata (landmarks)
        basename = os.path.basename(filename)
        if basename.endswith('.jpg'):
            basename = basename[:-len('.jpg')]
        if self._load_annotations:
            # self._annotations[basename] is the landmark object
            landmarks = self._annotations[basename]
        else:
            # self._annotations[basename] is the name of the annotation
            # file (relative to the annotation directory)
            annotation_filename = os.path.join(self.directory, 'annotation',
                                               self._annotations[basename])
            _, landmarks = self._load_annotation(annotation_filename)

        data.label = Region(landmarks)
