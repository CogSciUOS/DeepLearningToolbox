"""The Flickr-Faces-HQ Dataset (FFHQ).
"""

# standard imports
from pathlib import Path
import os
import logging

# third party imports
import numpy as np

# toolbox imports
from dltb.base.data import Data
from dltb.base.image import BoundingBox, Region
from dltb.tool.face.landmarks import FacialLandmarks
from dltb.datasource import ImageDirectory
from dltb.util import cache_path

# logging
LOG = logging.getLogger(__name__)


class CelebA(ImageDirectory):
    # pylint: disable=too-many-ancestors, too-many-instance-attributes
    """An interface to the CelebA dataset.  The dataset provides more than
    200,000 portrait pictures of celebrities, (exactly 202,599 images)
    annotated with 40 attributes, bounding boxes, landmarks, and
    identity information (10,177 identities).

    The data a provided in two formats: the raw images, and aligned
    and cropped versions of the images.  The raw images differ in
    size, while the aligned images all have size 178x218 pixels.


    Data format
    -----------

    For each format (raw and aligned) the data are provided in a
    separate directory. Each of these directories contains all the
    images as individual images files: 202,599 files in `img_celeba` and
    202,599 files in `img_align_celeba`).


    SQLite Database
    ---------------

    There exists an SQLite database file `celeba.sqlite` containing
    four tables (`attr`, `bbox`, `landmarks`, `landmark_align`).
    These tables allow to search the dataset for certain combinations
    of features.

    """

    def __init__(self, key: str = None, aligned: bool = False,
                 **kwargs) -> None:
        """Initialize the ImageNet dataset.

        Parameters
        ----------
        """
        # FIXME[hack]:
        # directory = ffhq_data or os.getenv('CELEBA_DATA', '.')
        description = 'CelebA dataset'
        directory = Path('/space/data/celeba/')
        image_suffix = 'jpg'
        anno = directory / 'Anno'

        self._aligned = aligned
        self._identities_file = anno / 'identity_CelebA.txt'
        self._attr_file = anno / 'list_attr_celeba.txt'
        if aligned:
            image_directory = directory / 'img_align_celeba'
            self._landmarks_file = anno / 'list_landmarks_align_celeba.txt'
            self._bbox_file = None
        else:
            image_directory = directory / 'img_celeba'
            self._landmarks_file = anno / 'list_landmarks_celeba.txt'
            self._bbox_file = anno / 'list_bbox_celeba.txt'

        super().__init__(key=key or "celeba",
                         directory=image_directory, suffix=image_suffix,
                         description=description, **kwargs)
        self._identities = None
        self._landmarks = None
        self._bboxes = None
        self._attr_names = None
        self._attributes = None
        LOG.info("Initialized CelebA: %s", self.directory)

    @property
    def aligned(self) -> bool:
        """A flag indicating whether the aligned (`True`) or the
        raw versions of the images (`False`) are used.
        """
        return self._aligned

    #
    # Preparation
    #

    def _prepare(self, **kwargs) -> None:
        # pylint: disable=arguments-differ
        """Prepare the ImageNet data.
        """
        LOG.debug("Preparing CelebA: %s", self.directory)

        basename = f"celeba-{'aligned' if self._aligned else 'raw'}"
        filenames_cache = basename + '.p'
        super()._prepare(filenames_cache=filenames_cache, **kwargs)

        if self._attr_file is not None:
            self._prepare_attributes()

        if self._landmarks_file is not None:
            self._prepare_landmarks()

        if self._bbox_file is not None:
            self._prepare_bboxes()

        if self._identities_file is not None:
            self._prepare_identities()

    def _prepare_attributes(self) -> None:
        attr_cache = cache_path('celeba-attributes.npy')
        if os.path.exists(attr_cache):
            self._attributes = np.load(attr_cache, mmap_mode='r')
            LOG.debug("Loaded %s CelebA attributes from '%s'",
                      self._attributes.shape, attr_cache)

        with open(self._attr_file, 'r') as infile:
            entries = int(infile.readline())
            self._attr_names = infile.readline().split()
            if self._attributes is None:
                LOG.debug("Loading %d*40 CelebA attributes from '%s'",
                          entries, self._attr_file)
                lookup = {name: index for index, name in
                          enumerate(self._filenames)}
                self._attributes = np.ndarray((entries, 40), dtype=np.bool_)
                for line in infile:
                    fields = line.split()
                    entry = lookup[fields.pop(0)]
                    self._attributes[entry] = \
                        tuple(field == '1' for field in fields)
                np.save(attr_cache, self._attributes)
                LOG.debug("Stored %s CelebA attributes into '%s'",
                          self._attributes.shape, attr_cache)

    def _prepare_landmarks(self) -> None:
        basename = f"celeba-{'aligned' if self._aligned else 'raw'}"
        landmark_cache = cache_path(basename + '-landmarks.npy')
        if os.path.exists(landmark_cache):
            self._landmarks = np.load(landmark_cache, mmap_mode='r')
            LOG.debug("Loaded %s CelebA landmarks from '%s'",
                      self._landmarks.shape, landmark_cache)
        else:
            with open(self._landmarks_file, 'r') as infile:
                entries = int(infile.readline())
                # landmark_name = infile.readline().split()
                infile.readline()
                LOG.debug("Loading %d Celeba landmarks from '%s'",
                          entries, self._landmarks_file)
                lookup = {name: index for index, name in
                          enumerate(self._filenames)}
                self._landmarks = np.ndarray((entries, 10), dtype=np.int16)
                for line in infile:
                    fields = line.split()
                    entry = lookup[fields.pop(0)]
                    self._landmarks[entry] = \
                        tuple(int(field) for field in fields)
                np.save(landmark_cache, self._landmarks)
                LOG.debug("Stored %s CelebA landmarks into '%s'",
                          self._landmarks.shape, landmark_cache)

    def _prepare_bboxes(self) -> None:
        bbox_cache = cache_path('celeba-bboxes.npy')
        if os.path.exists(bbox_cache):
            self._bboxes = np.load(bbox_cache, mmap_mode='r')
            LOG.debug("Loaded %s CelebA bounding boxes from '%s'",
                      self._bboxes.shape, bbox_cache)
        else:
            with open(self._bbox_file, 'r') as infile:
                entries = int(infile.readline())
                infile.readline()
                LOG.debug("Loading %d Celeba bounding boxes from '%s'",
                          entries, self._bbox_file)
                lookup = {name: index for index, name in
                          enumerate(self._filenames)}
                self._bboxes = np.ndarray((entries, 4), dtype=np.int16)
                for line in infile:
                    fields = line.split()
                    entry = lookup[fields.pop(0)]
                    self._bboxes[entry] = \
                        tuple(int(field) for field in fields)
                np.save(bbox_cache, self._bboxes)
                LOG.debug("Stored %s CelebA bounding boxes into '%s'",
                          self._bboxes.shape, bbox_cache)

    def _prepare_identities(self) -> None:
        LOG.debug("Loading Celeba identity information '%s'",
                  self._identities_file)
        self._identities = np.ndarray(len(self._filenames), dtype=np.int16)
        lookup = {name: index for index, name in
                  enumerate(self._filenames)}
        with open(self._identities_file, 'r') as infile:
            for line in infile:
                image, identity = line.split()
                self._identities[lookup[image]] = identity

    def _unprepare(self) -> None:
        self._landmarks = None
        self._bboxes = None
        self._attributes = None
        self._attr_names = None
        self._identities = None
        super()._unprepare()

    def _get_meta(self, data: Data, **kwargs) -> None:
        # pylint: disable=arguments-differ
        if self._identities is not None:
            data.add_attribute('identity', batch=True)
        if self._landmarks is not None:
            data.add_attribute('landmarks', batch=True)
        if self._bboxes is not None:
            data.add_attribute('bbox', batch=True)
        if self._attr_names is not None:
            for name in self._attr_names:
                data.add_attribute(name, batch=True)
        super()._get_meta(data, **kwargs)

    def _get_data(self, data: Data, **kwargs) -> None:
        # pylint: disable=arguments-differ
        """
        Raises
        ------
        ValueError:
            If the index is present, but the :py:class:`DataFiles`
            has no filename register.
        """
        super()._get_data(data, **kwargs)
        if self._identities is not None:
            data.identity = self._identities[data.index]
        if self._bboxes is not None:
            box = self._bboxes[data.index]
            data.bbox = Region(BoundingBox(x=box[0], y=box[1],
                                           width=box[2], height=box[3]))
        if self._landmarks is not None:
            landmarks = \
                FacialLandmarks(self._landmarks[data.index].reshape((-1, 2)))
            data.landmarks = landmarks  # Region(landmarks)
        if self._attr_names is not None:
            for index, name in enumerate(self._attr_names):
                setattr(data, name, bool(self._attributes[data.index, index]))
