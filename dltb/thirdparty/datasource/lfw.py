"""The Labeled Faces in the Wild (LFW) dataset.

from dltb.thirdparty.datasource.lfw import LabeledFacesInTheWild
lfw = LabeledFacesInTheWild()
lfw.sklearn is None



LFW function from scikit-learn (sklearn)
----------------------------------------

.. code-block:: python

   from sklearn.datasets import fetch_lfw_people, fetch_lfw_pairs
   from dltb.util.image import imshow

   lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

   imshow(lfw_people.images[1])

   for name in lfw_people.target_names:
       print(name)

   lfw_pairs_train = fetch_lfw_pairs(subset='train')


Scikit-learn based Datasource from dltb.thirdparty.skimage
----------------------------------------------------------


.. code-block:: python

   from dltb.thirdparty.sklearn import LFW

   lfw = LFW(min_faces_per_person=70)
   d = lfw[1]


"""

# standard imports
from typing import Iterable, Tuple, Optional, Union
import logging
import importlib
from pathlib import Path
from argparse import ArgumentParser

# thirdparty imports
import numpy as np

# toolbox imports
from dltb.config import config
from dltb.base.image import Image, Imagelike
from dltb.util.image import imresize
from dltb.util.itertools import SizedGenerator, ignore_errors
from dltb.datasource import Imagesource, DataDirectory

# logging
LOG = logging.getLogger(__name__)

# types
Pathlike = Union[str, Path]

LfwID = Tuple[str, int]
# LfwPair = LabeledPair[LfwID]
LfwPair = Tuple[LfwID, LfwID, bool]
ImagePair = Tuple[np.ndarray, np.ndarray, bool]


class LabeledFacesInTheWild(DataDirectory, Imagesource):
    # pylint: disable=too-many-ancestors
    """"Labeled Faces in the Wild" (LFW) has for a long time been
    a standard benchmarking  dataset in the domain of face recognition.
    It consists of 13.233 images (each of size 250x250 pixels)
    of 5.749 people, with 1.680 people having two or more images.

    The images are contained in a directory called `lfw`, with
    one subdirectory per person (i.e., 5.749 subdirectories), named
    after that person (e.g., `Aaron_Eckhart`).
    Each of these subdirectories contains images depicting this
    person.

    Pair mode
    ---------
    The LFW dataset is originally a face verification dataset,
    providing a list of face pairs labeled with a flag indicating
    if both images depict the same person (`True`) or different
    persons (`False`).  In pair mode, the entries of the LFW
    `Datasource` will be such pairs.  If pair mode is turned off,
    entries will be individual images.

    Scikit-Learn API
    ----------------

    Scikit-Learn includes functions to download and access the LFW
    dataset.  This class can make use of these functions, given that
    `sklearn` is installed.

    The functions of the scikit-learn API support an optional
    `data_home` (defaulting to `$SCIKIT_LEARN_DATA` and if that is
    unset to `~/scikit_learn_data`) argument to specify the location
    of the data files.  However, it appends to that location the
    (hard-coded!) string `'lfw_home'`, meaning that one has to put
    data a directory with that name (or create a symlink with that
    name) in order to use local data with scikit-learn.  That
    directory should contain the files `'pairs.txt'`,
    `'pairsDevTrain.txt'`, and `'pairsDevTest.txt'` as well as subdirectories
    `'lfw/'` and `'lfw_funneled/'` containing the original and funneled
    images.  If any of this is missing, it will automatically be downloaded.

    Scikit-Learn provides the functions `fetch_lfw_people` and
    `fetch_lfw_pairs` to load a `sklearn.utils.Bunch` of images.
    These function internally use pillow for loading and resizing
    images.

    References
    ----------
    [1] http://vis-www.cs.umass.edu/lfw/

    """

    def __init__(self, key: Optional[str] = None,
                 directory: Optional[Pathlike] = None,
                 meta_directory: Optional[Pathlike] = None,
                 **kwargs) -> None:
        """Initialize the Labeled Faces in the Wild (LFW) dataset.

        Parameters
        ----------
        directory: str
            The path to the LFW root directory. This directory
            should contain the 5.749 subdirectories holding images
            of the known persons.
        """
        if directory is None:
            directory = getattr(config, 'data_directory_lfw',
                                config.data_directory / 'lfw' / 'lfw')
        description = "Labeled Faces in the Wild"
        super().__init__(key=key or "lfw",
                         directory=directory,
                         description=description,
                         label_from_directory='name',
                         **kwargs)
        self.meta_directory = Path(meta_directory) if meta_directory \
            else Path(self.directory).parent
        self.sklearn = None
        self._number_of_pairs = None

    def _prepare(self) -> None:
        super()._prepare()

        try:
            self.sklearn = importlib.import_module('sklearn.datasets')
            LOG.info("Scikit-learn successfully imported.")
        except ImportError:
            # we will go without sklearn ...
            LOG.warning("Importing scikit-learn (sklearn) failed.")

    @property
    def pair_mode(self) -> bool:
        """A flag indicating if the :py:class:`Datasource` operates in
        pair mode.  In pair mode, the entries of the `Datasource`
        or (labeled) pairs of images, with the boolean label indicating
        whether both images depict the same entity (`True`) or two
        different entities (`False`).  The length of the datasource will
        be the number of pairs and each datapoint will be a (labeled) pair
        of images.
        """
        # FIXME[todo]: also index access __getitem__() and iteration
        # should yield pairs!
        return self._number_of_pairs is not None

    @pair_mode.setter
    def pair_mode(self, pair_mode: bool) -> None:
        # Setting the LFW Datasource into pair mode requires a list of
        # pairs.  The default choice will be to use the pairs listed
        # in the file 'pairs.txt'.
        if pair_mode:
            self._pairs_txt = self.meta_directory / 'pairs.txt'
            with open(self._pairs_txt, encoding='utf-8') as pairs_file:
                line = pairs_file.readline().strip()
            self._number_of_splits, self._split_size = \
                self._parse_pairs_header(line)
            self._number_of_pairs = \
                self._number_of_splits * self._split_size * 2
            self._pair_coding = 'lfw'  # or 'custom'
        else:
            self._pairs_txt = None
            self._number_of_pairs = None
            self._pair_coding = None

    def __len__(self) -> int:
        if self.pair_mode:
            return self._number_of_pairs
        return super().__len__()

    def _pairs_from_filelike(self, filename,
                             folds: Optional[Tuple[int, int]] = None
                             ) -> Iterable[LfwPair]:
        """
        Arguments
        ---------
        folds:
            If provided, this should be a pair (folds, fold_size),
            specifying how many folds of what size should be read.
            For each fold first `fold_size` positive pairs
            and then `fold_size` negative pairs are expected
            (that is for each fold, `2*fold_size` pairs will be iterated).
        """
        with open(filename, encoding='utf-8') as pairs_file:
            # ignore first line (we already read it when entering pair mode)
            line = pairs_file.readline()
            # folds = None
            yield self._parse_pairs_header(line)  # (folds, fold_size)

            if folds is None:
                for line in pairs_file:
                    yield self._parse_pair_line(line)
            else:
                for _fold in range(folds[0]):
                    for _entry in range(folds[1]):
                        line = pairs_file.readline()
                        yield self._parse_pair_line(line, True)
                    for _entry in range(folds[1]):
                        line = pairs_file.readline()
                        yield self._parse_pair_line(line, False)

    def pairs(self, filename: Optional[Pathlike] = None,
              load_images: bool = True,
              skip_errors: bool = False) -> Iterable[ImagePair]:
        """Pairwise iteration of the dataset.

        Arguments
        ---------
        filename:
            The filename of the pairs list.  If `None`, the default
            file `pairs.txt` for this :py:class:`LabeledFacesInTheWild`
            datasource is used.
        load_images:
            A flag indicating if for the pairs should be loaded.
            If `False`, instead of images, only the LfwID of the images
            are returned.
        skip_errors:
            A flag indicating if errors raised while reading the pairs
            from the file shoudld be ignored.  If `True`, inccorect
            entries will be skipped, otherweise a `ValueError` is
            raised.

        Result
        ------
        Each iteration will yield a labeled pair consisting of two
        images and a boolean label indicating if the images depict the
        same person (`True`) or different persons (`False`).

        """
        if not self.pair_mode:
            raise RuntimeError("Operation is not available - "
                               "LFW object is not in pair mode!")

        if filename is None:
            filename = self._pairs_txt

        # Create a generator to enumerate the pairs listed in the
        # file. The labeled pairs provided by this iterator will only
        # contain the LfwID for the images, not the actual images.
        generator = self._pairs_from_filelike(filename)

        # First item return by generator is the pair (folds, fold_size).
        # We will consume it to provide a length for the SizedGenerator.
        folds = next(generator)

        # If skip_errors is set, ValueError (from Parsing) will be
        # ignored - the respective pair is effectively skipped
        if skip_errors:
            generator = ignore_errors(generator,
                                      (ValueError, FileNotFoundError), LOG)

        # If the load_images flag is given, the labeled LfwID pairs will
        # be replaced by labeled image pairs.
        if load_images:
            generator = map(lambda pair: (self._read_lfw_image(pair[0]),
                                          self._read_lfw_image(pair[1]),
                                          pair[2]), generator)
        return SizedGenerator(generator, folds[0] * folds[1] * 2)

    #
    # LfwID and LfwPair
    #

    def read_lfw_image(self, name: str, number: int) -> np.ndarray:
        """Read a single image from the LFW database.

        Images are expected to be stored there under the standard LFW
        naming scheme, that is one folder per person.  Folder names
        are person names and image names within each folder are
        composed from person name, and underscore, and a running
        number (4 places with leading 0). All image files are expected
        to be JPEG encoded with file suffix `'.jpg'` (lower case).

        Arguments
        ---------
        rootdir:
            Root directory of the LFW images.
        name:
            Name of the person.
        number:
            Running number of the image for this person.

        Result
        ------
        image:
            The image data in standard format (numpy.ndarray, uint8, RGB,
            color last).
        """
        return self._read_lfw_image(LfwID(name, f"{name}_{number:04}.jpg"))

    def _read_lfw_image(self, lfw_id: LfwID) -> np.ndarray:
        """Read the image for the give LFW image ID.
        """
        filename = Path(self.directory) / lfw_id[0] / lfw_id[1]
        return self.load_datapoint_from_file(filename)

    @staticmethod
    def _parse_pairs_header(line: str) -> Tuple[int, int]:
        """Parse the header line from an LFW pairs file
        (like '`pairs.txt`').

        Arguments
        ---------
        line:
            The header line from the file.  The line is expected to
            contain a pair of integers, separated by space,
            specify number of folds and fold size.

        Result
        ------
        fold:
            Number of folds.
        fold_size:
            Size of each fold.  Note that the actual number of pairs
            is twice that number, as each fold consists of `fold_size`
            positive pairs followed by `fold_size` negative pairs.
        """
        fields = tuple(int(number) for number in line.split())
        if len(fields) == 1:
            folds, fold_size = 1, fields[0]
        elif len(fields) == 2:
            folds, fold_size = fields
        else:
            raise ValueError(f"Bad start line in pairs file: {line}. "
                             "Expecting pair (folds, fold_size)")
        return folds, fold_size

    def _parse_pair_line(self, line: str,
                         same: Optional[bool] = None) -> LfwPair:
        """Parse a line from an LFW pairs file (like '`pairs.txt`').

        Arguments
        ---------
        line:
            A line from a LFW pairs file.
        same:
            A flag indicating if the line is expected a same pair.
            If a value is provided here that does not match the
            information found in the line, a `ValueError` is raised.

        Result
        ------
        pair:
            The labeled LFW pair obtain from this line.

        Raises
        ------
        ValueError:
            The `same` value provided as argument does not match the
            information obtained from the line.
        """
        fields = line.split()

        if self._pair_coding == 'lfw':
            # "lfw": official Labeled faces in the Wild coding:
            # Lines have the following format:
            #   same:        dirname filenum-1 filenum-2
            #   different:   dirname-1 filenum-1 dirname-2 filenum-2

            if len(fields) == 3:
                name1 = name2 = fields[0]
                number1, number2 = int(fields[1]), int(fields[2])
                is_same = True
            elif len(fields) == 4:
                name1, name2 = fields[0], fields[2]
                number1, number2 = int(fields[1]), int(fields[3])
                is_same = False
            else:
                raise ValueError(f"Bad pair line in pairs file: {line}")
            pair = ((name1, f"{name1}_{number1:04}.jpg"),
                    (name2, f"{name2}_{number2:04}.jpg"), is_same)
        else:
            # "custom"/"lfc": custom coding used int the
            # Labeled Children in the Wild dataset:
            #   same:       dirname filename-1 filename-2
            #   different:  dirname-1 filename-1 dirname-2 filename-2
            if len(fields) == 3:
                name1 = name2 = fields[0]
                filename1, filename2 = fields[1], fields[2]
                is_same = True
            elif len(fields) == 4:
                name1, name2 = fields[0], fields[2]
                filename1, filename2 = fields[1], fields[3]
                is_same = False
            else:
                raise ValueError(f"Bad pair line in pairs file: {line}")
            pair = ((name1, filename1), (name2, filename2), is_same)

        if same is not None and same is not is_same:
            raise ValueError(f"Same value for line is expected to be {same} "
                             f"but actually is {is_same}.")
        return pair

    def read_lfw_pairs_into_arrays(self, filename: Optional[Pathlike] = None,
                                   progress: Optional[Iterable] = None,
                                   skip_errors: bool = False) \
            -> Tuple[np.ndarray, np.ndarray]:
        """

        Arguments
        ---------
        filename:
            Path to the pairs file.  This file lists the images to be read
            and provides the same information.
        progress:
            A progress indicator (currently only `tqdm` is supported)

        Result
        ------
        images:
            An array of shape (NUMBER_OF_PAIRS*2, 112, 112, 3) and
            dtype `numpy.uint8`. Images are stored in RGB.
        same:
            A boolean array of shape (NUMBER_OF_PAIRS,).
        """
        pairs = self.pairs(filename=filename, load_images=True,
                           skip_errors=skip_errors)
        number_of_pairs = len(pairs)
        images = np.ndarray((number_of_pairs * 2, 112, 112, 3), dtype=np.uint8)
        same = np.ndarray((number_of_pairs, ), dtype=bool)

        if skip_errors:
            pairs = ignore_errors(pairs, FileNotFoundError, LOG)

        if progress is not None:
            pairs = progress(pairs)

        for index, pair in enumerate(pairs):
            images[index*2] = pair[0]
            images[index*2+1] = pair[1]
            same[index] = pair[2]

        return images, same

    #
    # face cropping (requires third party software)
    #

    @staticmethod
    def crop_faces(image: Imagelike) -> np.ndarray:
        """Crop faces in the style of the original LFW dataset.  The procedure
        for obtaining the 250x250 pixel images is as follows: detect
        faces with the OpenCV Haar Cascade detector. Then scale the
        (square-shaped) bounding box by a factor of 2.2 in each
        direction.  Scale the resulting crop to 250x250 pixels.
        """
        opencv_face_module = \
            importlib.import_module('.face', 'dltb.thirdparty.opencv')
        detector = opencv_face_module.DetectorHaar()
        image = Image(image)
        bounding_boxes = list(detector.detect_boxes(image))
        faces = np.ndarray((len(bounding_boxes), 250, 250, 3), dtype=np.uint8)
        for index, box in enumerate(bounding_boxes):
            box = box.scale(2.2, reference='center')
            patch = box.extract.extract_from_image(image)
            faces[index] = imresize(patch, (250, 250))
        return faces

    #
    # Scikit-Learn API
    #

    def sklearn_fetch_people(self, min_faces_per_person: int = 70,
                             resize: float = 1.0, color: bool = False):
        """Use the `sklearn.fetch_lfw_people` function to fetch
        images of people from the LFW dataset.

        Arguments
        ---------
        min_faces_per_person:
            The minimal number of images that should exist for each
            person to be included in the result.  Persons with less
            images will be ignored.  Notice that most persons in LFW
            only have one image and hence will be ignored when this
            value is larger than 1.
        resize:
            A reizing factor. A value of `1.0` means that images are
            returned in their original size.

        Result
        ------
        A `sklearn.utils.Bunch`, providing the following properties:
        `DESCR`:
            a documentation string
        `data`:
            a numpy.ndarray of dtype float32 and shape  (COUNT, 62*47)
            with values from 0.0 to 255.0
        `images`:
            a numpy.ndarray of dtype float32 and shape (COUNT, 62, 47)
            with values from 0.0 to 255.0
        `target`:
            a numpy.ndarray of dtype int64 and shape (COUNT, ) with
            values from 0 to the maximal target ID
        `target_names`:
            a numpy.ndarray of dtype string and shape (LEN,)
        """
        if self.sklearn is None:
            raise NotImplementedError("Scikit-learn is required for "
                                      "fetch_people")

        return self.sklearn.\
            fetch_lfw_people(min_faces_per_person=min_faces_per_person,
                             resize=resize, color=color)

    def sklearn_fetch_pairs(self, subset: str = 'train', color: bool = False):
        """
        Arguments
        ---------
        subset:
            Either `'train'`,  `'test'`, or `'10_folds'`.

        Result
        ------
        A `sklearn.utils.Bunch`, providing the following properties:
        `DESCR`:
            a documentation string
        `data`:
            a numpy.ndarray of dtype float32 and shape  (COUNT, 62*47)
            with values from 0.0 to 255.0
        `pairs`:
            a numpy.ndarray of dtype float32 and shape (COUNT, 2, 62, 47)
            with values from 0.0 to 255.0
        `target`:
            a numpy.ndarray of dtype int64 and shape (COUNT, ) with
            values 0 (different person) and 1 (same person).
        `target_names`:
            a numpy.ndarray of dtype string and shape (LEN,)
        """
        if self.sklearn is None:
            raise NotImplementedError("Scikit-learn is required for "
                                      "fetch_pairs")
        return self.sklear.fetch_lfw_pairs(subset=subset, color=color)


def main():
    """Main program. Read the LFW images into two numpy arrays
    (image data and boolean same flags) and write those arrays
    into two files (`lfw_images.npy` and `lfw_same.npy`).
    """
    # python dltb/thirdparty/datasource/lfw.py --to-numpy \
    #    --data-directory=/space/data/lfw/lfw_align_112

    # FIXME[old]: remove if these values are no longer needed.
    # (may be entered into some config file ...)
    # SCRATCH_DIR = Path('/space/home/ulf')
    # DATA_DIR = Path('/space/data')
    # UOS:
    # SCRATCH_DIR = \
    #   Path('/net/projects/scratch/winter/valid_until_31_July_2022/krumnack')
    # DATA_DIR = SCRATCH_DIR / 'data'
    # LFW_IMAGEROOT = DATA_DIR / 'lfw' / 'lfw_align_112'
    # LFW_TARGET_DIR = SCRATCH_DIR / 'data'
    # LFW_PAIRS_FILE = Path('/net/projects/data/lfw/archive/pairs.txt')
    # LFW_PAIRS_FILE = DATA_DIR / 'lfw' / 'pairs.txt'

    parser = ArgumentParser(description="Labeled faces in the Wild script.")
    group1 = parser.add_argument_group("Commands")
    group1.add_argument('--to-numpy', action='store_true', default=False,
                        help="store dataset as numpy array")
    group2 = parser.add_argument_group("Directories")
    group2.add_argument('--data-directory', default=None,
                        help="path to the LFW data directory "
                        "(containing the image files)")
    group2.add_argument('--meta-directory', default=None,
                        help="path to the LFW meta directory "
                        "(containing 'pairs.txt')")
    group2.add_argument('--target-directory', default='.',
                        help="path for storing output data")
    group2.add_argument('--pairs', default=None,
                        help="the pairs fiels to be used")
    args = parser.parse_args()

    # Create the Datasource object
    lfw = LabeledFacesInTheWild(directory=args.data_directory,
                                meta_directory=args.meta_directory)

    # import optional third-party helpers
    try:
        tqdm = importlib.import_module('tqdm')
        progress = tqdm.tqdm
    except ImportError:
        progress = None

    target_directory = Path(args.target_directory)

    if args.to_numpy:
        target_file_images = target_directory / 'lfw_images.npy'
        target_file_same = target_directory / 'lfw_same.npy'

        # Check if target files already exist
        target_file_exists = False
        for target_file in target_file_images, target_file_same:
            if target_file.is_file():
                target_file_exists = True
                LOG.warning("Target file '%s' already exists", target_file)
        if target_file_exists:
            LOG.fatal("Operation aborted. Will not overwrite existing "
                      "target files!")
            return  # abort operation ...

        # make sure the Datasource is in pair_mode
        lfw.pair_mode = True

        # Read in the image data
        LOG.info("Reading images from '%s' according to pairs listed in '%s':",
                 lfw.directory, args.pairs)
        images, same = \
            lfw.read_lfw_pairs_into_arrays(progress=progress)

        # Write the resulting arrays
        LOG.info("Writing results to '%s' (as 'lfw_images.npy' and "
                 " 'lfw_same.npy')", target_directory)
        np.save(target_file_images, images)
        np.save(target_file_same, same)


if __name__ == '__main__':
    main()
