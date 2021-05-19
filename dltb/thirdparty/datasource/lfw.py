"""The Labeled Faces in the Wild (LFW) dataset.

from dltb.thirdparty.datasource.lfw import LabeledFacesInTheWild
lfw = LabeledFacesInTheWild()
lfw.prepare()
lfw.sklearn is None

"""

# standard imports
import logging
import importlib

# toolbox imports
from dltb.datasource import Imagesource, DataDirectory

# logging
LOG = logging.getLogger(__name__)


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

    Scikit-Learn API
    ----------------

    Scikit-Learn includes functions to download and access the LFW
    dataset.  This class can make use of these functions, given that
    `sklearn` is installed.

    References
    ----------
    [1] http://vis-www.cs.umass.edu/lfw/

    """

    def __init__(self, key: str = None, lfw_data: str = None,
                 **kwargs) -> None:
        """Initialize the Labeled Faces in the Wild (LFW) dataset.

        Parameters
        ----------
        lfw_data: str
            The path to the LFW root directory. This directory
            should contain the 5.749 subdirectories holding images
            of the known persons.
        """
        # directory = '/net/projects/data/lfw/lfw'  # FIXME[hack]
        if lfw_data is None:
            lfw_data = '/space/data/lfw/lfw'
        description = "Labeled Faces in the Wild"
        super().__init__(key=key or "lfw",
                         directory=lfw_data,
                         description=description,
                         label_from_directory='name',
                         **kwargs)
        self.sklearn = None

    def _prepare(self) -> None:
        super()._prepare()

        try:
            self.sklearn = importlib.import_module('sklearn.datasets')
            LOG.info("Scikit-learn successfully imported.")
        except ImportError:
            # we will go without sklearn ...w
            LOG.warning("Importing scikit-learn (sklearn) failed.")

    def fetch_people(self, min_faces_per_person: int = 70,
                     resize: float = 1.0, color: bool = False):
        """
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
        return self.sklear.\
            fetch_lfw_people(min_faces_per_person=min_faces_per_person,
                             resize=resize, color=color)

    def fetch_pairs(self, subset: str = 'train', color: bool = False):
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
