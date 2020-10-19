"""The WiderFace dataset.
"""

# standard imports
import os
import logging

# third party imports
import numpy as np

# toolbox imports
from util.image import BoundingBox, Region, Landmarks
from dltb.base.data import Data
from dltb.tool.classifier import ClassScheme
from dltb.datasource import Imagesource, Sectioned, DataDirectory

# logging
LOG = logging.getLogger(__name__)


class WiderfaceScheme(ClassScheme):
    """The WiderFace dataset divides its data into
    62 classes (actually just 61, as class 60 is missing).
    Class labels can be obtained from directory names in the
    data directories.
    """

    def __init__(self) -> None:
        """Iniitalization of the :py:class:`WiderfaceScheme`.
        """
        # The WIDER face dataset has 62 classes (but it seems
        # that only 61 are used - class '60' is missing).
        super().__init__(length=62, key='widerface')

    @property
    def prepared(self) -> bool:
        """Check if the :py:class:`WiderfaceScheme` has been initialized.
        """
        return 'text' in self._labels

    def prepare(self) -> None:
        """Prepare the labels for the Widerface dataset.
        The labels will be read in from the directory names
        in the WIDERFACE_DATA directory.
        """
        if self.prepared:
            return  # nothing to do ...

        widerface_data = os.getenv('WIDERFACE_DATA')
        train_dir = os.path.join(widerface_data, 'WIDER_train', 'images')
        text = [''] * len(self)
        for dirname in os.listdir(train_dir):
            number, label = dirname.split('--', maxsplit=1)
            text[int(number)] = label
        self.add_labels(text, 'text')


WiderfaceScheme()


class WiderFace(DataDirectory, Imagesource, Sectioned,
                sections={'train', 'val', 'test'}):
    # pylint: disable=too-many-ancestors
    """
    http://shuoyang1213.me/WIDERFACE/
    "Wider Face" is A face detection benchmark consisting of 32,203
    images with 393,703 labeled faces.

    The faces have wide variability in scale, pose, occlusion.
    Images are categorized in 61 event class.

    From each class train/validation/test datasets where split
    in relation 40%/10%/50%.

    Attributes
    ----------
    blur: Tuple[str]
    expression: Tuple[str]
    illumination: Tuple[str]
    occlusion: Tuple[str]
    invalid: Tuple[str]
    """

    blur = ('clear', 'normal blur', 'heavy blur')
    expression = ('typical expression', 'exaggerate expression')
    illumination = ('normal illumination', 'extreme illumination')
    occlusion = ('no occlusion', 'partial occlusion', 'heavy occlusion')
    pose = ('typical pose', 'atypical pose')
    invalid = ('valid image', 'invalid image')

    def __init__(self, section: str = 'train',
                 key: str = None, **kwargs) -> None:
        """Initialize the WIDER Face Datasource.
        """
        self._widerface_data = os.getenv('WIDERFACE_DATA', '.')
        self._section = section
        scheme = ClassScheme['widerface']
        directory = os.path.join(self._widerface_data,
                                 'WIDER_' + self._section, 'images')
        super().__init__(key=key or f"wider-faces-{section}",
                         section=section, directory=directory, scheme=scheme,
                         description=f"WIDER Faces", **kwargs)
        self._annotations = None

    def __str__(self):
        return f'WIDER Faces ({self._section})'

    #
    # Preparation
    #

    def _prepare(self, **kwargs) -> None:
        # pylint: disable=arguments-differ
        """Prepare the WIDER Face dataset. This will provide in a list of
        all images provided by the dataset, either by reading in a
        prepared file, or by traversing the directory.
        """
        LOG.info("Preparing WiderFace[%r]: %s",
                 self.preparable, self.directory)
        cache = f"widerface_{self._section}_filelist.p"
        super()._prepare(filenames_cache=cache, **kwargs)
        self._scheme.prepare()
        self._prepare_annotations()

    def _unprepare(self):
        """Prepare the WIDER Face dataset. This will provide in a list of
        all images provided by the dataset, either by reading in a
        prepared file, or by traversing the directory.
        """
        self._annotations = None
        super()._unprepare()

    def _prepare_annotations(self):
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

        # check if annotations file exists
        filename = None
        if self._widerface_data is not None:
            filename = os.path.join(self._widerface_data, 'wider_face_split',
                                    'wider_face_train_bbx_gt.txt')
            if not os.path.isfile(filename):
                return  # file not found

        # load the annotations
        try:
            with open(filename, "r") as file:
                for filename in file:
                    filename = filename.rstrip()
                    lines = int(file.readline())
                    faces = []
                    for line_number in range(lines):
                        # x1, y1, w, h, blur, expression, illumination,
                        #    invalid, occlusion, pose
                        attributes = tuple(int(a)
                                           for a in file.readline().split())
                        if len(attributes) == 10:
                            faces.append(attributes)
                        else:
                            LOG.warning("bad annotation for '%s', line %d/%d':"
                                        "got %d instead of 10 values",
                                        filename, line_number,
                                        lines, len(attributes))
                    if lines == 0:
                        # images with 0 faces nevertheless have one
                        # line with dummy attributes -> just ignore that line
                        file.readline()
                    # Store all faces for the current file
                    self._annotations[filename] = faces
        except FileNotFoundError:
            self._annotations = {}

    #
    # Data
    #

    def _get_meta(self, data: Data, **kwargs) -> None:
        data.add_attribute('label', batch=True)
        super()._get_meta(data, **kwargs)

    def _get_data_from_file(self, data, filename: str) -> str:
        """
        Arguments
        ---------
        filename: str
            The relative filename.
        """
        super()._get_data_from_file(data, filename)
        regions = []
        for (pos_x, pos_y, width, height, blur, expression, illumination,
             invalid, occlusion, pose) in self._annotations[filename]:
            region = Region(BoundingBox(x=pos_x, y=pos_y,
                                        width=width, height=height),
                            blur=blur, expression=expression,
                            illumination=illumination,
                            invalid=invalid, occlusion=occlusion,
                            pose=pose)
            regions.append(region)
        data.label = regions


# FIXME[todo]
class W300(DataDirectory, Imagesource):
    """The 300 Faces In-the-Wild Challenge (300-W), form the ICCV 2013.
    The challenge targets facial landmark detection, using a 68 point
    annotation scheme.

    Besides 300-W, there are several other datasets annotated in the
    same scheme: AFW, FRGC, HELEN, IBUG, LPFW, and XM2VTS.

    For more information visit:
    https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _load_annotation(filename: str) -> Landmarks:
        """Parse the landmark annotation file.  Each image of the dataset is
        accompanied by a file with the same name und the suffix '.pts'
        providing the positions of the 68 points.

        """
        # The file has the following format:
        #
        #    version: 1
        #    n_points:  68
        #    {
        #    403.167108 479.842932
        #    407.333804 542.927159
        #    ...
        #    625.877482 717.615332
        #    }
        #
        with open(filename) as file:
            _ = file.readline().split(':')[1]  # version
            n_points = int(file.readline().split(':')[1])
            points = np.ndarray((n_points, 2))
            _ = file.readline()  # '{'
            for i in range(n_points):
                pos_x, pos_y = file.readline.rstrip().split(' ')
                points[i] = float(pos_x), float(pos_y)
        return Landmarks(points)
