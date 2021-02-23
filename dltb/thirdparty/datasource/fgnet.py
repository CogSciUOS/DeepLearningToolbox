"""The FG-Net dataset.
"""

# standard imports
from typing import Tuple
import os
import logging

# third party imports
import numpy as np

# toolbox imports
from dltb.base.data import Data
from dltb.base.image import Region, Landmarks
from dltb.datasource import Imagesource, DataDirectory

# logging
LOG = logging.getLogger(__name__)


class FGNet(DataDirectory, Imagesource):
    """

    Installation
    ------------

    The file `FGNET.zip` unpacks to a folder `FGNET/` containing
    three subfolders:

    images:
        Face image files, JPEG format, different sizes (around 450x450 pixel)
    points:
        Files containing facial landmarks.
    Data_files:
        Data files (purpose not clear to me, most are empty).
    """

    def __init__(self, key: str = None, **kwargs) -> None:
        """Initialize the WIDER Face Datasource.
        """
        self._fgnet_directory = '/net/projects/data/FGNET'
        image_directory = os.path.join(self._fgnet_directory, 'images')
        super().__init__(key=key or "FG-Net", directory=image_directory,
                         description="FG-Net data", **kwargs)
        self._annotations = None

    @staticmethod
    def _load_annotation(filename: str) -> Tuple[str, Landmarks]:
        """Load annotation (facial landmark information) from a file.

        The annotation file consist of lines. The first line specifies
        the version and the second line the number of point (68),
        followed by an opening brace '{', 68 lines of pairs of float
        numbers (separated by space), and a closing brace '}'. Example:

        version: 1
        n_points: 68
        {
        65.3152 286.528
        71.8811 325.722
        78.6391 364.824
        ...
        179.133 398.014
        168.406 356.184
        }

        Arguments
        ---------
        filename: str
            The absolute name of the annotation file.
        """
        with open(filename) as file:
            # Ignore the first line (image name)
            _ignore = file.readline()  # version: 1
            n_points = int(file.readline().split(':')[-1])
            _ignore = file.readline() # '{'
            points = np.ndarray((n_points, 2))
            for i in range(n_points):
                x_pos, y_pos = file.readline().split(' ')
                points[i] = float(x_pos), float(y_pos)
        return Landmarks(points)

    #
    # Data
    #

    def _get_meta(self, data: Data, **kwargs) -> None:
        """Prepare the meta data of the :py:class:`Data` object as
        a preparation for getting the data.
        """
        data.add_attribute('label', batch=True)
        super()._get_meta(data, **kwargs)

    def _get_data_from_file(self, data: Data, filename: str) -> None:
        """Get the actual data. This includes reading the image data from
        a file and parsing the landmark data file.
        """
        super()._get_data_from_file(data, filename)

        # provide the metadata (landmarks)
        basename = os.path.basename(filename).lower()
        if basename.endswith('.jpg'):
            basename = basename[:-len('.jpg')]

        annotation_filename = \
            os.path.join(self._fgnet_directory, 'points', basename + '.pts')
        landmarks = self._load_annotation(annotation_filename)
        data.label = Region(landmarks)
