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
from dltb.datasource import DataDirectory

# logging
LOG = logging.getLogger(__name__)


class EnvironmentalSoundClassification(DataDirectory):
    """A :py:class:`Datasource` for accessing the
    Dataset for Environmental Sound Classification [1], ECS-10 and ECS-50

    [1] doi:10.7910/DVN/YDEPUT
    """

    def __init__(self, key: str = None, esc_data: str = None,
                 **kwargs) -> None:
        """Initialize the Environmental Sound Classification (ESC) dataset.

        Parameters
        ----------
        esc_data: str
            The path to the ESC root directory. This directory
            should contain the (10 or 50) subdirectories holding
            sound files for the respective classes.
        """
        # directory = '/space/data/ESC/ESC-10'  # FIXME[hack]
        if esc_data is None:
            esc_data = '/space/data/ESC/ESC-10'
        description = "Environmental Sound Classification"
        super().__init__(key=key or "esc",
                         directory=esc_data,
                         description=description,
                         label_from_directory='name',
                         **kwargs)
        LOG.info("Initialized the Environmental Sound Classification "
                 "dataset (directory=%s)", self.directory)

    def _prepare(self) -> None:
        super()._prepare()
        LOG.info("Prepared the Environmental Sound Classification "
                 "dataset (directory=%s)", self.directory)
