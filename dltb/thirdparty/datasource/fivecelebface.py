"""The five celebrities face recognition dataset.
"""

# standard imports
import os

# toolbox imports
from dltb.datasource import Imagesource, Sectioned, DataDirectory


class FiveCelebFace(DataDirectory, Imagesource, Sectioned,
                    sections={'train', 'val'}):
    # pylint: disable=too-many-ancestors
    """The 5 Celebrity Faces Dataset. [1]

    This is a small demo dataset for face recognition systems. It
    contains images (around 20 per person) for five celebrities (Ben
    Afflek, Elton John, Jerry Seinfeld, Madonna, and Mindy Kaling).

    The dataset is split into two sections: `train` and `val`.
    Each section provides data for each of the 5 classes in
    a separate subdirectory, the directory name providing the
    class label as string (`ben_afflek`, `elton_john`, `jerry_seinfeld`,
    `madonna`, and `mindy_kaling`).

    The base directory of the dataset (containing the subdirectories
    `train` and `val`) can be provided via the environment variable
    `FIVE_CELEB_DATA`.

    [1] https://www.kaggle.com/dansbecker/5-celebrity-faces-dataset
    """

    def __init__(self, key: str = None, section: str = 'train',
                 **kwargs) -> None:
        """Initialize the 5 Celebrity Faces Datasource.

        Arguments
        ---------
        section: str
            The section of the dataset: either `train` or `val`.
        """
        super().__init__(key=key or f"5-celeb-{section}", section=section,
                         description=f"5 Celebrity Faces dataset", **kwargs)
        five_celeb_data = os.getenv('FIVE_CELEB_DATA', '.')
        self.directory = os.path.join(five_celeb_data, section)
