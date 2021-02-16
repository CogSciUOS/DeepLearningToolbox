"""Integration of scikit-learn (sklearn) into the deep learning toolbox.
"""

# third party imports
import numpy as np
from sklearn.datasets import fetch_lfw_people  # , fetch_lfw_pairs

# toolbox imports
from ...datasource import Datasource, Imagesource, LabeledArray
from ...tool.classifier import ClassScheme

Datasource.register_instance('lfw-sklearn', __name__, 'LFW')


class LFW(Imagesource, LabeledArray):
    # pylint: disable=too-many-ancestors,too-many-instance-attributes
    """fetch_lfw_people(
    *,
    data_home=None,
    funneled=True,
    resize=0.5,
    min_faces_per_person=0,
    color=False,
    slice_=(slice(70, 195, None), slice(78, 172, None)),
    download_if_missing=True,
    return_X_y=False,


    from dltb.thirdparty.sklearn import LFW
    lfw = LFW(min_faces_per_person=70)
    lfw.prepare()
    d=lfw[1]


    Parameters
    ----------
    data_home: optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    funneled: boolean, optional, default: True
        Download and use the funneled variant of the dataset.

    resize: float, optional, default 0.5
        Ratio used to resize the each face picture.

    min_faces_per_person: int, optional, default 0
        The extracted dataset will only retain pictures of people that have at
        least `min_faces_per_person` different pictures.

    color: boolean, optional, default False
        Keep the 3 RGB channels instead of averaging them to a single
        gray level channel. If color is True the shape of the data has
        one more dimension than the shape with color = False.

    slice_: optional
        Provide a custom 2D slice (height, width) to extract the
        'interesting' part of the jpeg files and avoid use statistical
        correlation from the background

    download_if_missing : optional, True by default
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    return_X_y: bool, default=False.
        If True, returns ``(dataset.data, dataset.target)`` instead of a Bunch
        object. See below for more information about the `dataset.data` and
        `dataset.target` object.
    """

    def __init__(self, funneled: bool = True, color: bool = True,
                 resize: float = 0.5, min_faces_per_person: int = 0,
                 data_home=None, **kwargs) -> None:
        # pylint: disable=too-many-arguments
        # download_if_missing: bool = True,
        super().__init__(**kwargs)
        self._color = color
        self._resize = resize
        self._min_faces_per_person = min_faces_per_person
        self._funneled = funneled
        self._data_home = data_home
        # self._download_if_missing = download_if_missing
        self._lfw_people = None

    def _prepare(self) -> None:
        super()._prepare()
        self._lfw_people = \
            fetch_lfw_people(min_faces_per_person=self._min_faces_per_person,
                             resize=self._resize, color=self._color,
                             funneled=self._funneled)
        # note: the array will be of type float32, with values in range
        # 0.0 to 255.0

        # FIXME[bug]: some pylint issue her
        # pylint: disable=no-member
        self._array = self._lfw_people.images/255.0

        scheme = ClassScheme()
        scheme.add_labels(self._lfw_people.target_names, name='text')
        self._set_labels(self._lfw_people.target, scheme=scheme)

    @property
    def lfw_people(self):
        """A dictionary-like object, with the following attributes.

        data: numpy array of shape (SAMPLES, 2914)
            Each row corresponds to a ravelled face image
            of original size 62 x 47 pixels.
            Changing the ``slice_`` or resize parameters will change the
            shape of the output.
        images : numpy array of shape (SAMPLES, 62, 47)
            Each row is a face image corresponding to one of the 5749 people in
            the dataset. Changing the ``slice_``
            or resize parameters will change the shape of the output.
        target : numpy array of shape (SAMPLES,)
            Labels associated to each face image.
            Those labels range from 0-5748 and correspond to the person IDs.
        DESCR: str
            Description of the Labeled Faces in the Wild (LFW) dataset.

        Remark: data and image will be of dtype numpy.float32, with values
        in the range 0.0 to 255.0.  With `color=True` the color channel will
        be last, that is shape (13233, 62, 47, 3), in RGB color space.
        """
        return self._lfw_people
