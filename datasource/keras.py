"""A :py:class:`Datasource` based on predefined keras datasets.
"""

# Generic imports
import os.path
import importlib.util

# cifar10 and cifar100 labels are provided as pickled files
# FIXME[todo]: Six: Python 2 and 3 Compatibility Library - do we need this?
# From the docs: ""Some modules which had two implementations have
# been merged in Python 3. For example, cPickle no longer exists in
# Python 3; it was merged with pickle. In these cases, fetching the
# fast version will load the fast one on Python 2 and the merged
# module in Python 3.""
from six.moves import cPickle

# toolbox imports
from .datasource import Imagesource
from .array import LabeledArray


class KerasDatasource(LabeledArray, Imagesource):
    # pylint: disable=too-many-ancestors
    """Data source for Keras builtin datasets.

    Keras provides some methods to access standard datasets via its
    keras.datasets API. This API will automatically download and
    unpack required data into ~/.keras/datasets/.


    Attributes
    ----------

    **Class Attributes**

    KERAS_IDS: list
        A list of valid names for :py:class:`KerasDatasource`\ s.
        For each name there has to exists a package called
        keras.datasets.{name}.
    KERAS_DATA: dict
        A dictionary holding all loaded keras datasets.
        Maps the ID of the dataset to the actual data.
        The Keras data are provided as a pair of pairs of arrays:

           (x_train, y_train), (x_test, y_test)

        That is: data[0][0] are the input data from the training set
        and data[0][1] are the corresponding labels.
        data[1][0] and data[1][1] are test data and labels.

    **Instance Attributes**

    _keras_dataset_name: str
        The name of the Keras dataset (one of those listed in
        KERAS_IDS).
    _section_index: int
        Index for the section of the dataset: 0 for train and 1 for test.
    """
    KERAS_IDS = ['mnist', 'cifar10', 'cifar100', 'fashion_mnist']

    KERAS_DATA = {}

    _keras_dataset_name: str = None
    _section_index: int = 0
    _variant: str = None

    def __init__(self, name: str, section: str = 'train',
                 key: str = None, **kwargs) -> None:
        """Initialize a keras dataset.

        Arguments
        ---------
        name: str
            Name of the Keras dataset. This can be any name listed
            in KERAS_IDS.

        section: str
            The section of the dataset, either 'train' or 'test'

        Raises
        ------
        ValueError:
            There does not exist a package keras.datasets.{name}.
        """
        super().__init__(key=key or f"{name}-{section}",
                         description=f"Keras Datasoure '{name}-{section}'",
                         **kwargs)
        self._keras_dataset_name = name
        self._section_index = 0 if section == 'train' else 1

    @property
    def keras_module_name(self) -> str:
        """Fully qualified name of the keras module representing this dataset.
        """
        return self._keras_module_name(self, 'keras.datasets.' +
                                       self._keras_dataset_name)

    @staticmethod
    def _keras_module_name(self, name: str) -> str:
        """Fully qualified name of the given keras module (either
        in the `keras` package or in the `tensorflow.keras` reimplementation).

        Arguments
        ---------
        name: str
            The fully qualified keras module name
            (e.g. `keras.datasets.mnist`)
        """
        return 'tensorflow.' + name
        # return name

    def _preparable(self) -> bool:
        """Check if this Datasource is available.

        Returns
        -------
        True if the Datasource can be instantiated, False otherwise.
        """
        module_spec = importlib.util.find_spec(self.keras_module_name)
        return module_spec is not None and super()._preparable
 
    def _prepare(self):
        """Prepare this :py:class:`KerasDatasource`.
        This includes import the corresponding keras module and
        loading label names if available.
        """
        super()._prepare()

        name = self._keras_dataset_name
        if name not in self.KERAS_DATA:
            module = importlib.import_module(self.keras_module_name)
            self.KERAS_DATA[name] = module.load_data()
       
        data = self.KERAS_DATA[name][self._section_index]
        self._array = data[0]
        self._description = f'keras.datasets.{name}'

        if name == 'cifar100':
            # there exist two variants of the cifar100 dataset:
            # the "fine" one with 100 labels and a "coarse" one with
            # only 20 labels.
            self._variant = 'fine'

    def _prepare_labels(self) -> None:
        """Prepare the labels for a Keras datasource. Has to be called
        before the labels can be used.
        """

        #
        # Set the labels
        #
        name = self._keras_dataset_name
        data = self.KERAS_DATA[name][self._section_index]
        if name == 'cifar100' and self._variant == 'coarse':
            # FIXME[todo]: we have to find the mapping from "fine"
            # labels to "coarse" labels as provided on
            # https://www.cs.toronto.edu/~kriz/cifar.html
            super()._prepare_labels(labels=data[1]//5)
        else:
            super()._prepare_labels(data[1])

        import_name = self._keras_module_name('keras.utils.data_utils')
        data_utils = importlib.import_module(import_name)

        # Textual labels are provided in different ways for the different
        # Keras datasets.
        name = self._keras_dataset_name
        if name == 'cifar10':
            path = data_utils.get_file('cifar-10-batches-py', None)
            with open(os.path.join(path, "batches.meta"), 'rb') as file:
                meta = cPickle.load(file)
            self.add_label_format('text', meta['label_names'])
        elif name == 'cifar100':
            path = data_utils.get_file('cifar-100-python', None)
            with open(os.path.join(path, "meta"), 'rb') as file:
                meta = cPickle.load(file)
            if self._variant == 'fine':
                self.add_label_format('text', meta['fine_label_names'])
            elif self._variant == 'coarse':
                # there is also 'coarse_label_names' with 20 categories
                # label [0..100] -> label//5 [0..20]
                self.add_label_format('coarse_text', meta['coarse_label_names'])

    def __str__(self):
        # return Predefined.__str__(self) + ': ' + DataArray.__str__(self)
        # return self._keras_dataset_name + ': ' + DataArray.__str__(self)
        return "Keras Dataset: " + self._keras_dataset_name
