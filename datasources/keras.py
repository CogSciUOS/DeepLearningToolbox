from datasources import LabeledArray, Predefined

import os
import os.path
import importlib
import importlib.util


class KerasDatasource(LabeledArray, Predefined):
    """Data source for Keras builtin datasets.

    Keras provides some methods to access standard datasets via its
    keras.datasets API. This API will automatically download and
    unpack required data into ~/.keras/datasets/.


    Class attributes
    ----------------
    KERAS_IDS: list
        A list of valid names for :py:class:`KerasDatasource`s.
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

    Attributes
    ----------
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

    def __init__(self, name: str, section:str = 'train', **kwargs):
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
        if importlib.util.find_spec(f'keras.datasets.{name}') is None:
            raise ValueError(f'Unknown Keras dataset: {name}')
        super().__init__(id=f"{name}-{section}",
                         description=f"Keras Datasoure '{name}-{section}'",
                         **kwargs)
        self._keras_dataset_name = name
        self._section_index = 0 if section == 'train' else 1

    def _prepare_keras_dataset(self):
        name = self._keras_dataset_name
        if not name in self.KERAS_DATA:
            module = importlib.import_module(f'keras.datasets.{name}')
            if not name in self.KERAS_DATA:
                self.KERAS_DATA[name] = module.load_data()

    def _prepare_data(self):
        self._prepare_keras_dataset()

        name = self._keras_dataset_name
        data = self.KERAS_DATA[name][self._section_index]
        self.set_data_array(data[0], f'keras.datasets.{name}')

        if name == 'cifar100':
            # there exist two variants of the cifar100 dataset:
            # the "fine" one with 100 labels and a "coarse" one with
            # only 20 labels. 
            self._variant = 'fine'

    def _prepare_labels(self) -> None:
        """Prepare the labels for a Keras datasource. Has to be called
        before the labels can be used.
        """
        self._prepare_keras_dataset()

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

        from keras.utils.data_utils import get_file

        # cifar10 and cifar100 labels are provided as pickled files
        from six.moves import cPickle

        # Textual labels are provided in different ways for the different
        # Keras datasets.
        name = self._keras_dataset_name
        if name == 'cifar10':
            path = get_file('cifar-10-batches-py', None)
            with open(os.path.join(path, "batches.meta"), 'rb') as file:
                d = cPickle.load(file)
            self.add_label_format('text', d['label_names'])
        elif name == 'cifar100':
            path = get_file('cifar-100-python', None)
            with open(os.path.join(path, "meta"), 'rb') as file:
                d = cPickle.load(file)
            if self._variant == 'fine':
                self.add_label_format('text', d['fine_label_names'])
            elif self._variant == 'coarse':
                # there is also 'coarse_label_names' with 20 categories
                # label [0..100] -> label//5 [0..20]
                self.add_label_format('coarse_text', d['coarse_label_names'])

    def check_availability():
        """Check if this Datasource is available.

        Returns
        -------
        True if the Datasource can be instantiated, False otherwise.
        """
        return True

    def __str__(self):
        # return Predefined.__str__(self) + ': ' + DataArray.__str__(self)
        # return self._keras_dataset_name + ': ' + DataArray.__str__(self)
        return "Keras Dataset: " + self._keras_dataset_name
