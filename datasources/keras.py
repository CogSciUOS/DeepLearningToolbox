import os
import os.path
import importlib
import importlib.util

from datasources import DataArray, DataDirectory, Predefined


class KerasDatasource(DataArray, Predefined):
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

    _keras_dataset_name: str
        The name of the Keras dataset (one of those listed in
        KERAS_IDS).
    """
    KERAS_IDS = ['mnist', 'cifar10', 'cifar100', 'fashion_mnist']

    _keras_dataset_name = None

    def __init__(self, name: str):
        """Initialize a keras dataset.

        Arguments
        ---------
        name: str
            Name of the Keras dataset. This can be any name listed
            in KERAS_IDS.
            
        Raises
        ------
        ValueError:
            There does not exist a package keras.datasets.{name}.
        """
        if importlib.util.find_spec(f'keras.datasets.{name}') is None:
            raise ValueError(f'Unknown Keras dataset: {name}')
        self._keras_dataset_name = name
        DataArray.__init__(self, description=f"Keras Datasoure '{name}'")
        Predefined.__init__(self, name)

    def prepare(self):
        if self.prepared:
            return  # nothing to do (avoid preparing twice

        name = self._keras_dataset_name
        module = importlib.import_module(f'keras.datasets.{name}')
        data = module.load_data()

        # The Keras data are provided as a pair of pairs of arrays:
        #   (x_train, y_train), (x_test, y_test)
        # That is: data[0][0] are the input data from the training set
        # and data[0][1] are the corresponding labels.
        # data[1][0] and data[1][1] are test data and labels.
        self.setArray(data[0][0], f'keras.datasets.{name}')
        self.add_target_values(data[0][1])

        #
        # Also load the labels if available
        #
        from keras.utils.data_utils import get_file
        from six.moves import cPickle
        if name == 'cifar10':
            path = get_file('cifar-10-batches-py', None)
            with open(os.path.join(path, "batches.meta"), 'rb') as file:
                d = cPickle.load(file)
            self.add_target_labels(d['label_names'])
        elif name == 'cifar100':
            path = get_file('cifar-100-python', None)
            with open(os.path.join(path, "meta"), 'rb') as file:
                d = cPickle.load(file)
            self.add_target_labels(d['fine_label_names'])
            # there is also 'coarse_label_names'
            # with 20 categories

        self.change('state_changed')

    def check_availability():
        """Check if this Datasource is available.

        Returns
        -------
        True if the Datasource can be instantiated, False otherwise.
        """
        return True

    def __str__(self):
        # return Predefined.__str__(self) + ': ' + DataArray.__str__(self)
        return self._keras_dataset_name + ': ' + DataArray.__str__(self)
