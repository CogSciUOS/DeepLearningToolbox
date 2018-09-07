import os
import os.path
import importlib
import importlib.util

from datasources import DataArray, DataDirectory, Predefined

class KerasDataSource(DataArray, Predefined):
    '''Data source for Keras builtin datasets.

    Keras provides some methods to access standard datasets via its
    keras.datasets API. This API will automatically download and
    unpack required data into ~/.keras/datasets/.

    '''
    KERAS_IDS = ['mnist', 'cifar10', 'cifar100', 'fashion_mnist']
    
    _name = None
    
    def __init__(self, name: str):
        if importlib.util.find_spec(f'keras.datasets.{name}') is None:
            raise ValueError(f'Unknown Keras dataset: {name}')
        self._name = name
        Predefined.__init__(self, name)

        
    def prepare(self):
        if self._array is not None:
            return # Just prepare once!
        
        name = self._name
        module = importlib.import_module(f'keras.datasets.{name}')
        data = module.load_data()
        
        # The Keras data are provided as a pair of pair of arrays:
        #   (x_train, y_train), (x_test, y_test)
        # That is: data[0][0] are the input data from the training set
        # and data[0][1] are the correspondng test data
        self.setArray(data[0][0], f'keras.datasets.{name}')
        self.add_target_values(data[0][1])
        # Also load the labels if available
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

    def check_availability():
        '''Check if this Datasource is available.
        
        Returns
        -------
        True if the DataSource can be instantiated, False otherwise.
        '''
        return True

