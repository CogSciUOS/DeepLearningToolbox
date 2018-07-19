import os
import os.path
import importlib
import importlib.util

from datasources import DataArray, DataDirectory

class DataSet:
    '''Data source for Keras builtin datasets.

    Keras provides some methods to access standard datasets via its
    keras.datasets API. This API will automatically download and
    unpack required data into ~/.keras/datasets/.

    '''
    @staticmethod
    def load(name: str):
        if name != 'imagenet' and importlib.util.find_spec(f'keras.datasets.{name}') is not None:
            dataset = importlib.import_module(f'keras.datasets.{name}')
            data = dataset.load_data()
            # (x_train, y_train), (x_test, y_test)
            dataset = DataArray(data[0][0], f'keras.datasets.{name}')
            dataset.add_target_values(data[0][1])

            # Also load the labels if available
            from keras.utils.data_utils import get_file
            from six.moves import cPickle
            if name == 'cifar10':
                path = get_file('cifar-10-batches-py', None)
                with open(os.path.join(path, "batches.meta"), 'rb') as file:
                    d = cPickle.load(file)
                dataset.add_target_labels(d['label_names'])
            elif name == 'cifar100':
                path = get_file('cifar-100-python', None)
                with open(os.path.join(path, "meta"), 'rb') as file:
                    d = cPickle.load(file)
                dataset.add_target_labels(d['fine_label_names'])
                # there is also 'coarse_label_names'
                # with 20 categories

        elif name == 'imagenet':
            dir = os.path.join(os.environ.get('IMAGENET_DATA'),"val")
            dataset = DataDirectory(dir)
        else:
            raise ValueError(f'Unknown dataset: {name}')
        return dataset


    @staticmethod
    def getDatasets(quick = False):
        dataset_names = []

        # Check for Keras datasets
        # Attention: even finding the module spec for the
        # keras.dataset modules will also load the keras backend (which
        # is hugh and slow and may actually not be needed)!
        if quick:
            dataset_names.extend(['mnist', 'cifar10', 'cifar100'])
        else:
            for d in ['mnist', 'cifar10', 'cifar100', 'fashion_mnist']:
                if importlib.util.find_spec(f'keras.datasets.{d}') is not None:
                    dataset_names.append(d)
        

        # Check for ImageNet
        if os.environ.get('IMAGENET_DATA') is not None:
            dataset_names.append('imagenet')

        return dataset_names
