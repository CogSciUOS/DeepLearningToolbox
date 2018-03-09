from qtgui.datasources import DataArray

class DataSet(DataArray):
    '''Data source for Keras builtin datasets'''

    def __init__(self, name: str = None):
        super().__init__()
        self._name = name
        self.load(name)

    def load(self, name: str):
        try:
            from importlib import import_module
            dataset = import_module(f'keras.datasets.{name}')
            data = dataset.load_data()[0][0]
            self.setArray(data, name)
        except ImportError:
            raise ValueError(f'Unknown dataset: {name}')

    def getName(self) -> str:
        return self._name

    @staticmethod
    def getKerasDatasets():
        dataset_names = ['mnist', 'cifar10', 'cifar100', 'fashion_mnist']
        return dataset_names






