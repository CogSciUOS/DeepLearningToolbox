from qtgui.datasources import DataArray

class DataSet(DataArray):

    def __init__(self, name: str = None):
        super().__init__()
        self.load(name)

    def load(self, name: str):
        if name == 'mnist':
            from keras.datasets import mnist
            data = mnist.load_data()[0][0]
            self.setArray(data, 'MNIST')
        else:
            raise ValueError(f'Unknown dataset: {name}')

    def getName(self) -> str:
        return 'MNIST'



