from .network import Network


class Autoencoder(Network, method='network_changed'):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._encoder = None
        self._decoder = None

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder


class VariationalAutoencoder(Autoencoder):

    def sampleCode(self, n=1):
        pass

    def sampleData(self, n=1):
        pass

    def sampleCodeFor(self, input, n=1):
        pass

    def sampleDataFor(self, input, n=1):
        pass
