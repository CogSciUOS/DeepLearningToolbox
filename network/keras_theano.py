
# not implemented yet!


from .keras import Network as KerasNetwork

class Network(KerasNetwork):

    @classmethod
    def import_framework(cls):
        # The only way to configure the keras backend appears to be
        # via environment variable. We thus inject one for this
        # process. Keras must be loaded after this is done
        super(Network, cls).import_framework()

    def __init__(self, **kwargs):
        """
        Load Keras model.
        Parameters
        ----------
        modelfile_path
            Path to the .h5 model file.
        """
        raise RuntimeError('Currently, only TF backend is supported')
