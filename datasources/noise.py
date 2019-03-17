from datasources import Datasource, Predefined, InputData

import numpy as np


class DataNoise(Datasource, Predefined):
    """A :py:class:`DataNoise` is a :py:class:`DataNoise` that
    provides different kinds of noise.

    Attributes
    ----------
    _seed: 
        Seed for the random number generator.
    _shape: 
        Shape of the noise array to generate.
    """
    _shape: tuple = None

    def __init__(self, shape: tuple):
        """Create a new :py:class:`DataNoise`

        Arguments
        ---------
        shape: tuple
            Shape of the data to generate, e.g. (3,100,100) for RGB images.
        """
        super().__init__(f"<Noise Generator {shape}>")
        Predefined.__init__(self, "Noise")
        self._shape = shape

    def __getitem__(self, index: int) -> InputData:
        """

        Result
        ------
        data: np.ndarray
            The input data.
        label:
            The associated label, if known, None otherwise.
        """
        data = np.random.rand(*self._shape)
        return InputData(data, None)
