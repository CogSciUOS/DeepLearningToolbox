from datasources import Datasource, Loop, Predefined, InputData

import numpy as np


class DataNoise(Predefined, Loop):
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

    def __init__(self, id: str="Noise", description: str=None,
                 shape: tuple=(100,100), **kwargs):
        """Create a new :py:class:`DataNoise`

        Arguments
        ---------
        shape: tuple
            Shape of the data to generate, e.g. (3,100,100) for RGB images.
        """
        description = description or f"<Noise Generator {shape}>"
        super().__init__(id=id, description=description, **kwargs)
        self._shape = shape


    @property
    def fetched(self):
        return True

    def _fetch(self, **kwargs):
        pass

    def _get_data(self) -> np.ndarray:
        """

        Result
        ------
        data: np.ndarray
            The input data.
        """
        return np.random.rand(*self._shape)

    def __str__(self):
        return "Noise"
