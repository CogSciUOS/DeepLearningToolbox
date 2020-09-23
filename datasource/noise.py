"""A datasource providing different types of noise.
"""

# standard imports
from typing import Tuple

# third party imports
import numpy as np

# toolbox imports
from dltb.base.data import Data
from .datasource import Imagesource, Loop, Snapshot, Random


class Noise(Imagesource, Random, Loop, Snapshot):
    # pylint: disable=too-many-ancestors
    """A :py:class:`Noise` is a :py:class:`Datasource` that
    provides different kinds of noise.

    Attributes
    ----------
    shape: tuple
        Shape of the noise array to generate.
    distribution: str
        Either `uniform` or `normal`.
    """

    def __init__(self, key: str = "Noise", description: str = None,
                 shape: Tuple[int, ...] = (100, 100),
                 distribution: str = 'uniform', **kwargs) -> None:
        """Create a new :py:class:`Noise`

        Arguments
        ---------
        shape: tuple
            Shape of the data to generate, e.g. (3,100,100) for RGB images.
        """
        description = description or f"<Noise Generator {shape}>"
        super().__init__(key=key, description=description,
                         random_generator='numpy', **kwargs)
        self.shape = shape
        self.distribution = distribution

    def __str__(self):
        return "Noise"

    #
    # Data
    #

    def _get_random(self, data: Data, shape: Tuple[int, ...] = None,
                    distribution: str = None, **kwargs) -> None:
        # pylint: disable=arguments-differ
        """Generate a random datapoint. Parameters can be given as arguments
        or will be taken from object attributes.

        Arguments
        ---------
        shape: tuple
            The shape of the data to be generated.

        distribution: str
            The distribution (either `uniform` or `normal`).
        """
        shape = shape or self.shape
        distribution = distribution or self.distribution
        data.array = (np.random.rand(*shape) if distribution == 'uniform' else
                      np.random.randn(*shape))

    def _get_snapshot(self, data, snapshot: bool = True, **kwargs) -> None:
        self._get_random(data, **kwargs)
        super()._get_snapshot(data, snapshot, **kwargs)
