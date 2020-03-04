from util.image import Region

import os
import numpy as np

class Metadata:
    """Metadata for a datum from a Datasource.
    """
    _regions = None

    def __init__(self, description: str=None, label: str=None, **attributes):
        if description is not None:
            self._description = description
        if label is not None:
            self._label = label
        self.__dict__.update(attributes)

    def add_region(self, position, **attributes):
        if self._regions is None:
            self._regions = []
        self._regions.append(Region(position, **attributes))

    def has_regions(self):
        return self._regions is not None

    @property
    def regions(self):
        return self._regions

    def scale(self, factor) -> None:
        """Scale all positions of this metadata by a given factor.

        Arguments
        ---------
        factor:
            The scaling factor. This can either be a float, or a pair
            of floats in which case the first number is the horizontal (x)
            scaling factor and the second numger is the vertical (y)
            scaling factor.

        """
        if self.has_regions():
            for region in self.regions:
                region.scale(factor)

    @property
    def description(self):
        return self._description

    @property
    def label(self):
        return self._label

    def has_attribute(self, name) -> bool:
        return name in self.__dict__

    def set_attribute(self, name, value) -> None:
        self.__dict__[name] = value

    def get_attribute(self, name):
        return self.__dict__[name]

    def __str__(self):
        s = "Metadata:"
        for name, value in self.__dict__.items():
            s += os.linesep + "  " + name + ": "
            if isinstance(value, np.ndarray):
                s += f"Array({value.shape}, dtype={value.dtype})"
            else:
                s += str(value)
        return s
