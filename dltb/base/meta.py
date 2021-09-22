"""Abstract base class :py:class:`Metadata` for different kind of metadata.
"""

# standard imports
import os

# Generic imports
import numpy as np

# toolbox imports
from .image import Region


class Metadata:
    """Metadata for a datum from a Datasource.
    """
    _regions = None

    def __init__(self, description: str = None, label: str = None,
                 **attributes) -> None:
        if description is not None:
            self._description = description
        if label is not None:
            self._label = label
        self.__dict__.update(attributes)

    def add_region(self, position, **attributes):
        """Add a spatial region with attributes to this
        :py:class:`Metadata` object.
        """
        if self._regions is None:
            self._regions = []
        self._regions.append(Region(position, **attributes))

    def has_regions(self):
        """Check if there are regions registered with this
        :py:class:`Metadata` object.
        """
        return self._regions is not None

    def __bool__(self) -> bool:
        """`True` if there is at least one region registered with this
        :py:class:`Metadata` object.
        """
        return self._regions is not None and len(self._regions) > 0

    def __len__(self) -> int:
        """The number of regions registered with this
        :py:class:`Metadata` object.
        """
        return 0 if self._regions is None else len(self._regions)

    @property
    def regions(self):
        """The list of regions registered with this :py:class:`Metadata`
        object.
        """
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
        """A description for this :py:class:`Metadata`
        object.
        """
        return self._description

    @property
    def label(self):
        """A label for this :py:class:`Metadata` object.
        """
        return self._label

    def has_attribute(self, name) -> bool:
        """Check if this :py:class:`Metadata` object as the given attribute.

        """
        return name in self.__dict__

    def set_attribute(self, name, value) -> None:
        """Set an attribute for this :py:class:`Metadata` object.
        """
        self.__dict__[name] = value

    def get_attribute(self, name):
        """Get an attribute value for this :py:class:`Metadata` object.
        """
        return self.__dict__[name]

    def __str__(self):
        description = ("Metadata " +
                       ("with" if self.has_regions() else "without") +
                       " regions. Attributes:")
        for name, value in self.__dict__.items():
            description += os.linesep + "  " + name + ": "
            if isinstance(value, np.ndarray):
                description += f"Array({value.shape}, dtype={value.dtype})"
            else:
                description += str(value)
        return description
