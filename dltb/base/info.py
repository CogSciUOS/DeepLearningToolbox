"""An API for providing information.
"""

# standard imports
from typing import Union, Callable, Any, Optional, Iterable
from enum import Enum

# toolbox imports
from .prepare import Postinitializable
from .mixin import Mixin

Unit = Enum('Unit', ['MEMORY', 'TEMPERATURE'])


class Info:
    """A named piece of information.  This will be a value that can either
    be static (fixed) or dynamic.  A dynamic value may change over
    time, that is reading it out multiple times may result in
    different values.  Typical dynamical values are device
    temperature, amount of allocated memory, the load state of a
    component, etc.  Fixed values are the maximal tolarable
    temperature of a device, the amount of memory available, or the
    version of a package.

    Arguments
    ---------
    name:
        The name by which this info object is identified
    value:
        The value for the `Info` object
    title:
        A title to display when showing this `Info`
    description:
        A more detailed description of the `Info`
    group:
    unit:
    """
    unit: Optional[Unit] = None

    def __init__(self, name: str, value: Union[Callable, Any],
                 title: Optional[str] = None,
                 description: Optional[str] = None,
                 group: Optional[str] = None,
                 unit: Optional[Unit] = None) -> None:
        # pylint: disable=too-many-arguments
        self.name = name
        self._value = value
        self.title = name if title is None else title
        self.description = description
        self.group = group
        if unit is not None:
            self.unit = unit

    @property
    def value(self) -> Any:
        """The value of this `Info` object.
        """
        return self._value() if self.is_dynamic else self._value

    @property
    def is_dynamic(self) -> bool:
        """A flag indicating if the :py:prop:`value` of this `Info` object is
        dynamic (`True`) or if it is fixed (`False`). 

        """
        return isinstance(self._value, Callable)

    def format_value(self) -> str:
        """Format the value of this `Info` object.  Formatting takes the type
        of the value (like time, memory, temperature, etc.) into
        account.

        """
        value = f"{self.value}"
        if self.unit is not None:
            if self.unit == Unit.MEMORY:
                value += " bytes"
            elif self.unit == Unit.TEMPERATURE:
                value += " °C"
        return value

    def __str__(self) -> str:
        return self.title + ': ' + self.format_value()


class InfoSource(Mixin, Postinitializable):
    """An object that can provide information.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._information = {}

    def __post_init__(self) -> None:
        super().__post_init__()
        self.initialize_info()

    def initialize_info(self) -> None:
        """Initialize the `InfoSource` by adding `Info` objects, usually
        via the :py:meth:`addInfo` method.
        """
        # to be implemented by subclases

    def get_info(self, name: str) -> Info:
        """Get information for a given key.
        """
        return self._information[name]

    def info_names(self, **kwargs) -> Iterable[str]:
        """Enumerate information names.
        """
        for name, info in self._information.items():
            if self._info_filter(info, **kwargs):
                yield name

    def _info_filter(self, info: Info,
                     group: Optional[str] = None,
                     dynamic: Optional[bool] = None) -> bool:
        if group is not None and info.group != group:
            return False
        if dynamic is not None and info.dynamic != dynamic:
            return False
        return True

    def add_info(self, name: str, value: Union[Callable, Any],
                  **kwargs) -> None:
        """Add a new :py:class:`Info` object to this `InfoSource`.

        Arguments
        ---------
        name:
            The name of the `Info` object to be created.
        value:
            The value for that `Info` object, either a fixed value,
            or a function providing a dynamic value.
        """
        self._information[name] = Info(name, value, **kwargs)

    def __str__(self):
        return "\n".join(map(str, self._information.values()))
