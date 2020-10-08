"""
.. moduleauthor:: Ulf Krumnack

.. module:: dltb.base.fail

This module contains a base definition for the :py:class:`Config` class.
"""

# standard imports
from typing import Any

from dltb.base.observer import Observable, change


class Config(Observable, method='configChanged', changes={'config_changed'}):
    """A :py:class:Config object provides configuration data.  It is an
    :py:class:Observable, allowing :py:class:Engine and user
    interfaces to be notified on changes.

    """

    def __init_subclass__(cls: type, method: str = None, changes: list = None,
                          default: str = None) -> None:
        Observable.__init_subclass__.__func__(cls, method, changes)
        if default is not None:
            cls._default_change = default

    _default_change: str = 'config_changed'
    _config: dict = {}

    def __init__(self) -> None:
        super().__init__()
        self._values = {}

    def __getattr__(self, name) -> Any:
        if name not in self._config:
            raise AttributeError(name)

        return self._values.get(name, self._config[name]['default'])

    @change
    def _helper_setattr(self, name: str, value: Any) -> None:
        entry = self._config[name]
        if value != getattr(self, name):
            if value == entry['default']:
                del self._values[name]
            else:
                self._values[name] = value
            self.change(entry.get('change', self._default_change))

    def __setattr__(self, name, value) -> None:
        if name in self._config:
            self._helper_setattr(name, value)
        else:
            super().__setattr__(name, value)

    @change
    def assign(self, other: 'Config') -> None:
        """Add config value from another :py:class:`config` object.
        """
        for name in self._config:
            self._helper_setattr(name, getattr(other, name))

    def __copy__(self) -> 'Config':
        """Create a copy of this :py:class:Config object.
        This will copy the configuration values, but not the observers.
        """
        cls = self.__class__
        other = cls()
        other.assign(self)
        return other
