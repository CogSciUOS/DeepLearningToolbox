from observer import Observable, change

class Config(Observable):
    """A :py:class:Config object provides configuration data.  It is an
    :py:class:Observable, allowing :py:class:Engine and user
    interfaces to be notified on changes.

    """
    _config = {}

    def __init__(self, change_type: type, change_method: str,
                 default_change: str):
        super().__init__(change_type, change_method)
        self._values = {}
        self._default_change = default_change

    def __getattr__(self, name):
        if name in self._config:
            return self._values.get(name, self._config[name]['default'])
        else:
            raise AttributeError(name)

    @change
    def _helper_setattr(self, name, value):
        entry = self._config[name]
        if value != getattr(self, name):
            if value == entry['default']:
                del self._values[name]
            else:
                self._values[name] = value
            self.change(entry.get('change', self._default_change))

    def __setattr__(self, name, value):
        if name in self._config:
            self._helper_setattr(name, value)
        else:
            super().__setattr__(name, value)

    @change
    def assign(self, other):
        for name in self._config:
            self._helper_setattr(name, getattr(other, name))

    def __copy__(self):
        """Create a copy of this :py:class:Config object.
        This will copy the configuration values, but not the observers.
        """
        other = Config(self._change_type,
                       self._change_method,
                       self._default_change)
        other.assign(self)
        return other
