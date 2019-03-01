from base.observer import Observable, change

class Config(Observable,
             changes=['config_changed'],
             default='config_changed',
             method='configChanged'):
    """A :py:class:Config object provides configuration data.  It is an
    :py:class:Observable, allowing :py:class:Engine and user
    interfaces to be notified on changes.

    """
    _config = {}

    def __init__(self):
        super().__init__()
        self._values = {}

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
        cls = self.__class__
        other = cls()
        # FIXME[concept]: we have to create a new instance of the
        #  - hence we have to have some idea what arguments we have
        #    to give to the constructor ...
        #    self._change_type,
        #    self._change_method,
        #    self._default_change)
        other.assign(self)
        return other
