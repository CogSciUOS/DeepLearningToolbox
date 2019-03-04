from base.observer import Observable, change

class Config(Observable, method='configChanged', changes=['config_changed']):
    """A :py:class:Config object provides configuration data.  It is an
    :py:class:Observable, allowing :py:class:Engine and user
    interfaces to be notified on changes.

    """

    def __init_subclass__(cls: type, method: str=None, changes: list=None,
                          default: str=None):
        Observable.__init_subclass__.__func__(cls, method, changes)
        if default is not None:
            cls._default_change = default

    _default_change:str = 'config_changed'

    
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
        other.assign(self)
        return other
