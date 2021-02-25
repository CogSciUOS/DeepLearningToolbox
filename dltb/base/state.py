"""Base class for classes that have an internal state.
"""

from .observer import Observable


class Stateful(Observable, method='state_changed', changes={'state_changed'}):
    """Abstract base class for objects that can have different states.
    Each stateful object can be in a `ready` state, meaning that it
    can be used.
    """

    @property
    def ready(self) -> bool:
        """The object is ready and can be used.
        """
        return True
