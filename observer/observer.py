"""
.. moduleauthor:: Rasmus Diederichsen

.. module:: observer

This module contains definitions for observer functionality
"""

# FIXME[design]: There are some problems with circular import.
# However: when removing the `set_controller` method from the observer
# (which not really belongs there!), we could get rid of imports here!
#import controller.base

import threading


class BaseChange(set):
    """.. :py:class:: BaseChange

    A class whose instances are passed to observers on change
    notifications, to indicate what has changed in the observable.

    Subclasses should redefine the list of valid ATTRIBUTES.
    """

    ATTRIBUTES = []

    def __init__(self, *args, **kwargs):
        """
        """
        for k in args:
            self.add(k)
        for k, v in kwargs.items():
            if v:
                self.add(k)
            else:
                self.discard(k)

    def __getattr__(self, attr):
        """Override for making dict entries accessible by dot notation.

        Parameters
        ----------
        attr    :   str
                    Name of the attribute

        Returns
        -------
        object

        Raises
        ------
        AttributeError :
            For unknown attributes.
        """
        if attr not in self.ATTRIBUTES:
            raise AttributeError(f'{self.__class__.__name__} has no '
                                 'attribute \'{attr}\'.')
        return attr in self

    def __setattr__(self, attr: str, value: bool):
        """Override for disallowing arbitrary keys in dict.

        Parameters
        ----------
        attr    :   str
                    Name of the attribute
        value   :   bool

        Raises
        ------
        ValueError  :   For unknown attributes.
        """
        if attr not in self.ATTRIBUTES:
            raise AttributeError(f'{self.__class__.__name__} has no '
                                 'attribute \'{attr}\'.')
        self.add(attr)

    @classmethod
    def all(cls):
        """Create a :py:class:`BaseChange` instance with all properties
        set to ``True``.
        """
        return cls(*cls.ATTRIBUTES)


class Observer(object):
    """Mixin for inheriting observer functionality. An observer registers
    itself to a class which notifies its observers in case something
    changes. Every observer has an associated controller object for
    dispatching changes to the observed object.

    """

    def __init__(self, **kwargs):
        """Respected kwargs:

        Parameters
        ----------
        _model: Observable
            Observable to observe.
        """
        self._model = kwargs.get('model', None)
        self._controller = None

    def observe(self, observable: 'Observable') -> None:
        """Add self to observer list of ``observable``.

        Parameters
        ----------
        observable: Observable
            The object to be observed.
        """
        observable.addObserver(self)

    def setController(self, controller: 'controller.base.BaseController'):
        """Set the controller for this observer. Will trigger observation of
        the controller's model and also cause this Observer to be
        notified with a general (change.all()) change object, allowing
        it to update itself to reflect the state of the observable.

        Parameters
        ----------
        controller: BaseController
            Controller for mediating communication with the observer object.

        """
        if self._controller is not None:
            self._controller.get_observable().remove_observer(self)
        self._controller = controller
        observable = controller.get_observable()
        self.observe(observable)
        observable.notify(self)


class Observable:
    """.. :py:class:: Observable

    Attributes
    ----------
    _observers: set
        Objects observing this class for changes
    _change_type: type
        The type of the change object sent to the observers.
        Should be a subclass of BaseChange.
    _change_method: str
        The name of the method to invoke at the observers.
    _thread_local: threading.local
        Thread local data for this observable. Can be used to
        accumulate changes within a task.
    """

    def __init__(self, change_type: type, change_method: str):
        self._observers = set()
        self._change_type = change_type
        self._change_method = change_method
        self._thread_local = threading.local()

    def changelog(self) -> (BaseChange, bool):
        data = self._thread_local
        fresh = not hasattr(data, 'change')
        if fresh:
            data.change = self._change_type()
        return data.change, fresh

    def addObserver(self, observer):
        """Add an object to observe this Observable.

        Parameters
        ----------
        observer: object
            Object which wants to be notified of changes. Must supply
            a suitable change method.
        """
        self._observers.add(observer)

    def remove_observer(self, observer: Observer):
        """Remove an observer from this Observable.

        Parameters
        ----------
        observer: object
            Object which no longer wants to be notified of changes.
        """
        self._observers.remove(observer)

    def notifyObservers(self, info: BaseChange=None):
        """Notify all observers that the state of this Observable has changed.

        Parameters
        ----------
        info : BaseChange
            Changes in the Observable since the last update.
            If ``None``, do not publish update.
        """
        if info is None and hasattr(self._thread_local, 'change'):
            info = self._thread_local.change
            del self._thread_local.change
            
        if info:
            for observer in self._observers:
                self.notify(observer, info)

    def notify(self, observer: Observer, info: BaseChange=None):
        """Notify all observers that the state of this Observable has changed.

        Parameters
        ----------
        info : BaseChange
            Changes in the Observable since the last update.
            If ``None``, do not publish update.
        """
        if info is None:
            info = self._change_type.all()
        getattr(observer, self._change_method)(self, info)
