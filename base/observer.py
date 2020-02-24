"""
.. moduleauthor:: Rasmus Diederichsen

.. module:: observer

This module contains definitions for observer functionality
"""

import threading
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def change(function):
    """A decorator that indicates methods that can change the state of
    an Observable.

    The change logic:
    * A change is only reported to the observers (by calling
      :py:meth:`notifyObservers`) after the decorated method of
      the :py:class:`Observable` returns.
    * Changes can be nested. The change is only reported at the
      outermost function finishes.
    """
    def wrapper(self, *args, **kwargs):
        self._begin_change()
        logger.debug(f"in-{self._thread_local.change_count}"
                     f"({self.__class__.__name__},{function.__name__})")
        function(self, *args, **kwargs)
        logger.debug(f"out-{self._thread_local.change_count}"
                     f"({self.__class__.__name__},{function.__name__}):"
                     f"{self._thread_local.change}")
        return self._end_change()
    return wrapper



class Observable:
    """.. :py:class:: Observable
    
    A :py:class:`Observable` object is intended to notify
    :py:class:Engine and user interfaces to be notified on changes.

    Class Attributes
    ----------------
    Change: type    
        The type of the change object sent to the observers.
        Should be a subclass of Observable.Change.
    Observer: type
        The observer type for this observable. Every observer has
        to inherit from this type.
    _change_method: str
        The name of the method to invoke at the observers.
    _changeables: dict
        The changeable attributes of this Observable. The dictionary
        maps attribute names to change to be signaled when notifying
        the observers. A value of None means that no change should
        be signaled.

    Attributes
    ----------
    _observers: dict
        A mapping of observers (objects observing this class for changes)
        to pairs (notify, interest)
    _thread_local: threading.local
        Thread local data for this observable. Can be used to
        accumulate changes within a task.
    """

    def __init_subclass__(cls: type, method: str=None, changes: list=None,
                          changeables: dict=None):
        """Initialization of subclasses of :py:class:`Observable`.
        Each of this classes will provide some extra class members
        describing the observation (Change, Observer, _change_method).
        Values to initialize these class members can be passed on
        class definition as class arguments.

        Arguments
        ---------
        method: str
            Name of the method to be called in the observable.
        changes: list of str
            List of changes that can happpen. This is used to construct
            the :py:class:`Observable.Changes` type.
        changeables: dict
            A dictionary mapping names of changeable attributes to
            the change that should be signaled when notifying the
            observers. This dictionary will be merged with the
            changeables inherited from the super class.
        """
        logger.debug(f"Initializing new Observable class: {cls}, "
                     f"method={method}, changes={changes}, "
                     f"changeables={changeables}")
        if changes is not None:
            changes = cls.Change.ATTRIBUTES + changes
            cls.Change = type(cls.__name__ + ".Change", (Observable.Change,),
                              {'ATTRIBUTES': changes})

        if method is not None:
            cls._change_method = method
            def XChanged(self, observable:cls, info:cls.Change):
                raise NotImplementedError(f"{type(self).__name__} claims "
                                          f"to be a {cls.__name__}.Observer "
                                          f"but does not implement {method}.")
            cls.Observer = type(cls.__name__ + ".Observer",
                                (Observable.Observer,), {method: XChanged})

        if changeables is not None:
            cls._changeables = {**cls._changeables, **changeables}

    class Observer(object):
        """Mixin for inheriting observer functionality. An observer registers
        itself to a class which notifies its observers in case
        something changes. Every observer has an associated controller
        object for dispatching changes to the observed object.


        FIXME[hack]: this is still a hack! It assumes that the
        @change-decorated method is run by the :py:class:`AsyncRunner`.
        """

        def __init__(self, **kwargs):
            """Respected kwargs:

            Parameters
            ----------
            model: Observable
                Observable to observe.
            """
            self._model = kwargs.get('model', None)
            self._controller = None

        def observe(self, observable: 'Observable',
                    interests: 'Change') -> None:
            """Add self to observer list of ``observable``.

            Parameters
            ----------
            observable: Observable
                The object to be observed.
            """
            observable.add_observer(self, interests)

        def unobserve(self, observable: 'Observable') -> None:
            """Remove self as an observer from the list of ``observable``.

            Parameters
            ----------
            observable: Observable
                The object to be observed.
            """
            observable.remove_observer(self)

        # FIXME[concept]: this does not really belong here, but is a
        # different concept. Also it is problematic, as an observer may
        # observe multiple observables and hence also have multiple
        # controllers.
        def setController(self, controller: 'controller.base.BaseController',
                          name: str='_controller'):
            """Set the controller for this observer. Will trigger observation of
            the controller's model and also cause this Observer to be
            notified with a general (change.all()) change object, allowing
            it to update itself to reflect the state of the observable.

            Parameters
            ----------
            controller: BaseController
                Controller for mediating communication with the
                observer object.

            """
            print(f"{type(self).__name__}.setController({self}, {controller}, {name}!")
            from pydoc import locate
            if isinstance(self, locate('qtgui.utils.QObserver')):
                print("Trying to set Controller on QObserver!")
                return  # FIXME[hack]
            if getattr(self, name, None) is not None:
                observable = getattr(self, name).get_observable()
                if observable:
                    observable.remove_observer(self)
            setattr(self, name, controller)
            observable = controller.get_observable()
            self.observe(observable, interests=observable.Change.all())
            observable.notify(self)

    class Change(set):
        """.. :py:class:: Observable.Change

        A class whose instances are passed to observers on change
        notifications, to indicate what has changed in the observable.

        Subclasses should redefine the list of valid ATTRIBUTES.
        """

        ATTRIBUTES = ['observable_changed']

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
                                     f'attribute \'{attr}\'.')
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
            """Create a :py:class:`Change` instance with all properties
            set to ``True``.
            """
            return cls(*cls.ATTRIBUTES)

    _change_method: str = 'observable_changed'
    _changeables: dict = {}

    def __init__(self):
        self._observers = dict()
        self._thread_local = threading.local()

    def __bool__(self):
        # some subclasse may add a __len__ method, which may lead to
        # evaluating the Observable als False (if len is 0). We will
        # avoid this by explictly setting the truth value.
        return True

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in type(self)._changeables:
            self.change(type(self)._changeables[name])

    def _begin_change(self):
        data = self._thread_local
        if not hasattr(data, 'change'):
            data.change = self.Change()
            data.change_count = 0
        else:
            data.change_count += 1
        return data.change

    def _end_change(self):
        """End a change. This is an auxiliary function called by the
        change decorator.

        The function checks the nesting level of the change and if the
        outermost call is reached it notifies the observers. How the
        actual notification is realized depends on the executing
        thread. If we are in the main thread, we can simply call
        :py:meth:`notifyObservers`. If we run in another thread, we
        will return the :py:class:`Observable` along with the
        :py:class:`Change` object. The idea is that this will
        be passed to the main thread by the :py:class:`AsyncRunner`.
        """
        # FIXME[concept]: that does not 
        data = self._thread_local
        if not hasattr(data, 'change'):
            raise RuntimeError("No change was startetd.")

        if data.change_count > 0:
            data.change_count -= 1
        else:
            change = data.change
            del data.change
            del data.change_count
            if threading.current_thread() is threading.main_thread():
                logger.debug(f"end_change({change}):"
                             "current_thread is main_thread")
                self.notifyObservers(change)
            else:
                logger.debug(f"end_change({change}): not in main thread")
                return self, change

    def change(self, *args, **kwargs):
        """Register a change to be sent to the observers.

        The observers will not be notified immediatly, but only after
        the current change context is ended
        (:py:meth:_end_change). The ratio behind is that one may want
        to perform multiple changes and only notify observers on the
        final result.
        """
        if not hasattr(self._thread_local, 'change'):
            self.notifyObservers(self.Change(*args, **kwargs))
        else:
            self._thread_local.change |= {a for a in args}
            self._thread_local.change |= {k for k,v in kwargs.items() if v}
        

    def add_observer(self, observer: Observer,
                    interests: Change=None, notify=None) -> None:
        """Add an object to observe this Observable.

        Parameters
        ----------
        observer: object
            Object which wants to be notified of changes. Must supply
            a suitable change method.
        interests: Change
            The changes the observer is interested in. Default is
            all changes.
        notify:
            Method to be called to notify the observer. Default is
            self.notify()
        """
        self._observers[observer] = \
            (notify if notify else type(self).notify,
             interests if interests else self.Change.all())

    def remove_observer(self, observer: Observer):
        """Remove an observer from this Observable.

        Parameters
        ----------
        observer: object
            Object which no longer wants to be notified of changes.
        """
        del self._observers[observer]

    def notifyObservers(self, *args, **kwargs):
        """Notify all observers that the state of this Observable has changed.

        Parameters
        ----------
        info : Observable.Change
            Changes in the Observable since the last update.
            If ``None``, do not publish update.
        """
        if len(args) == 1 and isinstance(args[0], Observable.Change):
            info = args[0]
        else:
            info = self.Change(*args, **kwargs)
        
        logger.debug(f"{self.__class__.__name__}.notifyObservers({info})")
        if info:
            for observer, (notify, interests) in self._observers.items():
                if interests & info:
                    notify(self, observer, info)

    def notify(self, observer: Observer, info: Change=None):
        """Notify all observers that the state of this Observable has changed.

        Parameters
        ----------
        info : Observable.Change
            Changes in the Observable since the last update.
            If ``None``, do not publish update.
        """
        if info is None:
            info = self.Change.all()
        getattr(observer, self._change_method)(self, info)


    def print_observers(self):
        """Output the observers. Intended for debugging.
        """
        print(f"{len(self._observers)} Observers:")
        for observer, (notify, interest) in self._observers.items():
            print(f"  {observer}: {notify} ({interest})") 


def busy(function):
    """A decorator that marks a methods affecting the business of an
    BusyObservable. A method decorated this way sets the busy flag of
    the object and may not be called when this flag is already set.
    The change of the busy flag will be reported to the observers.
    """
    def wrapper(self, *args, **kwargs):
        self.busy = True
        result = function(self, *args, **kwargs)
        self.busy = False
        return result
    return wrapper

def busy_message(message):
    def decorator(function):
        def wrapper(self, *args, **kwargs):
            self.busy = message
            result = function(self, *args, **kwargs)
            self.busy = False
            return result
        return wrapper
    return decorator

class BusyObservable(Observable, changes=['busy_changed']):
    """A :py:class:Config object provides configuration data.  It is an
    :py:class:Observable, allowing :py:class:Engine and user
    interfaces to be notified on changes.

    """

    def __init_subclass__(cls: type, changes: list=[], **kwargs):
        if 'busy_changed' not in changes:
            changes.append('busy_changed')
        Observable.__init_subclass__.__func__(cls, changes=changes, **kwargs)

    def __init__(self):
        super().__init__()
        self._busy = False
        
    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def busy_message(self) -> str:
        return self.__dict__.get('_busy_message', 'busy') if self._busy else ''

    @busy.setter
    def busy(self, state: bool) -> None:
        if isinstance(state, str):
            message = state
            state = True
        else:
            message = None
        if self._busy == state:
            if self._busy:
                raise RuntimeError("Object is currently busy.")
            else:
                raise RuntimeError("Trying to unemploy an object "
                                   "thatis already lazy.")
        self._busy = state
        if message is None:
            self.__dict__.pop('_busy_message', None)
        else:
            self._busy_message = message
        self.notifyObservers('busy_changed')
