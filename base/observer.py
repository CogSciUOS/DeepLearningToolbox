"""
.. moduleauthor:: Rasmus Diederichsen

.. module:: observer

This module contains definitions for observer functionality
"""

# standard imports
from typing import Callable, Iterator, Tuple
import threading
import logging

# toolbox imports
from util.error import handle_exception
# pylint: disable=redefined-builtin
from util.debug import debug_object as object
# pylint: enable=redefined-builtin

# logging
LOG = logging.getLogger(__name__)


def change(function):
    """A decorator that indicates methods that can change the state of
    an Observable.

    The change logic:
    * A change is only reported to the observers (by calling
      :py:meth:`notify_observers`) after the decorated method of
      the :py:class:`Observable` returns.
    * Changes can be nested. The change is only reported at the
      outermost function finishes.
    """
    def wrapper(self, *args, **kwargs):
        # pylint: disable=protected-access
        self._begin_change()
        LOG.debug("in-%s(%s,%s)", self._thread_local.change_count,
                  type(self).__name__, function.__name__)
        function(self, *args, **kwargs)
        LOG.debug("out-%s(%s,%s): %s", self._thread_local.change_count,
                  type(self).__name__, function.__name__,
                  self._thread_local.change)
        return self._end_change()

    return wrapper


# FIXME[todo]: maybe make this a enum.Flag [new in version 3.6]
#  - note: enumerations cannot be extended!
#  - https://docs.python.org/3/library/enum.html#flag
class BaseChange(set):
    """.. :py:class:: Observable.Change

    A class whose instances are passed to observers on change
    notifications, to indicate what has changed in the observable.

    Subclasses should redefine the list of valid CHANGES.
    """

    CHANGES = frozenset(('observable_changed',))

    def __init__(self, *args, **kwargs):
        """Initialize the Change object.
        """
        super().__init__()
        for key in args:
            if isinstance(key, set):
                self |= key
            else:
                self.add(key)
        for key, value in kwargs.items():
            if value:
                self.add(key)
            else:
                self.discard(key)

    def __getattr__(self, attr):
        """Override for making dict entries accessible by dot notation.

        Parameters
        ----------
        attr: str
            Name of the attribute

        Returns
        -------
        object

        Raises
        ------
        AttributeError :
            For unknown attributes.
        """
        if attr not in self.CHANGES:
            raise AttributeError(f'{type(self).__name__} has no '
                                 f'attribute \'{attr}\'.')
        return attr in self

    def __setattr__(self, attr: str, value: bool):
        """Override for disallowing arbitrary keys in dict.

        Parameters
        ----------
        attr: str
            Name of the attribute
        value: bool

        Raises
        ------
        ValueError:
            The given attribute name is not known.
        """
        if attr not in self.CHANGES:
            raise AttributeError(f'{type(self).__name__} has no '
                                 'attribute \'{attr}\'.')
        if value:
            self.add(attr)
        else:
            self.discard(attr)

    def __iadd__(self, attr: str):
        if attr not in self.CHANGES:
            raise AttributeError(f'{type(self).__name__} has no '
                                 'attribute \'{attr}\'.')
        self.add(attr)

    @classmethod
    def all(cls):
        """Create a :py:class:`Change` instance with all properties
        set to ``True``.
        """
        return cls(*cls.CHANGES)


class Observation:
    """An observation describes what changes an observer is interested
    in and how it should be informed.

    Attributes
    ----------
    interest: Change
        Changes to be reported to the observer.
    notify: Callable
        The callable to be invoked to inform the Observer.
    """

    def __init__(self, interests: BaseChange, notify: Callable) -> None:
        self.interests = interests
        self.notify = notify

    def debug(self) -> None:
        """Debug this observation.
        """
        print(f"Observation.debug: {self.interests} -> {self.notify}")


class BaseObserver:
    """Mixin for inheriting observer functionality. An observer registers
    itself to a class which notifies its observers in case
    something changes. Every observer has an associated controller
    object for dispatching changes to the observed object.

    FIXME[hack]: this is still a hack! It assumes that the
    @change-decorated method is run by the :py:class:`AsyncRunner`.
    """

    def __init__(self, *args, **kwargs):
        """Respected kwargs:

        Parameters
        ----------
        model: Observable
            Observable to observe.
        """
        super().__init__(*args, **kwargs)
        self._model = kwargs.get('model', None)
        self._controller = None

    def observe(self, observable: 'Observable',
                interests: 'Observable.Change',
                notify: Callable = None) -> None:
        """Add self to observer list of ``observable``.

        Parameters
        ----------
        observable: Observable
            The object to be observed.
        """
        observable.add_observer(self, interests, notify)

    def unobserve(self, observable: 'Observable') -> None:
        """Remove self as an observer from the list of ``observable``.

        Parameters
        ----------
        observable: Observable
            The object to be observed.
        """
        observable.remove_observer(self)


class Observable(object):
    """.. :py:class:: Observable

    A :py:class:`Observable` object is intended to notify
    :py:class:Engine and user interfaces to be notified on changes.

    Attributes
    ----------
    The following are class attributes:

    Change: type
        The type of the change object sent to the observers.
        Should be a subclass of Observable.Change.
    Observer: type
        The observer type for this observable. Every observer has
        to inherit from this type.
    _change_method: str
        The name of the method to invoke at the observers. This will
        be initialized from the `method` class parameter given in the
        class definition. It can be overwritten for an individual
        :py:class:`Observer` by providing the `method` argument with
        :py:meth:`add_observer`.
    _changeables: dict
        The changeable attributes of this Observable. The dictionary
        maps attribute names to change to be signaled when notifying
        the observers. A value of None means that no change should
        be signaled.

    Each instance has its own version of the following attributes:

    _observers: dict
        A mapping of observers (objects observing this class for changes)
        to pairs (notify, interest)
    _thread_local: threading.local
        Thread local data for this observable. Can be used to
        accumulate changes within a task.

    Change contexts
    ---------------
    It may happen that several notifcations are caused sequentially,
    that should actually be considered as a single change of the
    observable and hence should only trigger one notification.
    A typical example is the change of multiple properties of an
    observable, that would each lead to a notification of all intersted
    observers. This can be realized by change contexts:

    >>> @change
    >>> def complex_change(self) -> None:
    >>>     self.do_first_change()
    >>>     self.do_second_change()

    FIXME[todo]: This can also done with a context handler

    >>> with observable.change():
    >>>    observable.do_first_change()
    >>>    observable.do_second_change()

    This will subpress the submission of any change notifications
    until the end of the outermost change context is reached. All
    changes occurring in the context will be accumulated and in the
    end a summarized notification is done.

    Internal Notes
    --------------
    (1) We have to take care when adding/removing observers during
    notification, as we are iterating over the observers,
    which may cause a RuntimeError if that dictionary is changed
    during iteration. Options for avoiding this problem:
    (A) Copy the dictionary of observers (or at least its keys) prior
    to iteration.
    (B) Block adding/removing observers during ongoing iterations -
    this may cause delays (if processing the notification by the
    observers takes some time) and may even hang the program
    (if processing removes the observer - remark: such deadlocks
    can be prevented by reentrant locks).
    (C) Queue add/remove operations and only execute them when no
    more iteration is going on.
    Currently we are using method (C).
    """

    def __init_subclass__(cls: type, method: str = None, changes: set = None,
                          changeables: dict = None, **kwargs):
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
            the :py:class:`Observable.Changes` type, by combining it
            with the parent class changes.
        changeables: dict
            A dictionary mapping names of changeable attributes to
            the change that should be signaled when notifying the
            observers. This dictionary will be merged with the
            changeables inherited from the super class.
        """
        super().__init_subclass__(**kwargs)
        LOG.debug("Initializing new Observable class: %s, method=%s, "
                  "changes=%s, changeables=%s",
                  cls, method, changes, changeables)
        if changes is not None:
            new_changes = frozenset(cls.Change.CHANGES | changes)
            cls.Change = type(cls.__name__ + ".Change", (Observable.Change,),
                              {'CHANGES': new_changes})

        if method is not None:
            cls._change_method = method

            def method_not_implemented(self, observable: cls, info:
                                       cls.Change):
                raise NotImplementedError(f"{type(self).__name__} claims "
                                          f"to be a {cls.__name__}.Observer "
                                          f"but does not implement {method}.")

            # FIXME[todo]: maybe we can make this an abstract base class?
            cls.Observer = type(cls.__name__ + ".Observer",
                                (Observable.Observer,),
                                {method: method_not_implemented})

        if changeables is not None:
            cls._changeables = {**cls._changeables, **changeables}

    Change: type = BaseChange
    Observer: type = BaseObserver
    _change_method: str = 'observable_changed'
    _changeables: dict = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._observers = dict()
        self._observers_new = dict()
        self._observers_lock = threading.RLock()
        self._observers_busy = 0
        self._thread_local = threading.local()

    def __del__(self):
        """Destructor.
        """
        # Make sure that all references to this observable are removed
        while self._observers:
            observer = next(iter(self._observers))
            observer.unobserve(self)

    def __bool__(self):
        # some subclasse may add a __len__ method, which may lead to
        # evaluating the Observable als False (if len is 0). We will
        # avoid this by explictly setting the truth value.
        return True

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in type(self)._changeables:
            self.change(type(self)._changeables[name])

    #
    # Change context
    #

    # FIXME[todo]: provide a context handler

    def _begin_change(self):
        data = self._thread_local
        if not hasattr(data, 'change'):
            data.change = type(self).Change()
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
        :py:meth:`notify_observers`. If we run in another thread, we
        will return the :py:class:`Observable` along with the
        :py:class:`Change` object. The idea is that this will
        be passed to the main thread by the :py:class:`AsyncRunner`.
        """
        data = self._thread_local
        if not hasattr(data, 'change'):
            raise RuntimeError("No change was startetd.")

        if data.change_count > 0:
            data.change_count -= 1
        else:
            data_change = data.change
            del data.change
            del data.change_count
            if threading.current_thread() is threading.main_thread():
                LOG.debug("end_change(%s): current_thread is main_thread",
                          data_change)
                self.notify_observers(data_change)
            else:
                LOG.debug("end_change(%s): not in main thread", change)
                return self, data_change

    @classmethod
    def observable_name(cls) -> str:
        """The name of this observable.
        """
        return cls.__name__

    def change(self, *args, debug: bool = False, **kwargs):
        """Register a change to be sent to the observers.

        The observers will not be notified immediatly, but only after
        the current change context is ended
        (:py:meth:_end_change). The ratio behind is that one may want
        to perform multiple changes and only notify observers on the
        final result.
        """
        if not hasattr(self._thread_local, 'change'):
            self.notify_observers(type(self).Change(*args, **kwargs),
                                  debug=debug)
        else:
            self._thread_local.change |= set(args)
            self._thread_local.change |= {k for k, v in kwargs.items() if v}

    #
    # Managing observables
    #

    def add_observer(self, observer: Observer, interests: Change = None,
                     notify: Callable = None) -> None:
        """Add an object to observe this Observable.

        Parameters
        ----------
        observer: object
            Object which wants to be notified of changes. Must supply
            a suitable change method.
        interests: Change
            The changes the observer is interested in. Default is
            all changes.
        notify: Callable
            Method to be called to notify the observer. Default is
            self.notify()
        """

        if notify is None:
            notify = self.notify_method(observer)
        if interests is None:
            interests = type(self).Change.all()
        observation = Observation(interests, notify)
        self._add_observation(observer, observation)

    def remove_observer(self, observer: Observer):
        """Remove an observer from this Observable.

        Parameters
        ----------
        observer: object
            Object which no longer wants to be notified of changes.
        """
        self._add_observation(observer, None)

    def _add_observation(self, observer: Observer,
                         observation: Observation) -> None:
        """Auxialiar function to add/set/remove observation for an
        :py:class:`Observable`. This method is desgned to be thread
        save.

        Arguments
        ---------
        observer: Observer
            The :py:class:`Observer` doing the :py:class:`Observation`.
        observation: Observation
            The :py:class:`Observation`. If there is already an
            observation registered for the given `observer`, that
            old observation is replaced by `observation` one.
            If `observation` is `None`, the observer will be removed
            from the register of this observable.
        """
        with self._observers_lock:
            self._observers_new[observer] = observation
        self._update_observations()

    def _update_observations(self) -> None:
        """Auxialiar function to update the dictionary of observables
        with newly added/set/removed observations from `_observers_new`.
        This method is desgned to be thread save: it will only perform
        the update if the observation currently are not busy.
        """
        with self._observers_lock:
            if bool(self._observers_new) and not self._observers_busy:
                for observer, observation in self._observers_new.items():
                    LOG.debug("new observer[%s] for %s: %s",
                              observer, self, observation)
                    if observation is None:
                        del self._observers[observer]
                    else:
                        self._observers[observer] = observation
                self._observers_new = dict()

    def _observations(self) -> Iterator[Tuple['Observable', Observation]]:
        """Thread safe iterator of observations. It is guaranteed
        to work even if observations are added or removed
        from this :py:class:`Observable` during iteration.
        """
        with self._observers_lock:
            self._observers_busy += 1

        try:
            for observer, observation in self._observers.items():
                yield observer, observation
        finally:
            self._observers_busy -= 1
            self._update_observations()

    #
    # Notifications
    #

    @classmethod
    def notify_method(cls, observer: Observer) -> Callable:
        """Obtain the method to call for notifiation on the
        given :py:class:Observer.
        """
        return getattr(observer, cls._change_method)

    def notify_observers(self, *args, debug: bool = False,
                         **kwargs) -> None:
        """Notify all observers of this :py:class:`Observable`
        on the given changes.

        Parameters
        ----------
        changes: Observable.Change
            Changes in the Observable since the last update.
            If ``None``, do not publish update.
        debug: bool
            A flag indicating if debug messages should be emitted
            for this notification.  The ratio behind this flag is
            to control the amount of debug output, as this method
            is expected to be called frequently.
        """
        if len(args) == 1 and isinstance(args[0], Observable.Change):
            changes = args[0]
        else:
            changes = type(self).Change(*args, **kwargs)

        LOG.debug("%s.notify_observers(%s)", type(self).__name__, changes)
        if not changes:
            return

        if debug:
            LOG.debug("-- Notifying %d observers on changes: %s",
                      len(self._observers), changes)
        for index, (observer, observation) in enumerate(self._observations()):
            debug_text = f"{index+1}) {observer}: " if debug else None
            self._notify(observation, changes, debug=debug_text, **kwargs)
        if debug:
            LOG.debug("-- Notifying observers done -------------------------")

    def notify(self, observer: Observer,
               changes: Change = None, **kwargs) -> None:
        """Notify the given observer that the state of this
        :py:class:`Observable` has changed.

        Parameters
        ----------
        observer: Observer
            The observer to notify.
        changes: Observable.Change
            Changes in the Observable since the last update.
            If ``None``, do not publish update.
        """
        try:
            observation = self._observers.get(observer)
        except KeyError:
            raise ValueError(f"Observer {observer} is not registered "
                             "for this observable")

        if observation is None:
            raise ValueError(f"Observer {observer} has no observation")

        if changes is None:
            changes = type(self).Change.all()

        self._notify(observation, changes)

    def _notify(self, observation: Observation, changes: Change,
                debug: str = None, **kwargs) -> None:
        """Send a notification for an :py:class:`Observation`. This
        essentially invokes `observation.notify` with the observable
        as first and the relevant changes as second argument.

        Arguments
        ---------
        observation: Observation
            The observation, including the notification method and
            the interests.
        changes: Changes
            The changes to report. If these changes do not meet the
            interests of the observation, no notification will be sent.
        debug: str
            A prefix for a debug message to be emitted. A debug message
            will only be emitted if this argument is not `None`.
        """
        relevant_changes = observation.interests & changes

        if debug is not None:
            LOG.debug("%s: interests=%s, relevant_changes=%s",
                      debug, observation.interests, relevant_changes)
        if not relevant_changes:
            return  # Nothing to do

        # We will catch all exceptions here and handle them with our
        # global exception handler (usually simply logging them), as
        # caller may not be interested in what went wrong on side of
        # the observer.
        # pylint: disable=broad-except
        try:
            observation.notify(self, changes, **kwargs)
        except Exception as exception:
            # We will not deal with exceptions raised
            # during not notification, but instead use
            # the default error handling mechanism.
            print(f"Notifying {changes} with method "
                  f"{observation.notify} failed.")
            LOG.error("Notifying %s with method %s failed.",
                      changes, observation.notify)
            handle_exception(exception)

    #
    # Debugging
    #

    def debug(self) -> None:
        """Output the observers. Intended for debugging.
        """
        print(f"debug: Observable[{type(self).__name__}/{self}] with "
              f"{len(self._observers)} Observers:")
        for i, (observer, observation) in enumerate(self._observers.items()):
            print(f"debug: ({i}) {observer}: "
                  f"{observation.notify} with ({observation.interests})")


