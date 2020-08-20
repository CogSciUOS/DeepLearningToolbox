"""
.. moduleauthor:: Rasmus Diederichsen

.. module:: observer

This module contains definitions for observer functionality
"""

from typing import Callable, Tuple, Any
import threading
import logging
logger = logging.getLogger(__name__)

from .meta import Metaclass
from util.error import handle_exception
from util.debug import debug_object as object

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
                     f"({type(self).__name__},{function.__name__})")
        function(self, *args, **kwargs)
        logger.debug(f"out-{self._thread_local.change_count}"
                     f"({type(self).__name__},{function.__name__}):"
                     f"{self._thread_local.change}")
        return self._end_change()
    return wrapper


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
        The name of the method to invoke at the observers.
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

    Notes
    -----
    We want to also allow for :py:class:`Observable` meta classes of
    :py:class:`Observable` classes. This means, that the `self` argument
    of methods can refer to an instance or a class. This makes
    referencing class attributes like `self.Change` complicated,
    as when applied to a class, the expression may be refers to
    the attribute of the class (wrong) not to that of the meta class,
    which would be the desired outcome.
    Patching the `getattr` mechanism is not an option here, as in
    other situations `Observable.Change` is really meant to provide
    the attribute of the class, while `MetaObservable.Change`
    is used for the attribute of the meta class.
    Hence we use the somewhat lengthy `type(self).Change` to access
    class attributes, which will work for instances and classes.
    """

    def __init_subclass__(cls: type, method: str = None, changes: set = None,
                          changeables: dict = None):
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
            # FIXME[old]: make sure that we are using sets (not lists) to specify the changes, then replace set(changes) -> changes
            #new_changes = frozenset(cls.Change.CHANGES | changes)
            new_changes = frozenset(cls.Change.CHANGES | set(changes))
            cls.Change = type(cls.__name__ + ".Change", (Observable.Change,),
                              {'CHANGES': new_changes})

        if method is not None:
            cls._change_method = method
            def method_not_implemented(self, observable:cls, info:cls.Change):
                raise NotImplementedError(f"{type(self).__name__} claims "
                                          f"to be a {cls.__name__}.Observer "
                                          f"but does not implement {method}.")
            # FIXME[todo]: maybe we can make this an abstract base class?
            # FIXME[concept]: maybe we want to be able to change
            # the notification method for subclasses?
            cls.Observer = type(cls.__name__ + ".Observer",
                                (Observable.Observer,),
                                {method: method_not_implemented})

        if changeables is not None:
            cls._changeables = {**cls._changeables, **changeables}

    class Observer:
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
                    interests: 'Change', notify: Callable=None) -> None:
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

        # FIXME[concept]: this does not really belong here, but is a
        # different concept. Also it is problematic, as an observer may
        # observe multiple observables and hence also have multiple
        # controllers.
        def setController(self, controller: 'controller.base.BaseController',
                          name: str = '_controller'):
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
            print(f"Observable.Observer[{type(self).__name__}].setController({self}, {controller}, {name}!")
            from pydoc import locate
            if isinstance(self, locate('qtgui.utils.QObserver')):
                print("Observable.Observer[{type(self).__name__}]: Trying to set Controller on QObserver!")
                return  # FIXME[hack]
            if getattr(self, name, None) is not None:
                observable = getattr(self, name).get_observable()
                if observable:
                    observable.remove_observer(self)
            setattr(self, name, controller)
            observable = controller.get_observable()
            self.observe(observable, interests=observable.Change.all())
            observable.notify(self)

    # FIXME[todo]: maybe make this a enum.Flag [new in version 3.6]
    #  - note: enumerations cannot be extended!
    #  - https://docs.python.org/3/library/enum.html#flag
    class Change(set):
        """.. :py:class:: Observable.Change

        A class whose instances are passed to observers on change
        notifications, to indicate what has changed in the observable.

        Subclasses should redefine the list of valid CHANGES.
        """

        CHANGES = frozenset(('observable_changed',))

        def __init__(self, *args, **kwargs):
            """Initialize the Change object.
            """
            for k in args:
                if isinstance(k, set):
                    self |= k
                else:
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
            if attr not in self.CHANGES:
                raise AttributeError(f'{type(self).__name__} has no '
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

    _change_method: str = 'observable_changed'
    _changeables: dict = {}

    def __init__(self, *args, sender=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._observers = dict()
        self._thread_local = threading.local()
        if sender is not None:
            self._sender = sender

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

    @classmethod
    def observable_name(cls) -> str:
        return cls.__name__
            
    def change(self, *args, **kwargs):
        """Register a change to be sent to the observers.

        The observers will not be notified immediatly, but only after
        the current change context is ended
        (:py:meth:_end_change). The ratio behind is that one may want
        to perform multiple changes and only notify observers on the
        final result.
        """
        if not hasattr(self._thread_local, 'change'):
            self.notifyObservers(type(self).Change(*args, **kwargs))
        else:
            self._thread_local.change |= {a for a in args}
            self._thread_local.change |= {k for k,v in kwargs.items() if v}
        

    def add_observer(self, observer: Observer,
                     interests: Change=None, notify: Callable=None) -> None:
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
        self._observers[observer] = \
            (notify if notify else type(self).notify,
             interests if interests else type(self).Change.all())

    def remove_observer(self, observer: Observer):
        """Remove an observer from this Observable.

        Parameters
        ----------
        observer: object
            Object which no longer wants to be notified of changes.
        """
        del self._observers[observer]

    @property
    def sender(self):
        return getattr(self, '_sender', self)

    # FIXME[name]: should be notify_observers
    def notifyObservers(self, *args, sender=None, **kwargs):
        """Notify all observers that the state of this Observable has changed.

        Parameters
        ----------
        changes : Observable.Change
            Changes in the Observable since the last update.
            If ``None``, do not publish update.
        """
        if len(args) == 1 and isinstance(args[0], Observable.Change):
            changes = args[0]
        else:
            changes = type(self).Change(*args, **kwargs)

        logger.debug(f"{type(self).__name__}.notifyObservers({changes})")
        if changes:
            sender = sender or self.sender

            # FIXME[bug]: RuntimeError: dictionary changed size during iteration
            # - we need some locking mechanism
            for observer, (notify, interests) in self._observers.items():
                relevant_changes = interests & changes
                #print(f"notifyObservers: interests={interests}, changes={changes}, relevant_changes={relevant_changes} [{bool(relevant_changes)}]")
                if not relevant_changes:
                    continue
                # FIXME[concept]: notify will usually be
                # Observable.notify or QObserver.QObserverHelper._qNotify
                # -> why not directly register the change_method
                #    instead of the notify callback?
                try:
                    notify(sender, observer,
                           type(self).Change(relevant_changes))
                except Exception as exception:
                    # We will not deal with exceptions raised during not
                    # notification, but instead use the default error
                    # handling mechanism.
                    handle_exception(exception)

    def notify(self, observer: Observer, info: Change=None, **kwargs) -> None:
        """Notify the given observer that the state of this
        :py:class:`Observable` has changed.

        Parameters
        ----------
        info : Observable.Change
            Changes in the Observable since the last update.
            If ``None``, do not publish update.
        """
        if info is None:
            info = type(self).Change.all()
        getattr(observer, type(self)._change_method)\
            (self.sender, info, **kwargs)

    def debug(self) -> None:
        """Output the observers. Intended for debugging.
        """
        print(f"debug: Observable[{type(self).__name__}] with "
              f"{len(self._observers)} Observers:")
        for i, (observer, (notify, interest)) in enumerate(self._observers.items()):
            print(f"debug: ({i}) {type(observer).__name__}: "
                  f"{notify} ({interest})") 


class MetaObservable(Metaclass):
    """Base class for observable meta classes.
    """

    def __init_subclass__(mcl, Observable=Observable, **kwargs) -> None:
        """Initialize subclasses of the :py:class:`MetaObservable`.
        Those subclasses will have an attribute `Observable` specifying
        the subclass of :py:class:`Observable` that should be used as
        Meta
        """
        super().__init_subclass__(**kwargs)
        mcl.Observable = Observable
        mcl.Change = Observable.Change
        mcl.Observer = Observable.Observer
        meta_methods = {'add_observer', 'remove_observer', 'change',
                        'notify', 'notifyObservers', 'debug'}
        # if issubclass(Observable, BusyObservable): FIXME[problem]: BusyObservable not known here
        meta_methods |= {'busy', 'busy_message', 'busy_start',
                         'busy_change', 'busy_stop', '_busy_run'}
        # FIXME[hack]: add_module_requirement is defined in base/register.py
        # meta_methods |= {'add_module_requirement'}
        mcl._meta_methods = frozenset(meta_methods)

    def __new__(mcl, clsname: str, superclasses: Tuple[type],
                attributedict: dict, **class_parameters) -> type:
        """Create a new :py:class:`MetaObservable`.
        """
        # FIXME[question]: this seems to do nothing - do we really need this?
        cls = super().__new__(mcl, clsname, superclasses, attributedict,
                              **class_parameters)
        return cls

    def __init__(cls, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not hasattr(cls, '_meta_observable'):
            logger.info(f"MetaObservable: class {cls} is observable")
            cls._meta_observable = type(cls).Observable(sender=cls)
            logger.debug(f"MetaObservable: {cls} is observable via "
                         f"meta observabe {cls._meta_observable}")
            cls.add_meta_object(cls._meta_observable, cls._meta_methods)

    def debug(cls):
        super().debug()
        cls._meta_observable.debug()

