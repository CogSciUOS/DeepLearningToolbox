import logging

from base import Runner
from base import Observable

def run(function):

    def wrapper(self, *args, **kwargs):
        return self.run(function, *args, **kwargs)

    return wrapper


class View:
    """A View can view one :py:class:`Observable`s. It allows to
    access (get but not set) the public (but not private) attributes
    of that observable.

    Attributes
    ----------
    <_view_attribute>: Observable
        An :py:class:`Observable` viewed by this :py:class:`View`.

    _observers: list
        A list of all :py:class:`Observer`s registered via
        :py:meth:`add_observer`. The ratio to hold this list is to
        be able in case of change of an :py:class:`Observable`,
        to transfer all Observers from the old Observable to
        to the new one.

    Class attributes
    ----------------
    _view_type: type
        The class of the objects that can be viewed by this View.
        Should be a subclass of :py:class:`Observable`
    _view_attribute: str
        The attribute name to store the reference to the observable.
    _view_type_mismatch: str
        Specifies what to do in case an Observable of inappropriate type
        is assigned. Possible values are: 'error' = raise an ValueError,
        'ignore' = ignore the assignment and continue using the old value,
        'none' = set the observable to None.
    """
    _logger = logging.getLogger(__name__)

    _view_type = Observable
    _view_attribute = '_observable'
    _view_type_mismatch = 'error'

    def __init_subclass__(cls: type,
                          view_type: type=None, view_attribute: str=None):
        """Initialization of subclasses of :py:class:`View`.
        
        Arguments
        ---------
        view_type: type
            The class of the objects that can be viewed by this View.
            Should be a subclass of :py:class:`Observable`
        view_attribute: str
            The attribute name to store the reference to the observable.
        """
        if view_type is not None:
            cls._view_type = view_type
            cls._view_attribute = '_' + view_type.__name__.lower()
        if view_attribute is not None:
            cls._view_attribute = view_attribute

    def __init__(self, observable=None, view_type_mismatch=None):
        self._observers = {}
        if view_type_mismatch:
            self._view_type_mismatch = view_type_mismatch
        self(observable)

    def __bool__(self):
        return getattr(self, type(self)._view_attribute, None) is not None

    def __call__(self, new_observable: Observable):
        """Set a new object to be viewed by this :py:class:`View`.
        """
        if new_observable and not isinstance(new_observable, self._view_type):
            if self._view_type_mismatch == "error":
                raise ValueError(f"'{type(self).__name__}' only views "
                                 f"'{self._view_type.__name__}', not "
                                 f"'{type(observable).__name__}'")
            if self._view_type_mismatch == "ignore":
                return
            new_observable = None
            
        old_observable = getattr(self, type(self)._view_attribute)
        if old_observable == new_observable:
            return  # nothing has changed ...
        
        super().__setattr__(type(self)._view_attribute, new_observable)

        # transfer all observers added via this View from tho old to
        # the new observable:
        observable = new_observable or old_observable
        self._logger.debug(f"TRANSFER ({self}) of {len(self._observers)} "
                           f"observers from old={old_observable} "
                           f"to new={new_observable}")
        for observer, interests in self._observers.items():
            self._logger.debug(f"  -> observer={observer}")
            if old_observable is not None:
                observer.unobserve(old_observable)
            if new_observable is not None:
                observer.observe(new_observable, interests)
            observable.notify(observer, observable.Change.all())
        self._logger.debug(f"TRANSFER completed.")

    def __getattr__(self, attr):
        if attr == type(self)._view_attribute:
            return None  # avoid RecursionError
        observable = getattr(self, type(self)._view_attribute, None)
        if observable is None:
            raise AttributeError(f"Cannot access attribute {attr} in "
                                 "observable as currently nothing is viewed.")
        if attr[0] == '_':
            raise AttributeError("Trying to access private attribute "
                                 f"{type(observable).__name__}.{attr}.")
        return getattr(observable, attr)

    def __setattr__(self, attr, value):
        if attr == type(self)._view_attribute:
            raise AttributeError("Setting the observable attribute "
                                 f"'{type(self)._view_attribute}'"
                                 "is forbidden.")
        super().__setattr__(attr, value)

    def isinstance(self, cls: type) -> bool:
        """Check if the observed object is an instance of a given class.

        Arguments
        ---------
        cls: type
            The type to check this object against.

        Results
        -------
        True if the viewed object is an instance of cls, else False.
        """
        return isinstance(getattr(self, self._view_attribute, None), cls)

    def _get_observable(self, o) -> Observable:
        """Get the Observable matching the given observable description.

        Result
        ------
            The matching :py:class:`Observable` or None if no matching
            Observable was found (or the matching Observable was not
            set yet or set to None).
        """
        for name, cls in self._observable_types.items():
            if self._match_observable(o, name, cls):
                return getattr(self, name)
        return None

    def add_observer(self, observer, interests=None):
        """Add a new :py:class:`Observer` to (one of) the
        :py:class:`Observervable`s controlled by this
        :py:class:`Controller`.

        Attributes
        ----------
        observer: Observer
            The observer to be added to the observable.
        """
        if interests is None:
            interests = self._view_type.Change.all()
        if observer in self._observers:
            raise RuntimeError(f"Adding observer {observer} twice to {self}.")
        self._observers[observer] = interests
        observable = getattr(self, self._view_attribute, None)
        self._logger.debug(f"ADD: {self}.add_observer({observer}) "
                           f"observable={observable}")
        if observable is not None:
            observer.observe(observable, interests)
            self._logger.debug(f"  =>{observable}.notify({observer}, "
                               f"{observable.Change.all()})")
            # Send a Change.all() notification to the new Observer, to
            # initiate a complete update
            observable.notify(observer, observable.Change.all())
        else:
            # There is no observable yet. We still want to notify the
            # Observer, so that it may reset its display.
            change = self._view_type.Change.all()
            notify = getattr(observer, self._view_type._change_method, None)
            if notify is None:
                raise NotImplementedError(f"{observer} wants to observe "
                                          f"{self} of type {self._view_type} "
                                          "but does not implement "
                                          f"'{self._view_type._change_method}'")
            else:
                notify(None, change)
        
    def remove_observer(self, observer, observable_class: type=None,
                        observable_name: str=None):
        """Remove an :py:class:`Observer` from (one of) the
        :py:class:`Observervable`s controlled by this
        :py:class:`Controller`.

        Attributes
        ----------
        observer: Observer
            The observer to be removed from the observable.
        """
        del self._observers[observer]
        observable = getattr(self, self._view_attribute, None)
        self._logger.debug(f"ADD: {self}.remove_observer({observer}) "
                           f"observable={observable}")
        if observable is not None:
            observer.unobserve(observable)
            change = self._view_type.Change.all()
            getattr(observer, self._view_type._change_method)(None, change)


class Controller(View):
    """Base class for all kinds of controllers.

    _runner: Runner
        The runner to be used when running asynchronous methods
        (i.e., the methods decorated by @run).

    """
   
    def __init__(self, runner: Runner=None, **kwargs) -> None:
        """Create a new Controller.

        Arguments
        ---------
        runner: Runner
            The runner to be used to (asynchronously) run operations.
        """
        super().__init__(**kwargs)
        self.runner = runner

    def __setattr__(self, name, value):
        observable = getattr(self, type(self)._view_attribute, None)
        if observable is not None:
            if name in observable._changeables:
                setattr(observable, name, value)
                return
            elif name in observable.__dict__:
                raise AttributeError("Trying to set uncontrolled attribute "
                                     f"{type(observable).__name__}.{name}.")
        super().__setattr__(name, value)

    @property
    def runner(self) -> Runner:
        """Get the :py:class:`Runner` associated with this Controller.
        """
        return self._runner

    @runner.setter
    def runner(self, runner: Runner) -> None:
        """Set that the :py:class:`Runner` for this Controller.

        Parameters
        ----------
        runner: Runner
            The runner to be used for asynchronous execution. If `None`,
            no asynchronous operations will be performed by this
            BaseController.
        """
        super().__setattr__('_runner', runner)
    
    def run(self, function, *args, **kwargs):
        """Run the given function. In synchronous mode, this simply
        means calling the function. in ansynchronous mode, the
        function will be started in another thread.

        There are different ways to return the results. These are
        mainly useful in asynchronous mode (but also available in
        synchronous mode).
        out_callback: callable
            call a function and provide the result(s) as arguments,
        out_store: object
            assign results to the given object (may be a tuple)
        out_notify: Change
            notify observers that the task was done.
        """
        self._logger.debug(f"{self}: Runing with runner {self._runner}"
                           f"{function}")  # "({args}, {kwargs})")
        if self._runner is None:
            return self._run(function, *args, **kwargs)
        else:
            self._runner.runTask(self._run, function, *args, **kwargs)

    def _run(self, function, *args, async_callback=None,
             out_store=None, out_notify=None, **kwargs):
        """Auxiliary function for asynchronous execution.
        """
        result = function(self, *args, **kwargs)
        if async_callback is None:
            return result
        elif isinstance(result, tuple):
            async_callback(*result)
        else:
            async_callback(result)

