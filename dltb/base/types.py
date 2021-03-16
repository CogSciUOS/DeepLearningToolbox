from typing import Union, Tuple
  
class Extendable:
    """A class whose instances can be dynamically extended by additional
    superclasses.

    Extended summary
    ----------------
    The instance can be extended in two ways:
    * upon construction, by passing a `extend` parameter, providing either
      a single class or a tuple of classes
    * after initialization, by infoking the :py:meth:`extend` method,
      providing one or more superclasses.

    Examples
    --------

    1. Add a superclass during construction: network is a
    TensorFlow Classifier.
    ```
    from network.tensorflow import Network
    from network import Classifier

    network = Network(extend=Classifier)
    ```
    """

    @staticmethod
    def _extend_class(cls: type, *superclasses) -> type:
        """Create an ad hoc class as a subclass of the main class and
        the given classes. Only those classes will be added that
        are not already a superclass of the given class.

        Parameters
        ----------
        cls:
            A class that should be extended.
        superclasses:
            Classes that should be added as superclasses.

        Returns
        -------
            The new ad hoc class formed from the original class extended
            by the superclass(es).
        """
        superclasses = tuple(c for c in superclasses if not issubclass(cls, c))
        if superclasses:
            clsname = cls.__name__ + 'Extended'
            cls = cls.__class__(clsname, (cls, ) + superclasses, {})
        return cls
        
    def __new__(cls, *args, extend: Union[type, Tuple[type]]=None,
                **kwargs) -> object:
        """Construct a new instance of the given class, extended by
        the additional classes provied in the `extend` parameter given
        to the constructor.

        Parameters
        ----------
        extend:
            One or more classes by which the class of the object should
            be extended.
        """
        if extend:
            superclasses = (extend,) if isinstance(extend, type) else extend
            cls = Extendable._extend_class(cls, *superclasses)
        return super().__new__(cls)

    def __init__(self, *args, extend=None, **kwargs) -> None:
        """An initialization, processing the extend parameter.
        """
        super().__init__(*args, **kwargs)
        
    def extend(self, superclass: type) -> None:
        """Extend this :py:class:`Extendable` by adding a superclass.
        After calling this method, the :py:class:`Extendable` will
        be an instance of that superclass (in addition to its prior class).

        Parameters
        ----------
        superclass:
            The new superclass to be added. If this :py:class:`Extendable`
            is already an instance of that class, nothing will change.

        Notes
        -----
        Extending an object by a new superclass, that does not require
        initialization (does not provide its own :py:meth:`__init__`
        method) should work fine.  However, if the superclass would
        require initialization, the situation is not so easy: simply
        calling its :py:meth:`__init__` will usually also invoke the
        initialization of the new superclasses parents, which may cause
        reinitialization of parts of the :py:class:`Extendable` that
        have already been initialized and hence may lead to unpredictable
        behaviour.  Hence that :py:meth:`__init__` method will not
        be invoked automatically and it is up to the caller to perform
        necessary initialization steps.
        """
        self.__class__ = Extendable._extend_class(self.__class__, superclass)


class Identifiable:
    _id: str = None
    _counter: int = 0

    def __init__(self, id=None, **kwargs):
        super().__init__(**kwargs)
        if id is None:
            self._ensure_id()
        else:
            self._id = id

    def _ensure_id(self):
        if self._id is None:
            Identifiable._counter += 1
            self._id = self.__class__.__name__ + str(Identifiable._counter)
        return self._id

    @property
    def key(self) -> str:
        return self.get_id()

    @property
    def id(self) -> str:
        return self.get_id()
    
    def get_id(self):
        return self._ensure_id()

    def __hash__(self):
        return hash(self._ensure_id())

    def __eq__(self, other):
        if isinstance(other, Identifiable):
            return self._ensure_id() == other._ensure_id()
        return False

    def __str__(self):
        return str(self._id)
