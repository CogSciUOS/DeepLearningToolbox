"""Framework for storing initialization values, providing a class
:py:class:`Initalization` for storing such information and
a mixin class :py:class:`Initializable` for utilizing it.

The main motivation for this framework is the realization of deferred
initialization: it allows to create a catalogs of large objects (like
networks, datasources, tools, etc.) without actually initializing
them.  So these objects can be offered to the user, but only will be
initialized once she decides to actually use them.  This should speed
up the start time of the program and reduce unnecessary use of
resources.

A second benefit of this class is the initialization register
providing a list of (uninitialzed and initialized) instances of the
class.  The class will automatically update the initialization status
of all registered :py:class:`Intialization` objects, once they get
initialized.
"""
# standard imports
from typing import Union, Optional, Tuple, Iterable
import inspect

# toolbox imports
from .meta import Constructable
from .mixin import Mixin
from .register import RegisterEntry, Register
from ..util.importer import Importer


class Initialization(RegisterEntry):
    """A container to hold values allowing to initialize a class. This
    includes the target class, positional and keyword arguments as
    well as a reference to the object (once it was initialized).

    """

    classname: str
    args: list
    kwargs: dict
    cls: Optional[type] = None
    obj: Optional[object] = None

    def __init__(self, target: Optional[Union[type, str]] = None,
                 args: Optional[list] = None, kwargs: Optional[dict] = None,
                 obj: Optional[object] = None,
                 **kwargs_) -> None:
        super().__init__(**kwargs_)
        if obj is not None:
            if isinstance(target, type) and not isinstance(obj, target):
                raise TypeError(f"objec of type {type(obj)} is not an "
                                f"instance of {target}")
            target = type(obj)
        self.obj = obj

        if isinstance(target, str):
            self.classname = target
            self.cls = None
        else:
            self.classname = \
                type(target).__module__ + '.' + type(target).__name__
            self.cls = target

        self.args = () if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

    def initialize(self) -> object:
        """Initialize the object from the initialization parameters.
        """
        if self.obj is not None:
            return self.obj  # object was already initialized

        if self.cls is None:
            _, self.cls = Importer.import_module_class(cls=self.classname)

        self.obj = self.cls(*self.args, **self.kwargs)
        return self.obj

    @property
    def initialized(self) -> bool:
        """A flag indicating if the target object is already initialized.
        """
        return self.obj is not None


class Initializable(Mixin, Constructable):
    """Mixin class for declaring a class as `Initializable`.  Such a class
    accepts an extra argument ``initialize` at construction time,
    allowing to initialize the object from :py:class:``Initialization``
    information.

    A class declared as ``Initializable`` also introduces a
    ``initialization_register``, storing :py:class:``Initialization``
    objects.  This allows to store initialization for deferred
    initialization.

    Example
    -------

    .. code-block:: python

        class MyClass(Initializable):
            ...

        # register arguments for initializing 'instance-1' (but do
        # not initialize yet). Will create an `Initialization` object.
        MyClass(initialization='instance-1', arg1=4, arg2='foo')

        # perform the actual initialization of 'instance-1', using
        # the arguments stored before.
        MyClass(initialize='instance-1')

        # register arguments for initializing 'instance-2' of another
        # class (typically a subclass of MyClass).
        MyClass(initialization=(('instance-2', 'mymodule.MySubClass'),
                                arg1=4, arg2='foo')

        # perform initialization and register the initialization
        # arguments parameters as 'instance-3'
        MyClass(initialize='instance-3', arg1=7, arg2='bar')

    Of course, registration and instantiation can be done in different
    parts of the program.

    Arguments
    ---------
    initialization:
        The name (``key``) or a tuple (``key``, ``target_class``)
        specifying an :py:class:`Initialization` object.  This object
        will be created to store the initialization arguments.
        No actual instantiation of the class will takes place. Instead
        The :py:class:`Initialization` object is returned.
    initialize:
        An :py:class:`Initialization` (or the ``key`` of a registered
        initialization) to be used to initialize the object.  If a
        ``key`` or a tuple (``key``, ``target_class``) is provided and
        no corresponding entry is present in the ``initialization_register``,
        a new entry is added to the register based on the provided arguments,
        that will then be used for initialization the new instance.

    Attributes
    ----------
    initialization_register:
        A :py:class:`Register` holding :py:class:`Intialization` objects
        for initializing instances of this class.

    """

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        if Initializable in cls.__bases__:
            cls.initialization_register = Register()

    @classmethod
    def initialization_arguments(cls) -> Iterable[inspect.Parameter]:
        """Iterate the init arguments for the given class.

        This information is obtained by traversing the class hiearchy
        and inspecting the implemented ``__init__`` method to collect
        declared arguments.

        Result:
            An ``inspect.Parameter`` object providing ``name``, ``default``
            and ``annotation``.

        """
        found = set(('self', 'args', 'kwargs'))
        for base in cls.__mro__:
            if not hasattr(base, '__init__'):
                continue  # class defines no __init__ method
            signature = inspect.signature(base.__init__)
            for arg in signature.parameters.keys():
                if arg.startswith('_'):
                    continue  # don't report private arguments
                if arg in found:
                    continue  # don't report arguments already found in subclasses
                found.add(arg)
                yield signature.parameters[arg]

    @classmethod
    def register_initialization_arguments(cls, target: type = None, **kwargs) -> Initialization:
        """Add an entry to the :py:class:`Initialization` register.

        Arguments
        ---------
        kwargs:
            Arguments to be passed to the constructor of the
            :py:class:`Initialization` class.
        """
        initialization = Initialization(target=target, **kwargs)
        cls.initialization_register.add(initialization)
        return initialization

    #
    # implementation
    #

    @classmethod
    def _constructor_hook(cls, initialization:
                          Optional[Union[Tuple[str, Union[str, type]],
                                         str]] = None, initialize:
                          Optional[Union[str, Initialization]] = None,
                          **kwargs) -> Tuple[type, Optional[dict]]:
        # pylint: disable=arguments-differ
        """Check initialization arguments for ``initialization`` and
        ``initialize`` keywords.  This function realizes the tha actual
        initiaization magic, by hooking into the initialization process
        offered by the :py:class:``Mixin`` class.

        Arguments
        ---------
        initialization, initialize:
            See class description.

        Results
        -------
        result:
            The class to be initialized or an already initialized instance
            of the class or in :py:class:`Initialization` object (if the
            ``initialization`` object was given).
        kwargs:
            The keyword arguments to be used for initialization or ``None``
            if an initialized object is returned.
        """

        if initialization is not None:
            # create an Initialization, add it toe the initialization
            # register and return it.
            if isinstance(initialization, str):  # initialization is key
                target = cls
            else:  # initialization is (key, cls)
                initialization, target = initialization

            initialization = cls.register_initialization_arguments(
                key=initialization, target=target, kwargs=kwargs)
            return initialization, None

        if initialize is not None:
            if isinstance(initialize, str):
                target = cls
            else: # initialize is (key, cls)
                initialize, target = initialize

            if initialize not in cls.initialization_register:
                initialization = cls.register_initialization_arguments(
                    key=initialize, target=target, kwargs=kwargs)
            elif not kwargs:
                initialization = cls.initialization_register[initialize]
            else:
                raise ValueError(f"Initialization '{initialize}' already "
                                 "registered - cannot initialize with "
                                 "additional arguments.")
            return initialization.initialize(), None

        return super()._constructor_hook(**kwargs)
