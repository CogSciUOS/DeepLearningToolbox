"""A :py:class:`Register` is basically an :py:class:`Observable` container,
that notifies its :py:class:`Observer`s when entries are registered or
unregistered.

"""
# Generic imports
from abc import ABCMeta, abstractmethod
from typing import Iterator, Tuple, Union
import importlib
import logging

# Toolbox imports
from .observer import Observable
from .busy import BusyObservable, busy
from .prepare import Preparable
from .fail import Failable
from ..util.debug import debug_object
from ..thirdparty import check_module_requirements


# Logging
LOG = logging.getLogger(__name__)


class Registrable(metaclass=ABCMeta):
    # pylint: disable=too-few-public-methods
    """A :py:class:`Registrable` is an object that may be put into a
    :py:class:`Register`. This basically mean that it has a unique key
    (accessible via the :py:attr:`key` property).

    Attributes
    ----------
    key: str
        A (supposed to be unique) key for the new instance.
    """
    @property
    @abstractmethod
    def key(self):
        """Get the "public" key used to identify this entry in a register.  The
        key is created upon initialization and should not be changed
        later on.

        """
        # to be implemented by subclasses


class RegisterEntry(Registrable, debug_object):
    # pylint: disable=too-few-public-methods
    """A :py:class:`RegisterEntry` is a base class for
    :py:class:`Registrable` objects. It realizes the :py:attr:`key`
    property by storing the key value as private property.

    Attributes
    ----------
    _key: str
        A (supposed to be unique) key for the new instance.

    Class Attributes
    ----------------
    _key_counter: int
        A counter for generating unique keys.

    """

    _key_counter = 0

    def __init__(self, *args, key: str = None, **kwargs) -> None:
        """Iniitalization of a :py:class:`RegisterEntry`. This will
        ensure that the object has a key.
        """
        super().__init__(*args, **kwargs)
        self._key = key or self._generate_key()
        LOG.debug("RegisterEntry: init instance of class %s with key '%s'",
                  type(self).__name__, self.key)

    def _generate_key(self) -> str:
        """Generate a key.
        """
        type(self)._key_counter += 1
        return type(self).__name__ + str(self._key_counter)

    @property
    def key(self):
        """Get the "public" key used to identify this entry in a register.  The
        key is created upon initialization and should not be changed
        later on.

        """
        return self._key


class Register(Observable, method='register_changed',
               changes={'entry_added', 'entry_changed', 'entry_removed'}):
    """

    **Changes**

    'entry_added':
        A new entry was added to this :py:class:`Register`.
    'entry_changed'
        A register entry has changed.
    'entry_removed':
        A key was remove from this :py:class:`Register`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._entries = {}

    def add(self, entry: Registrable) -> None:
        """Add a new entry to this :py:class:`Register`
        """
        key = entry.key
        self._entries[key] = entry
        self.register_change(key, 'entry_added')

    def remove(self, entry: Registrable) -> None:
        """Remove an entry from this :py:class:`Register`
        """
        key = entry.key
        del self._entries[key]
        self.register_change(key, 'entry_removed')

    #
    # The Container[Registrable] protocol
    #

    def __contains__(self, entry: Union[str, Registrable]) -> bool:
        """Check if the given entry is registered.

        Argument
        --------
        entry: Union[str, Registrable]
            Either the entry or the key.
        """
        return ((entry.key if isinstance(entry, Registrable) else entry)
                in self._entries)

    #
    # The Sized protocol
    #

    def __len__(self) -> int:
        """The number of entries in this register.
        """
        return len(self._entries)

    #
    # The Iterable interface
    #

    def __iter__(self) -> Iterator[Registrable]:
        """Iterate the entries registered in this :py:class:`Register`.
        """
        return map(lambda item: item[1], self._entries.items())

    #
    # item access
    #

    def __getitem__(self, key: str) -> Registrable:
        """Lookup the entry for the given key in this :py:class:`Register`.
        """
        return self._entries[key]

    def keys(self, **kwargs) -> Iterator[str]:
        """Ieterate the keys of the entries registered
        in this :py:class:`Register`.
        """
        return map(lambda entry: entry.key, self.entries(**kwargs))

    def entries(self) -> Iterator[str]:
        """Iterate the entries of this :py:class:`Register`.
        """
        return iter(self)

    def __iadd__(self, entry: Registrable) -> None:
        """Add an new entry or change an existing entry
        in this :py:class:`Register`
        """
        self.add(entry)

    def __delitem__(self, entry: Union[str, Registrable]) -> None:
        """Remove an entry from this :py:class:`Register`
        """
        key = entry if isinstance(entry, str) else entry.key
        del self._entries[key]
        self.register_change(key, 'entry_removed')

    def register_change(self, key: str, *args, **kwargs) -> None:
        """Notify observers on a register change.
        """
        changes = self.Change(*args, **kwargs)
        LOG.debug("register change on %s: %s with key='%s'",
                  self, changes, key)

        if not changes:
            return

        self.notify_observers(changes, key=key)


#
# The ClassRegister
#

class StatefulRegisterEntry(BusyObservable, Registrable, Failable,
                            method='entry_changed'):
    """Base class for observable register items
    """

    @property
    @abstractmethod
    def initialized(self) -> bool:
        """A flag indicatining if the entry is initalized.
        """


class StatefulRegister(Register, StatefulRegisterEntry.Observer):
    """Base class for registers with stateful items.
    """

    def add(self, entry: StatefulRegisterEntry) -> None:
        # pylint: ignore
        """Add a new entry to this :py:class:`StatefulRegister`
        """
        super().add(entry)
        # 'busy_changed', state_changed
        interests = StatefulRegisterEntry.Change('state_changed')
        self.observe(entry, interests=interests)

    def remove(self, entry: StatefulRegisterEntry) -> None:
        """Remove an entry from this :py:class:`StatefulRegister`
        """
        self.unobserve(entry)
        super().remove(entry)

    def entry_changed(self, entry: StatefulRegisterEntry,
                      _change: StatefulRegisterEntry.Change) -> None:
        """React to a change of the observed
        :py:class:`StatefulRegisterEntry`. Such a change will be
        propagated to observers of the register.
        """
        self.register_change(entry.key, 'entry_changed')

    def entries(self, initialized: bool = None) -> Iterator[Registrable]:
        # pylint: disable=arguments-differ
        """Iterate the entries of this register.
        """
        return (super().entries() if initialized is None else
                filter(lambda entry: initialized is entry.initialized,
                       super().entries()))

#
# The ClassRegister
#


class ClassRegisterEntry(StatefulRegisterEntry):
    """A :py:class:`ClassRegisterEntry` represents information that
    can be used to import a class. This includes module and class.
    """

    def __init__(self, module_name: str = None, class_name: str = None,
                 cls: type = None, **kwargs) -> None:
        """

        Arguments
        ---------
        module_name: str
            Fully qualified Module name.
        class_name: str
            Class name, either short or fully qualified
            (including module name). In the latter case, no module
            name has to be provided.
        cls: type
             The instantiated class object. If given, `module_name`
             and `class_name` will be determined automatically and
             do not have to be provided.
        """
        super().__init__(**kwargs)
        if not (module_name and class_name) and cls is None:
            raise ValueError("Provide either module and class name or class "
                             "for ClassRegisterEntry")

        if cls is not None:
            module_name, class_name = cls.__module__, cls.__name__
        elif module_name is None and '.' in class_name:
            module_name, class_name = class_name.rsplit('.', maxsplit=1)

        self.cls = cls
        self.module_name = module_name
        self.class_name = class_name

    def __str__(self) -> str:
        """String representation of this :py:class:`ClassRegisterEntry`
        """
        info = f"class {self.module_name}.{self.class_name}: "
        info += f"initialized={self.initialized}"
        if not self.initialized:
            info += f" ({'' if self.initializable else 'not '} initializable)"
        return info

    @property
    def key(self):
        """The unique key identifying a class is the canonical (full)
        class name (including the module name).
        """
        return self.module_name + '.' + self.class_name

    @property
    def initializable(self) -> bool:
        """Check if this :py:class:`ClassRegisterEntry` can be initialized.
        Some classes may not be initializable due to unfulfilled requirements
        (like Python modules that have not been installed).
        """
        return self.initialized or check_module_requirements(self.module_name)

    @property
    def initialized(self) -> bool:
        """Check if the class represented by this Entry has been
        initialized.
        """
        return self.cls is not None

    @busy("Initializing class")
    def initialize(self) -> None:
        """Initialize this class. This essentially means to load
        the module holding the class definition.
        """
        if self.cls is not None:
            return  # Nothing to do

        message = (f"Initialization of class '{self.class_name}' "
                   f"from module {self.module_name}")
        with self.failure_manager(logger=LOG, message=message):
            module = importlib.import_module(self.module_name)
            self.cls = getattr(module, self.class_name)


class ClassRegister(StatefulRegister):
    """A register for :py:class:`ClassRegisterEntry`s.
    """

    def __init__(self, base_class: type = None, **kwargs) -> None:
        if base_class is None:
            raise ValueError("No base class was provided "
                             "for the new ClassRegister.")
        super().__init__(**kwargs)
        self._base_class = base_class

    @property
    def base_class(self) -> type:
        """The base class of this :py:class:`ClassRegister`.
        All classes registered have to be subclasses of this base class.
        """
        return self._base_class

    def __getitem__(self, key: Union[str, type]) -> Registrable:
        """Lookup the entry for the given key in this :py:class:`Register`.
        """
        if isinstance(key, type):
            key = f"{key.__module__}.{key.__name__}"
        return super().__getitem__(key)

    def initialized(self, full_name: str) -> bool:
        """Check if a given class is initialized.
        """
        return full_name in self and self[full_name].initialized

    def new_class(self, cls: type) -> None:
        """Add a new class to this :py:class:`ClassRegisterEntry`.
        If there is already a :`ClassRegisterEntry` for this class,
        it will be updated, otherwise a new one will be created.
        """
        key = cls.__module__ + '.' + cls.__name__
        if key in self:
            self[key].cls = cls
            self.register_change(key, 'entry_changed')
        else:
            self.add(ClassRegisterEntry(cls=cls))


class InstanceRegister(StatefulRegister):
    """A register for :py:class:`InstanceRegisterEntry`s.
    """

    def __init__(self, base_class: type = None, **kwargs) -> None:
        if base_class is None:
            raise ValueError("No base class was provided "
                             "for the new InstanceRegister.")
        super().__init__(**kwargs)
        self._base_class = base_class

    @property
    def base_class(self) -> type:
        """The base class of this register. All instances registered
        have to be instances of this class (or one of its subclasses).
        """
        return self._base_class


class InstanceRegisterEntry(StatefulRegisterEntry, RegisterEntry):
    """A :py:class:`InstanceRegisterEntry` represents information that
    can be used to create an object. This includes module and
    class name as well as initialization parameters.
    """

    def __init__(self, key: str = None, obj: object = None,
                 cls: type = None, class_entry: ClassRegisterEntry = None,
                 args=(), kwargs=None, **_kwargs) -> None:
        # pylint: disable=too-many-arguments
        # too-many-arguments:
        #    It is fine to give all these arguments here, as they
        #    are used to initialize one record of (initialization)
        #    information.
        if key is None and obj is not None and isinstance(obj, Registrable):
            key = obj.key

        super().__init__(key=key, **_kwargs)

        if key is None:
            raise ValueError("No key provided for new InstanceRegisterEntry.")

        if obj is not None:
            if cls is None:
                cls = type(obj)
            elif cls is not type(obj):
                raise TypeError(f"Type mismatch between class {cls} "
                                f"and object of type {type(obj)}.")

        if cls is not None:
            if class_entry is None:
                class_entry = cls.class_register[cls]
            elif class_entry is not cls.class_register[cls]:
                raise TypeError("Type mismatch between class entry of "
                                f"type {class_entry.cls} and class {cls}.")

        if class_entry is None:
            raise ValueError("No class entry provided for "
                             "InstancRegisterEntry.")
        self._class_entry = class_entry
        self.obj = obj
        self.args, self.kwargs = args, (kwargs or {})

    def __str__(self) -> str:
        """String representation of this :py:class:`InstancRegisterEntry`.
        """
        return f"{self.key}: {self._class_entry}"

    @property
    def key(self) -> str:
        """The unique key identifying a class is the canonical (full)
        class name (including the module name).
        """
        return self._key

    @property
    def cls(self) -> type:
        """The unique key identifying a class is the canonical (full)
        class name (including the module name).
        """
        return self._class_entry.cls

    @property
    def initializable(self) -> bool:
        """Check if this :py:class:`InstanceRegisterEntry` can be initialized.
        Some keys may not be initializable due to unfulfilled requirements
        (like Python modules that have not been installed).
        """
        return self.initialized or self._class_entry.initializable

    @property
    def initialized(self) -> bool:
        """A flag indicating if this :py:class:`InstanceRegisterEntry`
        is initialized (`True`) or not (`False`).
        """
        return self.obj is not None and self.obj is not True

    @busy("Initializing instance")
    def initialize(self, prepare: bool = False) -> None:
        # pylint: disable=arguments-differ
        """Initialize this :py:class:`InstanceRegisterEntry`.
        When finished, the register observers will be informed on
        `entry_changed`, and the :py:attr:`initialized` property
        will be set to `False`.
        """
        if self.obj is not None:
            return  # Nothing to do

        # Initialize the class object
        self._class_entry.initialize(busy_async=False)

        message = (f"Initialization of InstanceRegisterEntry '{self.key}' "
                   f"from class {self._class_entry}")
        with self.failure_manager(logger=LOG, message=message):
            self.obj = self._class_entry.cls(*self.args, key=self.key,
                                             **self.kwargs)

        self.change('state_changed')

        if prepare and isinstance(self.obj, Preparable):
            self.obj.prepare()

    @busy("Uninitializing instance")
    def uninitialize(self) -> None:
        """Unitialize this :py:class:`InstanceRegisterEntry`.
        When finished, the register observers will be informed on
        `entry_changed`, and the :py:attr:`initialized` property
        will be set to `False`.
        """
        obj = self.obj
        if obj is None:
            return  # nothing to do
        self.obj = None
        self.clean_failure()
        self.change('state_changed')
        del obj

    @property
    def register(self) -> InstanceRegister:
        """The instance register of the :py:class:`RegisterClass` to
        which the object this :py:class:`InstanceRegisterEntry` belongs
        to.
        """
        return (None if self._entry_class.cls is None else
                self._entry_class.cls.instance_register)


class RegisterClass(ABCMeta):
    # pylint: disable=no-value-for-parameter
    # no-value-for-parameter:
    #     we disable this warning as there is actually a bug in pylint,
    #     not recognizing `cls` as a valid first parameter instead
    #     of `self` in metaclasses, and hence taking cls.method as
    #     an unbound call, messing up the whole argument structure.
    """A metaclass for classes that allow to register subclasses and
    instances.

    Class attributs
    ---------------

    A class assigned to this meta class will have the following
    attributes (these are considered private properties of the class,
    subject to change, that should be used outside the class):

    instance_register: InstanceRegister
        A InstanceRegister containing all registered keys for the class.
        The register contains :py:class:`InstanceRegisterEntry`s
        describing all registered instances, including all information
        required for instantiation. If an instances has been instantiated,
        `entry.obj` will be that instance.

    class_register: ClassRegister
        A ClassRegister containing all registered subclasses for
        the classes. The register contains :py:class:`ClassRegisterEntry`s
        describing all registered classes, including information
        for import and initialization. If a class has been created,
        `entry.cls` will be that class.

    Subclasses are automatically added to the `class_register` upon
    initialization, that is usually when the module defining the class
    is imported. Instances are also automatically added upon
    initialization. Instances are registered by a unique key (of type
    `str`).

    ** Preregistering classes and instances **

    It is also possible to preregister subclasses and instances.

    Subclasses can be preregistered by calling
    :py:meth:`register_class`. The idea of preregistering classes is
    to specify requirements that have to be fulfilled in order
    initialize (import) the class. This allows to filter out those
    subclasses with unmet requirements.  The actual instantiation
    (import) of a class can be performed in a background thread by
    calling `py:meth:`ClassRegisterEntry.initialize`.

    Preregistering instances is done by the
    :py:meth:`register_instance` method, providing the class name and
    initialization parameters. Preregistering instances has mutliple
    purposes. It allows to provide initialization arguments and
    thereby store commonly used instances for fast access. As
    registration is fast (no additional imports or data loading takes
    place during registration), it can be done in the initialization
    phase of the toolbox without significant performance effects (only
    the abstract register base classes have to be imported). The
    actual instantiation can be done in a background thread by calling
    `py:meth:`InstancRegisterEntry.initialize`.

    A registered instance can be accessed by its key using the expression
    `Class[key]`. If not instantiated yet, the instance will be
    created (synchronously).
    """

    class RegisterClassEntry(RegisterEntry):
        # pylint: disable=too-few-public-methods
        """A registrable object that generates keys from a register.
        """

        def _generate_key(self) -> str:
            # pylint: disable=no-member
            return (self.__class__.__name__ + "-" +
                    str(len(self.instance_register)))

    def __new__(mcs, clsname: str, superclasses: Tuple[type],
                attributedict: dict, **class_parameters) -> None:
        # pylint: disable=bad-mcs-classmethod-argument
        """A new class of the meta class scheme is defined.

        If this is a new base class, we add new class and key
        registers to that class.

        Parameters
        ----------
        clsname: str
            The name of the newly defined class.
        superclasses: str
            A list of superclasses of the new class.
        attributedict: dict
            The attributes (methods and class attributes ) of the
            newly defined class.
        class_parameters:
            Addition class parameters specificied in the class
            definition.
        """
        is_base_class = not any(issubclass(supercls, mcs.RegisterClassEntry)
                                for supercls in superclasses)
        if is_base_class:
            superclasses += (mcs.RegisterClassEntry, )

        # class_parameters are to be processed by __init_subclass__()
        # of some superclass of the newly created class ...
        cls = super().__new__(mcs, clsname, superclasses, attributedict,
                              **class_parameters)
        if is_base_class:
            LOG.info("RegisterClass: new base class %s.%s",
                     cls.__module__, cls.__name__)
            cls.base_class = cls
            cls.instance_register = InstanceRegister(cls)
            cls.class_register = ClassRegister(cls)
        return cls

    def __init__(cls, clsname: str, _superclasses: Tuple[type],
                 _attributedict: dict, **_class_parameters) -> None:
        """Initialize a class declared with this :py:class:`RegisterClass`.
        This initialization will add a private class dictionary to
        hold the registered instances of the class.

        """
        # super().__init__(clsname, superclasses, **class_parameters)
        super().__init__(clsname)

        # add the newly initialized class to the class register
        cls.class_register.new_class(cls)

    def __call__(cls, *args, **kwargs) -> None:
        """A hook to adapt the initialization process for classes
        assigned to a :py:class:`RegisterClass`.
        This hook will automatically register all instances of that
        class in the register. If no register key is provided,
        a new one will be created from the class name.
        """
        LOG.debug("RegisterClass: {cls.__name}.__call__({args}, {kwargs})")
        new_entry = super().__call__(*args, **kwargs)

        # Some old code invents its own keys, even if an explict key
        # argument is present.  This may lead to unexpected behaviour and
        # hence we issue a warning in this case
        if 'key' in kwargs and new_entry.key != kwargs['key']:
            LOG.warning("Key mismatch for new register entry: "
                        "should be '%s' but is '%s'",
                        kwargs['key'], new_entry.key)

        # If the initialization has been invoked directly (not via
        # Register.instance_register.initialize), we will register the
        # new instance now.
        if new_entry not in cls.instance_register:
            entry = InstanceRegisterEntry(obj=new_entry,
                                          args=args, kwargs=kwargs)
            cls.instance_register.add(entry)

        LOG.info("RegisterClass: new instance of class %s with key '%s'",
                 type(new_entry).__name__, new_entry.key)
        return new_entry

    #
    # class related methods
    #

    def register_class(cls, name: str, module: str = None) -> None:
        """Register a (sub)class of the :py:class:`RegisterClass`'s
        base class.  These are the classes that upon initialization
        will be registered at this :py:class:`RegisterClass`.

        Subclasses of the base class will be automatically registered
        once the module defining the class is imported.  However, it
        is possible to register classes in advance, allowing a user
        interface to offer the initialization (import) of that class.
        A class (registered or not) can be initialized by calling
        :py:class:`initializa_class`.

        Attributes
        ----------
        name: str
            The class name. Either the fully qualified name,
            including the module name, or just the class name.
            In the second case, the module name has to be provided
            by the argument.
        module: str
            The module name. Only required if not provided as part
            of the class name.
        """
        full_name = name if module is None else ".".join((module, name))
        if full_name not in cls.class_register:
            if module is None:
                module, name = name.rsplit('.', maxsplit=1)
            entry = ClassRegisterEntry(module_name=module, class_name=name)
            cls.class_register.add(entry)

    #
    # instance related methods
    #

    def register_instance(cls, key: str, module_name: str, class_name: str,
                          *args, **kwargs) -> None:
        """Register a key with this class. Registering a key will
        allow to initialize an instance of a class (or subclass)
        of this register.

        Arguments
        ---------
        key: str
            The unique key to be registered with this class.
        module_name: str
            The module in which the class is defined.
        class_name: str
            The name of the class.
        *args, **kwargs:
            Arguments to be passed to the constructor when initializing
            the key.
        """
        if key in cls.instance_register:
            raise ValueError(f"Duplcate registration of {cls.__name__}"
                             f"with key '{key}'.")

        full_name = module_name + '.' + class_name
        if full_name not in cls.class_register:
            cls.register_class(full_name)
        class_entry = cls.class_register[full_name]

        entry = InstanceRegisterEntry(key=key, class_entry=class_entry,
                                      args=args, kwargs=kwargs)

        cls.instance_register.add(entry)

    def __len__(cls):
        return len(cls.instance_register)

    def __contains__(cls, key) -> bool:
        """Check if a given key was registered with this class.
        """
        return key in cls.instance_register

    def __getitem__(cls, key: str) -> object:
        """Access the instantiated entry for the given key.
        If the entry was not instantiated yet, it will be
        instantiated now.
        """
        if key not in cls:
            raise KeyError(f"No key '{key}' registered for class "
                           f"{cls.instance_register.base_class}"
                           f"[{cls.__name__}]. Valid keys are: "
                           f"{list(cls.instance_register.keys())}")
        entry = cls.instance_register[key]
        if not entry.initialized:
            entry.initialize(busy_async=False)

        return entry.obj

    def keys(cls) -> Iterator[str]:
        """Iterate keys of all registered instances.
        """
        for key in cls.instance_register.keys():
            yield key

    #
    # Debugging
    #

    def debug_register(cls):
        """Output debug information for this :py:class:`RegisterClass`.
        """
        print(f"debug: register {cls.__name__} ")
        print(f"debug: {len(cls.class_register)} registered subclasses:")
        for i, entry in enumerate(cls.class_register):
            print(f"debug: ({i+1}) {entry.key}: "
                  f"initialized={entry.initialized}")
        print(f"debug: {len(cls.instance_register)} registered instances:")
        for i, entry in enumerate(cls.instance_register):
            print(f"debug: ({i+1}) {entry.key}: "
                  f"initialized={entry.initialized}")
