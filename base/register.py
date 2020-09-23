"""A :py:class:`Register` is basically an :py:class:`Observable` container,
that notifies its :py:class:`Observer`s when entries are registered or
unregistered.

"""
# Generic imports
from abc import ABCMeta, abstractmethod
from typing import Iterator, Tuple, Union
import importlib
import inspect
import logging

# Toolbox imports
from dltb.base.state import Stateful
from util.debug import debug_object as object
from .observer import Observable
from .prepare import Preparable
from .fail import Failable

# Logging
LOG = logging.getLogger(__name__)


class Registrable(object, metaclass=ABCMeta):
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


class RegisterEntry(Registrable):
    """A :py:class:`RegisterEntry` is a base class for
    :py:class:`Registrable` objects. It realizes the
    :py:attr:`key` property by storing the key value as
    private property.

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

    def __contains__(self, entry: Union[str, Registrable]) -> bool:
        """Check if the given entry is registered.

        Argument
        --------
        entry: Union[str, Registrable]
            Either the entry or the key.
        """
        if isinstance(entry, Registrable):
            entry = entry.key
        return entry in self._entries

    def __getitem__(self, key: str) -> Registrable:
        """Lookup the entry for the given key in this :py:class:`Register`.
        """
        return self._entries[key]

    def __len__(self) -> int:
        """The number of entries in this register.
        """
        return len(self._entries)

    def __iter__(self) -> Iterator[Registrable]:
        """Enumerate the entries registered in this :py:class:`Register`.
        """
        for entry in self._entries.values():
            yield entry

    def keys(self) -> Iterator[str]:
        """Enumerate the keys of the entries registered
        in this :py:class:`Register`.
        """
        for entry in self._entries.values():
            yield entry.key

    def register_change(self, key: str, *args, **kwargs) -> None:
        """Notify observers on a register change.
        """
        changes = self.Change(*args, **kwargs)
        LOG.debug("register change on %s: %s with key='%s'",
                  self, changes, key)

        if not changes:
            return

        self.notify_observers(changes, key=key)



class ClassRegisterEntry(Failable, Stateful):  # FIXME[todo]: Busy
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
        if not (module_name and class_name) and cls is None:
            raise ValueError("Provide either module and class name or class "
                             "for ClassRegisterEntry")

        if cls is not None:
            module_name, class_name = cls.__module__, cls.__name__
        elif class_name is not None and '.' in class_name:
            module_name, class_name = class_name.rsplit('.', maxsplit=1)

        self.cls = cls
        self.module_name = module_name
        self.class_name = class_name
        super().__init__(**kwargs)

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
        return (self.initialized or
                RegisterClass.check_module_requirement(self.module_name))

    @property
    def initialized(self) -> bool:
        """Check if the class represented by this Entry has been
        initialized.
        """
        return self.cls is not None

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


class ClassRegister(Register):
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

class InstanceRegister(Register):
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


class InstanceRegisterEntry(ClassRegisterEntry, Stateful.Observer):
    """A :py:class:`InstanceRegisterEntry` represents information that
    can be used to create an object. This includes module and
    class name as well as initialization parameters.
    """

    def __init__(self, key: str = None, obj: object = None, cls: type = None,
                 args=(), kwargs=None, **_kwargs) -> None:
        # pylint: disable=too-many-arguments
        # too-many-arguments:
        #    It is fine to give all these arguments here, as they
        #    are used to initialize one record of (initialization)
        #    information.

        if obj is not None:
            cls = type(obj)

        super().__init__(cls=cls, **_kwargs)
        if key is None:
            if obj is not None and isinstance(obj, Registrable):
                key = obj.key
            else:
                raise ValueError("No key provided for new "
                                 "InstanceRegisterEntry.")
        self._key = key
        self.obj = obj
        self.args, self.kwargs = args, (kwargs or {})

    @property
    def key(self):
        """The unique key identifying a class is the canonical (full)
        class name (including the module name).
        """
        return self._key

    @property
    def initializable(self) -> bool:
        """Check if this :py:class:`InstanceRegisterEntry` can be initialized.
        Some keys may not be initializable due to unfulfilled requirements
        (like Python modules that have not been installed).
        """
        return (self.initialized or
                RegisterClass.check_module_requirement(self.module_name))

    @property
    def initialized(self) -> bool:
        """A flog indicating if this :py:class:`InstanceRegisterEntry`
        is initialized (`True`) or not (`False`).
        """
        return self.obj is not None and self.obj is not True

    # FIXME[todo]: make busy -> no two concurrent initializations ...
    def initialize(self, prepare: bool = False) -> None:
        """Initialize this :py:class:`InstanceRegisterEntry`.
        When finished, the register observers will be informed on
        `entry_changed`, and the :py:attr:`initialized` property
        will be set to `False`.
        """
        if self.obj is not None:
            return  # Nothing to do

        # Initialize the class object
        super().initialize()

        message = (f"Initialization of InstanceRegisterEntry '{self.key}' "
                   f"from class {self.module_name}.{self.class_name}")
        with self.failure_manager(logger=LOG, message=message):
            self.obj = True  # mark this InstanceRegisterEntry as busy
            self.obj = self.cls(*self.args, key=self.key, **self.kwargs)

        register = type(self.obj).instance_register
        register.register_change(self.obj.key, 'entry_changed')

        if prepare and isinstance(self.obj, Preparable):
            self.obj.prepare()

    def uninitialize(self) -> None:
        """Unitialize this :py:class:`InstanceRegisterEntry`.
        When finished, the register observers will be informed on
        `entry_changed`, and the :py:attr:`initialized` property
        will be set to `False`.
        """
        obj = self.obj
        if obj is None:
            return  # nothing to do
        self._set_object(None)
        self.clean_failure()
        self.register.register_change(obj.key, 'entry_changed')
        del obj

    @property
    def register(self) -> InstanceRegister:
        """The instance register of the :py:class:`RegisterClass` to
        which the object this :py:class:`InstanceRegisterEntry` belongs
        to.
        """
        return None if self.cls is None else self.cls.instance_register

    @property
    def busy(self) -> bool:
        """Check if this :py:class:`InstanceRegisterEntry` is busy.
        """
        return self.obj is True

    def state_changed(self, obj: Stateful, change: Stateful.Change) -> None:
        """
        """
        self.register.register_change(obj.key, 'entry_changed')




class RegisterClassEntry(RegisterEntry):
    """A registrable object that generates keys from a register.
    """

    def _generate_key(self) -> str:
        return (self.__class__.__name__ + "-" +
                str(len(self.instance_register)))


class RegisterClass(ABCMeta):
    # pylint: disable=no-value-for-parameter
    # no-value-for-parameter:
    #     we disable this warning as there is actually a bug in pylint,
    #     not recognizing `cls` as a valid first parameter instead
    #     of `self` in metaclasses, and hence taking cls.method as
    #     an unbound call, messing up the whole argument structure.
    """A Metaclass for classes that allows to register instances.
    Instances are registered by a unique key (of type `str`).

    The :py:class:`RegisterClass` also supports delayed
    initialization.  Upon registration, just class name and
    initialization parameters are provided and initialization takes
    place only when the instanced is accessed for the first time
    (via the cls[] operator).

    The :py:class:`RegisterClass` also registers subclasses of
    the base class, allowing for iterating over subclasses and
    delayed initialization (import) of individual subclasses.

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

    Changes
    -------

    'key_added':
        A new key was added to the list of keys.
    'class_added':
        A class was added to the list of classes.
    'key_changed':
        The status of an individual key has changed (unititialized,
        initializing, initialized, failure).
    'class_changed':
        The status of an individual class has changed (unititialized,
        initializing, initialized, failure).
    """

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
        is_base_class = not any(issubclass(supercls, RegisterClassEntry)
                                for supercls in superclasses)
        if is_base_class:
            superclasses += (RegisterClassEntry, )

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

        if 'key' in kwargs and new_entry.key != kwargs['key']:
            print("Key mismatch for new register entry: "
                  f"should be '{kwargs['key']}' but is '{new_entry.key}'")

        # If the initialization has been invoked directly (not via
        # Register.register_initialize_key), we will register the new
        # instance now.
        if new_entry not in cls.instance_register:
            class_name = '.'.join((cls.__module__, cls.__name__))
            entry = InstanceRegisterEntry(key=new_entry.key,
                                          class_name=class_name, obj=new_entry,
                                          args=args, kwargs=kwargs)
            cls.instance_register.add(entry)

        LOG.info("RegisterClass: new instance of class %s with key '%s'",
                 type(new_entry).__name__, new_entry.key)
        return new_entry

    #
    # module related methods (FIXME[todo]: move to own module)
    #

    # A dictionary mapping modulenames MOD to a list of modulenames,
    # that are requirements for importing MOD (e.g. 'mtcnn' is a
    # requirement for importing 'tools.face.mtcnn').
    _module_requirements = {}

    @staticmethod
    def add_module_requirement(module: str, requirement: str) -> None:
        """Add a requirement for the given module.
        """
        if module not in RegisterClass._module_requirements:
            RegisterClass._module_requirements[module] = []
        RegisterClass._module_requirements[module].append(requirement)

    @staticmethod
    def check_module_requirement(module: str) -> bool:
        """Check if the given module requires other modules, and
        if these modules can be found.
        """
        if module not in RegisterClass._module_requirements:
            return True  # no requirements for that module
        for requirement in RegisterClass._module_requirements[module]:
            if not importlib.util.find_spec(requirement):
                return False
        return True

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
            entry = ClassRegisterEntry(module_name=module, class_name=name)
            cls.class_register.add(entry)

    # FIXME[old]: only used in toolbox/shell.py
    def classes(cls, initialized: bool = None,
                abstract: bool = None) -> Iterator[str]:
        """Return an iterate over the classes registered with this
        :py:class:`RegisterClass`.

        Arguments
        ---------
        initialized: bool
            If set, only iterate over initialized classes (True) or
            initialized not yet imported (False). A class gets initialized
            once the module defining it is imported. If this argument
            is not given, all classes will be considered.
        abstract: bool
            If set, only iterate over abstract classes (True) or
            non abstract classes (False).
        """
        for entry in cls.class_register:
            if initialized is not None and initialized != entry.initialized:
                continue
            if not abstract and (entry.initialized and
                                 inspect.isabstract(entry.cls)):
                continue
            yield entry.key

    #
    # key related methods
    #

    def register_key(cls, key: str, module_name: str, class_name: str,
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

        entry = InstanceRegisterEntry(key=key, module_name=module_name,
                                      class_name=class_name,
                                      args=args, kwargs=kwargs)

        cls.instance_register.add(entry)

    #
    # Access to instances
    #

    def __len__(cls):
        return len(cls.instance_register)

    def __contains__(cls, key) -> bool:
        """Check if a given key was registered with this class.
        """
        return key in cls.instance_register

    def __getitem__(cls, key: str):
        """Access the instantiated entry for the given key.
        If the entry was not instantiated yet, it will be
        instantiated now.
        """
        entry = cls._assert_key(key)
        if entry.initialized:
            return entry.obj

        raise KeyError(f"Key '{key}' is not yet initialized.")

    def _assert_key(cls, key: str) -> None:
        """Asserts that the given key was registered with this class.

        Raises
        ------
        KeyError:
            If a key was provided, that was not registered with
            this class.
        """
        if key not in cls:
            raise KeyError(f"No key '{key}' registered for class "
                           f"{cls.instance_register.base_class}"
                           f"[{cls.__name__}]. "
                           f"Valid keys are: {list(cls.register_keys())}")
        return cls.instance_register[key]

    #
    # FIXME[old]
    #

    # FIXME[todo]: replace by Class[key].initialized
    def key_is_initialized(cls, key: str) -> bool:
        """Check if a given key is initialized.
        """
        return cls._assert_key(key).initialized


    # FIXME[todo]: this is not really busy: we may initialize different(!)
    # keys at thesame time ...
    # @busy("initializing key")
    # FIXME[todo]: replace by Class[key, True]
    def register_initialize_key(cls, key: str) -> object:
        """Initialize an instance of the :py:class:`RegisterClass`'s base
        class (or one of its subclassse) using a registered key. This
        is basically equivalent to importing and initializing the
        class using the arguments specified when registering the key.

        Attributes
        ----------
        key: str
            A unique key identifying an instance of this class.  The
            key has to be registered using :py:meth:`register_key`
            prior to calling this method.

        Returns
        ------
        new_instance:
            The newly initialized instance of the class.
        """

        # check if that key is already initialized
        entry = cls._assert_key(key)
        if not entry.initialized:
            entry.initialize()
            # FIXME[todo]: error handling

        return entry.obj

    # FIXME[todo]: remove
    @staticmethod
    def _name_for_entry(entry) -> str:
        """Get the name for the given entry.
        """
        return f"({entry.key})" if entry.initialized else f"[{entry.key}]"

    # FIXME[todo]: rename to 'instances'
    def register_keys(cls, initialized: bool = None) -> Iterator[str]:
        """Iterate all keys registered with this
        :py:class:`RegisterClass`.
        """
        for entry in cls.instance_register:
            if initialized is None or initialized == entry.initialized:
                yield entry.key

    # FIXME[todo]: rename to 'instances'
    def register_instances(cls, initialized: bool = None) -> Iterator[object]:
        """Iterate all initialized objects from this
        :py:class:`RegisterClass`.
        """
        for entry in cls.instance_register:
            if entry.initialized and (initialized is None or initialized):
                yield entry.obj
            elif (not entry.initialized and
                  (initialized is None or not initialized)):
                yield entry

    # FIXME[todo]: rename to 'instances'
    def entries(cls, initialized: bool = None) -> Iterator[Tuple[str, str]]:
        """Iterate pairs of (key, name) for all keys registered with
        this :py:class:`RegisterClass`.
        """
        for entry in cls.instance_register:
            if initialized is None or initialized == entry.initialized:
                yield (entry.key, cls._name_for_entry(entry))

    def debug_register(cls):
        """Output debug information for this :py:class:`RegisterClass`.
        """
        print(f"debug: register {cls.__name__} ")
        print(f"debug: {len(cls.instance_register)} keys:")
        for i, entry in enumerate(cls.instance_register):
            print(f"debug: ({i}) {entry.key}: initialized={entry.initialized}")
