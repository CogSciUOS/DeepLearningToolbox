
# Generic imports
from abc import ABCMeta
from typing import Iterator, Tuple
import importlib
import inspect
from abc import ABCMeta
import logging

# Toolbox imports
from util.error import handle_exception
from util.debug import debug_object as object
from .observer import MetaObservable
from .busy import BusyObservable, busy
from .fail import Failable


LOG = logging.getLogger(__name__)


class Registrable(object):
    """A :py:class:`Registrable` is an object that may be put into a
    register. This basically mean that it has a unique key (accessible
    via the :py:meth:`key` property).

    Parameters
    ----------
    key: str
        A (supposed to be unique) key for the new instance.

    """

    def __init__(self, *args, key: str = None, **kwargs) -> None:
        """Iniitalization of a :py:class:`Registrable`. This will
        ensure that the object has a key.
        """
        super().__init__(*args, **kwargs)
        self._key = key or self._generate_key()
        LOG.debug("Registrable: init instance of class %s with key '%s'"
                  f"{type(self).__name__} with key '{self.key}'")

    def _generate_key(self) -> str:
        """Generate a key.
        """
        # FIXME[bug]: this seems not to be a meaningful implementation.
        self._key_counter = \
            self._key_counter + 1 if hasattr(self, '_key_counter') else 0
        return self._key_counter

    @property
    def key(self):
        """Get the "public" key used to identify this item in a register.  The
        key is created upon initialization and should not be changed
        later on.

        """
        return self._key

class RegisterObservable(BusyObservable,
                         method='register_changed',
                         changes=['key_added', 'key_changed',
                                  'class_added', 'class_changed']):
    """

    **Changes**
    'key_added'
    'key_changed'
    """

    # FIXME[hack]: this could be done by notifyObservers ....
    def register_change(self, key: str, *args, **kwargs) -> None:
        changes = self.Change(*args, **kwargs)
        LOG.debug(f"register change on {self.sender}: "
                  f"{changes} with key='{key}'")

        if not changes:
            return

        # FIXME[problem]: it may happen, that the list of observers
        # changes in reaction to this notification
        # (e.g. a QRegisterController may not be interested in key changes,
        # once it can observe the real object).
        # Such a change will raise a RuntimeError:
        # "dictionary changed size during iteration".
        # Hence we make of copy of this dict:
        for observer, (notify, interests) in list(self._observers.items()):
            relevant_changes = interests & changes
            if not relevant_changes:
                continue
            try:
                notify(self.sender, observer,
                       self.Change(relevant_changes), key=key)
            except Exception as exception:
                # We will not deal with exceptions raised during not
                # notification, but instead use the default error
                # handling mechanism.
                handle_exception(exception)


class MetaRegistrable(Registrable):
    """A registrable object that generates keys from a register.x
    """

    def _generate_key(self) -> str:
        return (self.__class__.__name__ + "-" +
                str(len(self._register_key_table)))


class MetaRegisterEntry(Failable):  # FIXME[todo]: Busiable

    def __init__(self, key: str = None,
                 module_name: str = None, class_name: str = None,
                 cls=None, obj=None, args=(), kwargs={},
                 *_args, **_kwargs) -> None:

        if not (module_name and class_name) and not cls and not obj:
            raise ValueError("Provide either object or class "
                             "for MetaRegisterEntry")

        super().__init__(*_args, **_kwargs)
        self.key = key
        self.obj = obj
        self.cls = cls or (obj and type(obj))
        self.module_name = module_name or self.cls.__module__
        self.class_name = class_name or self.cls.__name__
        self.args, self.kwargs = args, kwargs

    def initialize(self, prepare: bool = False):
        if self.obj is not None:
            return  # Nothing to do
        self.obj = True
        self._prepare_class()
        message = (f"Initialization of MetaRegisterEntry '{self.key}' "
                   f"from class {self.module_name}.{self.class_name}")
        with self.failure_manager(logger=LOG, message=message):
            self.obj = self.cls(*self.args, key=self.key, **self.kwargs)
            register = type(self.obj)
            register.register_change(self.obj.key, 'key_changed')
        if self.obj is True:
            self.obj = None

    def uninitialize(self):
        obj = self.obj
        if obj is None:
            return  # nothing to do
        self.obj = None
        self.clean_failure()
        register = type(obj)
        register.register_change(obj.key, 'key_changed')
        del obj

    @property
    def initialized(self) -> bool:
        return self.obj is not None and self.obj is not True

    #@staticmethod
    def _prepare_class(self):
        # FIXME[concept]: we may want to make this part of a class register
        # and inform observers that the status of the class changed
        # if not issubclass(new_class, cls):
        #     raise TypeError(f"Class {new_class} "
        #                     f"is not a subclass of {cls}.")
        #cls._register_class_table[full_name] = new_class
        #cls.register_change(class_name, 'class_changed')
        # FIXME[old]: the docstring needs to be adapted once the concept is fixed!
        """Initialize a (sub)class of the :py:class:`MetaRegister`'s
        base class.  These are the classes that upon initialization
        will be registered at this :py:class:`MetaRegister`.
        This is basically equivalent to `from module import name`.

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

        Returns
        ------
        new_class: type
            The class object for the given class name.
        """

        if self.cls:
            return  # Nothing to do

        message = (f"Initialization of class '{self.class_name}' "
                   f"from module {self.module_name}")
        with self.failure_manager(logger=LOG, message=message):
            module = importlib.import_module(self.module_name)
            self.cls = getattr(module, self.class_name)

    @property
    def busy(self) -> bool:
        return self.obj is True


class MetaRegister(MetaObservable, Observable=RegisterObservable): #, ): # FIXME[todo]: ABCMeta:
    """A Metaclass for classes that allows to register instances.
    Instances are registered by a unique key (of type `str`).

    The :py:class:`MetaRegister` also supports delayed
    initialization.  Upon registration, just class name and
    initialization parameters are provided and initialization takes
    place only when the instanced is accessed for the first time
    (via the cls[] operator).

    The :py:class:`MetaRegister` also registers subclasses of
    the base class, allowing for iterating over subclasses and
    delayed initialization (import) of individual subclasses.

    Class attributs
    ---------------

    A class assigned to this meta class will have the following
    attributes (these are considered private properties of the class,
    subject to change, that should be used outside the class):

    _register_key_table: dict
        A dictionary containing all registered keys. The associated values
        are either the initialized object for the key, or a tuple
        of the parameters required to initialize the object:
        `(module_name, class_name, args, kwargs)`,
        with `module_name` being the name of the module defining the class
        `class_name` being the name of the class, and `args`, `kwargs`
        being the arguments to be passed to the constructor to initialize
        the object.
    
    _register_class_table: dict
        A dictionary containing all registered subclasses.
        The keys are fully qualified class names (including the module name).
        The associated value is either the instantiated class object
        or None if the class has not yet been initialized.

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

    def __new__(mcl, clsname: str, superclasses: Tuple[type],
                attributedict: dict, **class_parameters) -> None:
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
        is_base_class = not any(issubclass(supercls, MetaRegistrable)
                                for supercls in superclasses)
        if is_base_class:
            superclasses += (MetaRegistrable, )
        # class_paramters are to be processed by __init_subclass__()
        # of some superclass of the newly created class ...
        cls = super().__new__(mcl, clsname, superclasses, attributedict,
                              **class_parameters)
        if is_base_class:
            LOG.info("MetaRegister: new base class "
                     f"{cls.__module__}.{cls.__name__}")
            cls._register_base_class = cls
            cls._register_key_table = {}
            cls._register_class_table = {}
        return cls

    def __init__(cls, clsname: str, superclasses: Tuple[type],
                 attributedict: dict, **class_parameters) -> None:
        """Initialize a class declared with this :py:class:`MetaRegister`.
        This initialization will add a private class dictionary to
        hold the registered instances of the class.

        """
        # super().__init__(clsname, superclasses, **class_parameters)
        super().__init__(clsname)

        # add the newly initialized class to the class register
        cls_name = cls.__module__ + '.' + cls.__name__
        cls._register_class_table[cls_name] = cls
        cls.register_change(cls_name, 'class_added')

    def __call__(cls, *args, **kwargs) -> None:
        """A hook to adapt the initialization process for classes
        assigned to a :py:class:`MetaRegister`.
        This hook will automatically register all instances of that
        class in the register. If no register key is provided,
        a new one will be created from the class name.
        """
        LOG.debug("MetaRegister: {cls.__name}.__call__({args}, {kwargs})")
        new_entry = super().__call__(*args, **kwargs)

        if 'key' in kwargs and new_entry.key != kwargs['key']:
            print("Key mismatch for new register entry: "
                  f"should be '{kwargs['key']}' but is '{new_entry.key}'")

        # If the initialization has been invoked directly (not via
        # Register.register_initialize_key), we will register the new
        # instance now. Notice, that this instance may not be fully
        # initialized yet, as __init__() of subclasses has not been
        # finished.
        if new_entry.key not in cls._register_key_table:
            class_name = '.'.join((cls.__module__, cls.__name__))
            cls._register_key_table[new_entry.key] = \
                MetaRegisterEntry(key=new_entry.key,
                                  class_name=class_name, obj=new_entry,
                                  args=args, kwargs=kwargs)
            cls.register_change(new_entry.key, 'key_added')
        # else:
        #    cls.register_change(new_entry.key, 'key_changed')

        LOG.info("MetaRegister: new instance of class "
                 f"{type(new_entry).__name__} with key '{new_entry.key}'")
        return new_entry

    @classmethod
    def observable_name(mcl) -> str:
        return 'Register'

    def register_change(cls, key: str, info) -> None:
        cls._meta_observable.register_change(key, info)

    #
    # module related methods (FIXME[todo]: move to own module)
    #

    # A dictionary mapping modulenames MOD to a list of modulenames,
    # that are requirements for importing MOD (e.g. 'mtcnn' is a
    # requirement for importing 'tools.face.mtcnn').
    _module_requirements = {}

    @staticmethod
    def add_module_requirement(module: str, requirement: str) -> None:
        if module not in MetaRegister._module_requirements:
            MetaRegister._module_requirements[module] = []
        MetaRegister._module_requirements[module].append(requirement)

    @staticmethod
    def check_module_requirement(module: str) -> bool:
        if module not in MetaRegister._module_requirements:
            return True  # no requirements for that module
        for requirement in MetaRegister._module_requirements[module]:
            if not importlib.util.find_spec(requirement):
                return False
        return True

    #
    # class related methods
    #

    def register_class(cls, name: str, module: str = None) -> None:
        """Register a (sub)class of the :py:class:`MetaRegister`'s
        base class.  These are the classes that upon initialization
        will be registered at this :py:class:`MetaRegister`.

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
        if full_name not in cls._register_class_table:
            cls._register_class_table[full_name] = None

    def class_is_initialized(cls, name: str, module: str = None) -> bool:
        """Check if a given class is initialized.
        """
        full_name = name if module is None else ".".join((module, name))
        return cls._register_class_table.get(full_name, None) is not None

    def register_get_class(cls, name: str, module: str = None) -> type:
        full_name = name if module is None else ".".join((module, name))
        subclass = cls._register_class_table.get(full_name, None)
        if subclass is None:
            pass   # FIXME[todo]: class was already initialized
        return subclass

    # FIXME[todo]: this is not really busy: we may initialize different(!)
    # classes at thesame time ...
    # @busy("initializing class")
    def register_initialize_class(cls, name: str, module: str = None) -> type:
        """Initialize a (sub)class of the :py:class:`MetaRegister`'s
        base class.  These are the classes that upon initialization
        will be registered at this :py:class:`MetaRegister`.
        This is basically equivalent to `from module import name`.

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

        Returns
        ------
        new_class: type
            The class object for the given class name.
        """
        if module is None:
            full_name = name
            module_name, class_name = name.rsplit('.', maxsplit=1)
        else:
            full_name = ".".join((module, name))
            module_name, class_name = module, name

        new_class = cls._register_class_table.get(full_name, None)
        if new_class is not None:
            return new_class  # class was already initialized

        try:
            # FIXME[todo]: instantiation may fail - we need a concept
            # to deal with such situations.
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            LOG.error(f"Initialization of class '{full_name}' "
                      f"by {cls.__name__}: import of {module_name} failed: "
                      f"{error}")
            raise error
        except Exception as exception:
            LOG.error(f"Initialization of class '{full_name}' "
                      f"by {cls.__name__}: import of {module_name} failed: "
                      f"{exception}")
            raise exception

        new_class = getattr(module, class_name)
        if not issubclass(new_class, cls):
            raise TypeError(f"Class {new_class} is not a subclass of {cls}.")

        # FIXME[todo]: realy change class before notifying observers
        cls._register_class_table[full_name] = new_class
        cls.register_change(class_name, 'class_changed')
        return new_class

    def classes(cls, initialized: bool = None,
                abstract: bool = None) -> Iterator[str]:
        """Return an iterate over the classes registered with this
        :py:class:`MetaRegister`.

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
        for full_name, value in cls._register_class_table.items():
            is_initialized = value is not None
            if initialized is not None and initialized != is_initialized:
                continue
            if is_initialized and not abstract and inspect.isabstract(value):
                continue
            yield full_name

    #
    # key related methods
    #

    def __contains__(cls, key) -> bool:
        """Check if a given key was registered with this class.
        """
        return key in cls._register_key_table

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
                           f"{cls._register_base_class}[{cls.__name__}]."
                           f"Valid keys are: {list(cls.register_keys())}")
        return cls._register_key_table[key]

    def key_is_initializable(cls, key: str) -> bool:
        """Check if the given key can be initialized.
        Some keys may not be initializable due to unfulfilled requirements
        (like Python modules that have not been installed).
        """
        entry = cls._assert_key(key)
        return MetaRegister.check_module_requirement(entry.module_name)

    def key_is_initialized(cls, key: str) -> bool:
        """Check if a given key is initialized.
        """
        return cls._assert_key(key).initialized

    def __getitem__(cls, key: str):
        """Access the instantiated item for the given key.
        If the item was not instantiated yet, it will be
        instantiated now.
        """
        entry = cls._assert_key(key)
        if entry.initialized:
            return entry.obj

        raise KeyError(f"Key '{key}' is not yet initialized.")

    def __len__(cls):
        return len(cls._register_key_table)

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
        if key in cls._register_key_table:
            raise ValueError(f"Duplcate registration of {cls.__name__}"
                             f"with key '{key}'.")

        register_entry = MetaRegisterEntry(key=key, module_name=module_name,
                                           class_name=class_name,
                                           args=args, kwargs=kwargs)

        cls._register_key_table[key] = register_entry
        cls.register_change(key, 'key_added')

    # FIXME[todo]: this is not really busy: we may initialize different(!)
    # keys at thesame time ...
    # @busy("initializing key")
    def register_initialize_key(cls, key: str) -> object:
        """Initialize an instance of the :py:class:`MetaRegister`'s base
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

    def FIXME_OLD_register_initialize_key(cls, key: str) -> object:
        module_name, class_name, args, kwargs = item
        new_cls = cls.register_initialize_class(class_name, module_name)

        try:
            # FIXME[todo]: instantiation may fail - we need a concept
            # to deal with such situations.
            LOG.info(f"Initializing register key '{key}': "
                     f"{new_cls}({args}, {kwargs}), "
                     f"register[{key}]={cls._register_key_table[key]}")
            new_instance = new_cls(*args, key=key, **kwargs)
            LOG.info(f"Initialization of key '{key}' succeeded.")
        except Exception as exception:
            LOG.error(f"Initialization of {new_cls.__name__} to obtain "
                      f"'{key}' of class {cls.__name__} failed: {exception}")
            raise exception

        if not isinstance(new_instance, cls):
            LOG.error(f"New object for key '{key}' has wrong type "
                      f"{type(new_instance)} but should be "
                      f"a subclass of {cls}.")
            raise TypeError(f"The object instantiated from key '{key}' "
                            f"of type {type(new_instance)} "
                            f"is not an instance of {cls}.")
        # FIXME[bug]: there seems to be a problem with double
        # instantiation -
        cls._register_key_table[new_instance.key] = new_instance
        if new_instance.key != key:
            del cls._register_key_table[key]
            error = ("Inconsistent key value for new instance: "
                     f"expected '{key}', but got '{new_instance.key}'.")
            LOG.error(error)
            raise ValueError(error)
        cls.register_change(key, 'key_changed')

        return new_instance   

    def _name_for_item(cls, entry) -> str:
        return f"({entry.key})" if entry.initialized else f"[{entry.key}]"

    def name_for_key(cls, key: str) -> str:
        # FIXME[concept]: we need to develop a name concept.
        # The idea of a name would be to have an key as a unique
        # internal identifier and a name for display in user interface
        # (potentially even localizable). For now, we will
        # just use the key as name.
        entry = cls._assert_key(key)
        return cls._name_for_item(entry)

    def register_keys(cls, initialized: bool = None) -> Iterator[str]:
        """Iterate all keys registered with this
        :py:class:`MetaRegister`.
        """
        for key, entry in cls._register_key_table.items():
            if initialized is None or initialized == entry.initialized:
                yield key

    def register_instances(cls, initialized: bool = None) -> Iterator[object]:
        """Iterate all initialized objects from this
        :py:class:`MetaRegister`.
        """
        for entry in cls._register_key_table.values():
            if entry.initialized and (initialized is None or initialized):
                yield entry.obj
            elif (not entry.initialized and
                  (initialized is None or not initialized)):
                yield entry

    def items(cls, initialized: bool = None) -> Iterator[Tuple[str, str]]:
        """Iterate pairs of (key, name) for all keys registered with
        this :py:class:`MetaRegister`.
        """
        for key, entry in cls._register_key_table.items():
            if initialized is None or initialized == entry.initialized:
                yield (key, cls._name_for_item(entry))

    def debug_register(cls):
        print(f"debug: register {cls.__name__} ")
        print(f"debug: {len(cls._register_key_table)} keys:")
        for i, (key, entry) in enumerate(cls._register_key_table.items()):
            print(f"debug: ({i}) {key}: initialized={entry.initialized}")


# FIXME[todo]: read on ABCMeta and abstract classes
# https://docs.python.org/3/library/abc.html
# Is this really what we want
# Application: Datasource
class ABCMetaRegister(MetaRegister, ABCMeta):
    """A meta class for abstract register classes.
    """
    pass
