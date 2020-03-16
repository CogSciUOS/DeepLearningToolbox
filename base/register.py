from typing import Iterator
import importlib

class RegisterMetaclass(type):
    """A Metaclass for classes that allows to register instances.
    Instances are registered by a unique key (of type `str`).

    The :py:class:`RegisterMetaclass` also supports delayed
    initialization.  Upon registration, just class name and
    initialization parameters are provided and initialization takes
    place only when the instanced is accessed for the first time
    (via the cls[] operator).

    """

    @staticmethod
    def _register_init(self, key: str=None, id: str=None, *args, **kwargs) -> None:
        """A hook to adapt the initialization process for classes
        assigned to this :py:class:`RegisterMetaclass`.
        This hook will automatically register all instances of that
        class in the register. If no register key is provided,
        a new one will be created from the class name.
        """
        if hasattr(self, '_register_init_orig'):
            self._register_init_orig(*args, **kwargs)
        else:
            super().__init__(*args, **kwargs)
        print(f"INIT INSTANCE of class {self.__class__.__name__}: {args}, {kwargs}")
        if key is None:
            key = id
        if key is None:
            key = (self.__class__.__name__ + "-" +
                   str(len(self._item_lookup_table)))
        self._id = key
        self._item_lookup_table[key] = self

    def __init__(cls, *args, **kwargs) -> None:
        """Initialize a class declared with this
        :py:class:`RegisterMetaclass`.
        This initialization will add a private class dictionary
        to hold the registered instances of the class.
        """
        super().__init__(*args, **kwargs)
        if not hasattr(cls, '_item_lookup_table'):
            print(f"NEW class {cls.__name__} has __new__: {hasattr(cls, '__new__')}; has __init__: {hasattr(cls, '__init__')}")
            cls._item_lookup_table = {}
            #cls.__new__ = RegisterMetaclass._new_instance
            if hasattr(cls, '__init__'):
                cls._register_init_orig = cls.__init__
            cls.__init__ = RegisterMetaclass._register_init

    def __contains__(cls, key) -> bool:
        """Check if a given key was registered with this class. 
        """
        return key in cls._item_lookup_table
        
    def _assert_key(cls, key: str) -> None:
        """Asserts that the given key was registered with this class.

        Raises
        ------
        KeyError:
            If a key was provided, that was not registered with
            this class.
        """
        if key not in cls:
            raise KeyError(f"No key '{key}' registered "
                           f"for class {cls.__name__}. Valid keys are: "
                           f"{list(cls.keys())}")
        return cls._item_lookup_table[key]

    def key_is_initialized(cls, key: str) -> bool:
        """Check if a given key is initialized.
        """
        return isinstance(cls._assert_key(key), cls)

    def __getitem__(cls, key: str):
        """Access the instantiated item for the given key.
        If the item was not instantiated yet, it will be
        instantiated now.
        """
        item = cls._assert_key(key)
        if isinstance(item, cls):
            return item

        try:
            # FIXME[todo]: instantiation may fail - we need a concept
            # to deal with such situations.
            module_name, class_name, args, kwargs = item
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            print(f"Preparation of '{key}' of class {cls.__name__}: "
                  f"import of {module_name} failed: ", error)
            raise error
        except BaseException as exception:
            print(f"Preparation of '{key}' of class {cls.__name__}: "
                  f"import of {module_name} failed: ", exception)
            raise exception

        new_cls = getattr(module, class_name)
        try:
            # FIXME[todo]: instantiation may fail - we need a concept
            # to deal with such situations.
            new_instance = new_cls(*args, **kwargs)
        except BaseException as exception:
            print(f"Instantiation of {new_cls.__name__} to obtain "
                  f"'{key}' of class {cls.__name__} failed: {exception}")
            raise exception

        # FIXME[bug]: there seems to be a problem with double
        # instantiation - 
        #cls._item_lookup_table[key] = new_instance
        if new_instance.id != key:
            del cls._item_lookup_table[key]
        return new_instance
        

    def register(cls, key, module_name, class_name, *args, **kwargs) -> None:
        """Register 
        """
        if key in cls._item_lookup_table:
            raise ValueError(f"Duplcate registration of {cls.__name__}"
                             f"with key '{key}'.")
        cls._item_lookup_table[key] = (module_name, class_name, args, kwargs)
        
    def _name_for_item(cls, item) -> str:
        return f"({key})" if isinstance(item, cls) else f"[{key}]"

    def name_for_key(cls, key: str) -> str:
        # FIXME[concept]: we need to develop a name concept.
        # The idea of a name would be to have an key as a unique
        # internal identifier and a name for display in user interface
        # (potentially even localizable). For now, we will
        # just use the key as name.
        cls._assert_key(key)
        return cls._name_for_item(cls._item_lookup_table[key])

    def keys(cls, initialized: bool=None) -> Iterator[str]:
        """Iterate all keys registered with this
        :py:class:`RegisterMetaclass`.
        """
        for key, item in cls._item_lookup_table.items():
            if initialized is None or initialized == isinstance(item, cls):
                yield key

    def items(cls, initialized: bool=None) -> Iterator[str]:
        """Iterate pairs of (key, name) for all keys registered with
        this :py:class:`RegisterMetaclass`.
        """
        for key, item in cls._item_lookup_table.items():
            if initialized is None or initialized == isinstance(item, cls):
                yield (key, cls._name_for_item(item))
