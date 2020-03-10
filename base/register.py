import importlib

class RegisterMetaclass(type):

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(cls, '_item_lookup_table'):
            cls._item_lookup_table = {}

    def __getitem__(cls, key):
        cls._assert_key(key)
        
        item = cls._item_lookup_table[key]
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
        except:
            print(f"Instantiation of {new_cls.__name__} to obtain "
                  f"'{key}' of class {cls.__name__} failed: ", error)
            raise error

        #cls._item_lookup_table[key] = new_instance
        return new_instance
        
    def __contains__(cls, key):
        return key in cls._item_lookup_table

    def register(cls, key, module_name, class_name, *args, **kwargs):
        if key in cls._item_lookup_table:
            raise ValueError(f"Duplcate registration of {cls.__name__}"
                             f"with key '{key}'.")
        cls._item_lookup_table[key] = (module_name, class_name, args, kwargs)

    def _assert_key(cls, key):
        if key not in cls:
            raise KeyError(f"No key '{key}' registered "
                           f"for class {cls.__name__}. Valid keys are: "
                           f"{list(cls.keys())}")
        
    def _name_for_item(cls, item):
        return f"({key})" if isinstance(item, cls) else f"[{key}]"

    def name_for_key(cls, key):
        # FIXME[concept]: we need to develop a name concept.
        # The idea of a name would be to have an key as a unique
        # internal identifier and a name for display in user interface
        # (potentially even localizable). For now, we will
        # just use the key as name.
        cls._assert_key(key)
        return cls._name_for_item(cls._item_lookup_table[key])

    def keys(cls, initialized: bool=None):
        for key, item in cls._item_lookup_table.items():
            if initialized is None or initialized == isinstance(item, cls):
                yield key

    def items(cls, initialized: bool=None):
        """Returns keys mapped to names.
        """
        for key, item in cls._item_lookup_table.items():
            if initialized is None or initialized == isinstance(item, cls):
                yield (key, cls._name_for_item(item))
