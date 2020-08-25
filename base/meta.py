"""A structure representing data (and metadata).
"""

# standard imports
from typing import Any
import logging

# logging
LOG = logging.getLogger(__name__)

class Metaclass(type):
    """A base class for metaclasses.

    A :py:class:`MetaClass` allows to register "meta objects" with for
    a class by invoking the class method :py:meth:`add_meta_object`.
    Instances of a that class will then use attributes of that object
    in case they do not natively have that attribute, or if the
    attribute is an unbound function, while the meta object provides a
    bound function.

    """

    def __init_subclass__(mcl, **kwargs) -> None:
        LOG.info("Metaclass: new metaclass %s", mcl.__name__)
        super().__init_subclass__(**kwargs)

    def __init__(cls, *args, **kwargs) -> None:
        cls._meta_objects = []
        super().__init__(*args, **kwargs)
        
    def add_meta_object(cls, obj, methods) -> None:
        cls._meta_objects.append((obj, methods))

    def __getattribute__(cls, attr: str) -> Any:
        """Adaptation of the attribute lookup process.

        Notes
        -----
        We adapt the attribute access in the following way: if we
        access methods of the metaclass, that would be hidden by
        methods with the same name in the actual class (e.g. if the
        actual class is an observer as well), then we want the
        bound method from the meta class (instead of the unbound method
        from the actual class).

        FIXME: the same would probably apply to attributes like
        'Change', 'Observer'
        """

        value = super().__getattribute__(attr)
        if not (callable(value) or isinstance(value, property)):
            # attribute is neither callable nor a property
            # -> no binding required
            return value

        if hasattr(value, '__self__'):
            # attribute is already a bound function - fine
            return value

        if isinstance(value, type):
            # attribute is class - fine
            return value

        if attr.startswith('__'):
            # internal method, like __init__ or __del__ - do not adapt
            return value

        for meta_object, meta_methods in cls._meta_objects:
            if attr in meta_methods:
                #print(f"Metaclass[{cls}]: Adapting meta method '{attr}' "
                #      f"({getattr(meta_object, attr)})")
                return getattr(meta_object, attr)

        # FIXME[concept]: we need some more pincipled way of determining
        # what methods to adapt ...
        #LOG.warning("Metaclass[%s]: Not adapting meta method '%s' (%s) "
        #            "[type=%s, callable=%s, is_property=%s]",
        #            cls, attr, value, type(value), callable(value),
        #            isinstance(value, property))
        return value

    def debug(cls):
        print(f"debug: {cls.__name__} with {len(cls._meta_objects)} "
              f"meta objects (id={id(cls._meta_objects)})")
        for i, (meta_object, meta_methods) in enumerate(cls._meta_objects):
            print(f"debug: ({i}) {meta_object}")
