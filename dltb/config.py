"""Configuration of the Deep Learning Toolbox (dltb).

The configuration mechanism is currently not much more than a stub.

Some ideas:

* Configuration properties should have unique names.
* Configuration properties have to be defined before they can be used.
  This should avoid using unspecified values.
* Configuration properties should have some documentation that explains
  their meaning and suitable values
* Properties may also contain information where they have been defined
  and how they obtained their value.
* Default values can be specified
* Values can be defined to be constructed from other values, so that
  changing the other values also changes that property.
* Configuration values may be initialized from files.
* Configuration values may be initialized from command line arguments.
* Configuration values may be initialized from environment variables.
* Configuration values may be adapted programmatically or using a graphical
  user interface.
* Configuration properties should be grouped, potentially in a hierarchy
* For use in GUI, the Config object should be Observable (there already
  exist some old observable configuration mechanism for activation
  maximization)


Directories and paths:

* this may also be fused with the appdirs stuff used
  in `dltb/utils/__init__.py`
* home = os.environ['HOME']
* work = os.environ.get('WORK', home)
* space = os.environ.get('SPACE', home)
* self['temp'] = work
* self['models'] = os.path.join(space, 'models')
* self['data'] = os.path.join(space, 'data')
* self['opencv_models'] = \
* os.path.join(self['models'], 'opencv')
* self['tensorflow_models'] = os.path.join(self['models'], 'tensorflow')

"""
# FIXME[todo]: this implementation is still incomplete


# standard imports
from typing import Tuple, Dict, Any, Union, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
import os
import logging
import inspect

# toolbox imports
from .types import Pathlike, as_path
from .typing import get_args

# Logging
LOG = logging.getLogger(__name__)

# Xtype: either a type or a (finite) set of values.
#
# Having a type consisting of a finite collection of values seems to be
# a common situation (especially with configuration values), but I have
# not yet found a python way to declare such a type.
#
# Potential solutions do not really work out:
#
# 1. Enum:  dtype = Enum('DtypeEnum', {n: idx for n, idx in enumerate(dtype)})
#    This is a proper type, but instances are only the attributes of this
#    type: dtype.value1, dtype.value2, ...
#    There seems to be no way to directly check isinstance(value, dtype)
#
# 2. Literal:  dtype = Literal[(n for n in dtype)]
#    Literals are mainly used for type checking, there is no
#    isinstance() check - the values are still instances of their
#    original type (int, str, ...), but nut of the restricted Literal.
#
# Hence, until figuring out a better solution, we will go with our custom
# approach.
Xtype = Union[type, set]


@dataclass
class Property:
    """A property to be used by a :py:class:`Config` object.
    Provides a typed property with different default mechanisms.

    Each property has a default value, which must not be `None`. This
    default value is set at initialization and usually is not changed
    afterwards.  The property can also have an actual value
    overwriting the default value.  This value is usually set later
    and can also be overwritten.  Setting the value to `None` clear
    the value so that the default value is used again.

    The value (as well the defaul value) can be proper values or
    functions that can compute the actual value based on a
    :py:class:`Config` object.

    The `Property` has also a fixed type (`dtype`) which is set upon
    initialization and should not be changed later on.  However, there
    is no type checking performed by the `Property` class itself (as
    this may require the Config object to compute the actual type of a
    function value).  Hence the type checking should be done by the
    class that assigns and uses this `Property`.

    Arguments
    ---------

    argparse:
        A name by which this property should be added to an argument
        parser.  If `True`, the option name will be used.  If `False`,
        this property is not added to the command line parser.
    """

    dtype: Xtype
    default: Any
    _value: Any = None
    title: Optional[str] = None
    description: Optional[str] = None
    argparse: Union[bool, str] = False
    debug: Optional[Tuple[str, int]] = False

    @property
    def value(self) -> Any:
        """Get the value of this `Property`.  This is either the actual value,
        or if no value has been set, the default value is used.
        """
        return self.default if self._value is None else self._value

    @value.setter
    def value(self, value: Any) -> None:
        """Set the value for this `Property`
        """
        self._value = value


class Config:
    """A extensible configruation object providing values for
    named properties.
    """

    # mapping names to properties
    _config: Dict[str, Property] = None
    _values: Dict[str, Any] = None

    def __init__(self) -> None:
        self._config = {
            'config_debug':
            Property(dtype=bool, default=False,
                     description="Operate the Config class "
                     "in debug mode."),
            'config_assign_unknown':
            Property(dtype={'add', 'remember', 'ignore', 'error'},
                     default='remember',
                     description="How to react when assigning an unknown "
                     "configuration property: "
                     "error = raise an exeception; "
                     "add = create new property and use value as default; "
                     "remember = remember value and typecheck once the "
                     "property is registered; "
                     "ignore = ignore the assignment."),
            'config_warn_unknown':
            Property(dtype=bool, default=True,
                     description="Warn if unknown configuration values "
                     "are assigned."),
            'config_use_unknown':
            Property(dtype=bool, default=False,
                     description="Use unknown (not officially registered) "
                     "configuration values if they were remembered.")
        }
        self._values = {}
        # if self.config_assign_unknown == 'remember' else None

    def _check_type(self, value: Any, dtype: Optional[Xtype] = None) -> type:
        """Check if a given `value` has a suitable type.
        Adapt the value if automatic conversion is provided for this type.

        Arguments
        ---------
        value:
            The value whose type should be checked. This can be an actual
            value, or a function that constructs a value from this
            :py:class:`Config` object.
        dtype:
            The desired type.  If `None` no type check is performed,
            and the type of `value` is returned.

        Result
        ------
        value:
            The `value`, potentially converted to the expected type.
        dtype:
            The type of value. Same as `dtype` if provided.

        Raises
        ------
        TypeError:
            If `value` is not compatible with `dtype`.
        """
        if isinstance(value, Callable):
            value = value(self)

        if dtype is None:
            return value, type(value)

        if isinstance(dtype, set):
            if value not in dtype:
                raise TypeError(f"Value '{value}' is no member of {dtype}.")
            return value, dtype, 

        if issubclass(dtype, Path):
            if not isinstance(value, get_args(Pathlike)):
                raise TypeError(f"Value '{value}' cannot be used as Path.")
            return as_path(value), dtype

        if value is not None and not isinstance(value, dtype):
            raise TypeError(f"Value '{value}' is no instance of {dtype}.")
        return value, dtype

    def __getattr__(self, name: str) -> Any:
        if name in self._config:
            value = self._config[name].value
        elif self.config_use_unknown and name in self._values:
            value = self._values[name]
        else:
            raise AttributeError(f"Config object has no attribute '{name}'")
        return value(self) if isinstance(value, Callable) else value

    def __setattr__(self, name: str, value: Any) -> None:
        if name[0] != '_':
            if name in self._config:
                # we know this property - assign new value after typecheck
                prop = self._config[name]
                value, _dtype = self._check_type(value, prop.dtype)
                prop.value = value
            else:
                # no property with that name was registered yet -
                # we can apply different policies: error, add, remember, ignore
                if self.config_assign_unknown == 'error':
                    raise AttributeError(f"Assigning unknown property {name} "
                                         f"(value: {value})")

                if self.config_warn_unknown:
                    LOG.warning("Assigning value '%s' to unknown property "
                                "'%s' (%s).", value, name,
                                self.config_assign_unknown)
                if self.config_assign_unknown == 'add':
                    self.add_property(name, value)
                elif self.config_assign_unknown == 'remember':
                    self._values[name] = value
                # else (self.config_assign_unknown == 'ignore'): do nothing
        else:
            super().__setattr__(name, value)

    def add_property(self, name: str, default: Any,
                     dtype: Optional[Xtype] = None, **kwargs) -> None:
        """Set a default value for a given property.

        Arguments
        ---------
        name:
            The name of the new property.

        default:
            The default value for the new property.

        dtype:
            The type for values of the new property.  If provided,
            the `default` value has to match this type.  If `None`,
            the type will be automatically derived from `default`.

        kwargs:
            Further keyword arguments to be passed to the new
            :py:class:`Property` object.
        """

        if name in self._config:
            raise ValueError("Double registration of configuration property "
                             f"'name' (with default value {default}).")

        default, dtype = self._check_type(default, dtype)
        prop = Property(dtype=dtype, default=default, **kwargs)
        self._config[name] = prop

        if self.config_debug:
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe)[1]
            prop.debug = (calframe.filename, calframe.lineno)

        if name in self._values:
            # we have remembered a value that was assigned before the property
            # was defined (self.config_assign_unknown='remember')
            try:
                value = self._values.pop(name)
                setattr(self, name, value)
            except TypeError:
                LOG.warning("Previously assigned value '%s' of type %s is "
                            "incompatible with new property '%s' of type %s "
                            "and therefore was dropped.",
                            value, type(value), name, dtype)

    def set_from_string(self, line: str) -> None:
        """Set a configuration value from a string.  That string may be
        read from a file or provided at the command line.
        """
        if '=' in line:
            name, value = (s.strip() for s in line.split('='))
            setattr(self, name, value)
        elif line in self._config and self._config[line].dtype is bool:
            setattr(self, name, True)
        else:
            raise ValueError("Invalid configuration line: {line}")

    def set_from_file(self, infile: str) -> None:
        """Set configuration values from a file.  The file is expected
        to contain one configuration statement per line.
        Lines can contain comments, started with `#`.
        Empty lines are ignored.
        """
        for line in infile:
            line = line.split('#', maxsplit=1)[0].trim()
            if line:
                self.set_from_string(line)


config = Config()

#
# Global Flags
#

config.add_property('warn_missing_dependencies', default=False,
                    title="Warn on missing dependencies",
                    description="Emit warnings concerning missing "
                    "dependencies, i.e. third-party modules not installed "
                    "on this system.")

config.add_property('thirdparty_info', default=False)

config.add_property('run_slow_tests', default=False,
                    description="should slow tests be performed when running "
                    "the test suite?"
                    "Slow tests may perform file downloads, running networks, "
                    "etc. and hence significantly slow down the test time.")

config.add_property('use_cpu', default=False,
                    description="Should we use CPU "
                    "(even if GPU is available)?")

config.add_property('prepare_on_init', default=True,
                    description="prepare Preparable objects upon"
                    "initialization. This should be the normal behaviour "
                    "in single-threaded applications, but may be changed "
                    "when performing asynchronous execution.")


#
# Directories
#

config.add_property('base_directory',
                    default=Path(__file__).parents[1],
                    description="A base directory.  "
                    "Other directories can be defined based on this path.")

config.add_property('model_directory',
                    default=lambda c: c.base_directory / 'models',
                    description="A directory for storing models "
                    "(architecture and weights). Mainly used for "
                    "pretrained models, downloaded from some "
                    "third-party repository.")

config.add_property('data_directory',
                    default=lambda c: c.base_directory / 'data',
                    description="A directory for storing data. "
                    "Intended as to be used as a base directory in which "
                    "subdirectories for individual datasets can be created.")

config.add_property('work_directory',
                    default=Path(os.environ.get('WORK', '.')) / 'dltb-data',
                    description="A place where data generated by "
                    "the toolbox can be stored")

config.add_property('github_directory',
                    default=lambda c: c.base_directory / 'github',
                    description="A directory into which github repositories "
                    "can be cloned.  Cloning a repository can be a way to "
                    "uses software for which no (current) package "
                    "is released.")

config.add_property('activations_directory',
                    default=lambda c: c.work_directory / 'activations')
