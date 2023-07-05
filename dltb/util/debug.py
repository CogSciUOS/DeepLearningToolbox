"""
.. moduleauthor:: Rasmus Diederichsen, Ulf Krumnack

.. module:: util.debug

This module collects miscellaneous debugging utilities.

"""

# standard imports
from typing import Any
import gc
import os
import subprocess
import traceback
import inspect

# toolbox imports
from . import error


def debug_method(func):
    """A decorator to trace function calls
    """
    def closure_method(*args, **kwargs):
        # pylint: disable=protected-access
        try:
            self = func.__self__
            method = func.__name__
            self_text = f"{type(self).__name__}.{method}"
            indent = getattr(self, '_debug_indent', '')
            print(f"debug: {indent}{self_text}({self}, {args}, {kwargs})")
            self._debug_indent = indent + '  '
            result = func(*args, **kwargs)
            print(f"debug: {indent}{self_text}: result: {result}")
            return result
        except BaseException as exception:
            print(f"debug: {indent}{self_text}: exception: {exception}")
            raise exception
        finally:
            if indent:
                self._debug_indent = indent
            else:
                del self._debug_indent
    return closure


def debug(func):
    """A decorator to trace function calls
    """
    def closure(*args, **kwargs):
        try:
            indent = '  '
            func_name = f"{indent}{func.__name__}"
            print(f"debug: {func_name}({args}, {kwargs})")
            result = func(*args, **kwargs)
            print(f"debug: {func_name}: result: {result}")
            return result
        except BaseException as exception:
            print(f"debug: {func_name}: exception: {exception}")
            raise exception
    return closure


def debug_exception(func):
    """A decorator to output exceptions raised in a function.
    """
    def closure(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BaseException as exception:
            error.handle_exception(exception)
            raise exception
    return closure


def stacktrace() -> None:
    """Output the stack traceback at the current position.
    """
    traceback.print_stack()

def debug_stack() -> None:
    """Debug the current stack.
    """
    stack = inspect.stack()
    # each line in the stack consists of the following six fields:
    # [0] <class 'frame'>:
    # [1] <class 'str'>: file name
    # [2] <class 'int'>: line number
    # [3] <class 'str'>: the function
    # [4] <class 'list'>: the lines
    # [5] <class 'int'>: ?
    cwd = os.getcwd()
    for idx, line in enumerate(stack):
        if line[4] is None:
            continue  # for frozen modules line[4] is None
        if True or "import " in line[4][0]:
            file = line[1]
            if file.startswith(cwd):
                file = '.' + file[len(cwd):]
            if True or file.startswith('.'):
                print(f"[{idx}/{len(stack)}] {file}:{line[2]}: "
                      f"{line[4][0]}", end='')


def edit(filename: str, lineno: int = None) -> int:
    """Open a file in an (external) editor.

    Arguments
    ---------
    filename: str
        The fully qualified name of the file to open.
    lineno: int
        The line number to move the cursor to.

    Result
    ------
    retcode
        if >=0, the return code from the editor invocation call
        otherwise the negative signal number by which the editor
        was interupted.

    Raises
    ------
    OSError:
        In case of an error in the command invocation process.
    """
    command = f"metaulf-edit {filename}:{lineno}"
    return subprocess.call(command, shell=True)


class debug_object:
    """A mixin class intended to support debugging of classes in
    class hierarchies.

    Attributes
    ----------
    _debug: bool (optional)
        If present (and `True`), debugging is activated for this instance
        of the class.  Usually this attribute is not present (no
        initialization upon object construction).
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        new_bases = filter(lambda c: c is not debug_object, cls.__bases__)
        cls.__bases__ = tuple(new_bases) + (debug_object, )

    def __init__(self, *args, **kwargs) -> None:
        # TypeError: object.__init__() takes no parameters
        if args:
            print(f"ERROR:[type(self).__module].{type(self).__name__}.__init__:"
                  f" unexpected arguments: {args}")
            self.debug_mro(prefix="  ")
        if kwargs:
            print(f"{type(self)}.__init__: "
                  f"unexpected keyword arguments: {kwargs}")
            self.debug_mro(prefix="  ")
        super().__init__(*args, **kwargs)

    def __getattribute__(self, attr: str) -> Any:
        value = super().__getattribute__(attr)
        return (debug(value)
                if callable(value) and hasattr(self, '_debug')
                else value)

    def debug_on(self) -> None:
        """Turn on debugging for this object.
        """
        self._debug = True

    def debug_off(self) -> None:
        """Turn off debugging for this object.
        """
        if hasattr(self, '_debug'):
            del self._debug

    def debug_mro(self, prefix: str = '') -> None:
        """Output the method resolution order (MRO) as well as the
        direct base classes of this object's class.
        """
        print(f"{prefix}Bases:", [cls.__module__ + '.' + cls.__name__
                                  for cls in type(self).__bases__])
        print(f"{prefix}MRO:", [cls.__module__ + '.' + cls.__name__
                                for cls in type(self).__mro__])


def mro_implementations(method_name: str, target: object,
                        files: bool = False) -> None:
    """Output a list of classes implementing a given method.

    Arguments
    ---------
    method_name:
        Name of the method of interest.
    target:
        Target (object or class) of interest.
    files:
        Show filenames where method is implemented.
    """
    print(f"Implementations of method 'method_name' in the MRO of object:")
    mro = target.__mro__ if isinstance(target, type) else type(target).__mro__
    for cls in mro:
        if hasattr(cls, method_name):
            if files:
                print(f"- {cls.__module__.__file__}: {cls}.{method_name}()")
            else:
                print(f"- {cls}.{method_name}()")


def debug_references(variable):
    """Output a list of references to the given variable.
    """
    print(f"Ref count vor variable of type {type(variable)}:",
          sys.getrefcount(variable))
    for idx, referrer in enumerate(gc.get_referrers(variable), start=1):
        if isinstance(referrer, dict):
            print(f"  ({idx}) Referrer: {type(referrer)}",
                  # tuple(referrer.keys()),
                  referrer.get('__name__', "?"), referrer.get('__package__', '?'),
                  referrer.get('sys', "?"), referrer.get('builtins', '?'),
                  referrer.get('__main__', "?"))
        elif isinstance(referrer, tuple):
            print(f"  ({idx}) Referrer: {type(referrer)}",
                  len(referrer))
        else:                
            print(f"  ({idx}) Referrer: {type(referrer)}")
