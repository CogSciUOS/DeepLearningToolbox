"""
.. moduleauthor:: Rasmus Diederichsen, Ulf Krumnack

.. module:: util.debug

This module collects miscellaneous debugging utilities.

"""

from typing import Any
import subprocess
import traceback

from . import error


def debug(func):
    """A decorator to trace function calls
    """
    def closure(*args, **kwargs):
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

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        new_bases = filter(lambda c: c is not debug_object, cls.__bases__)
        cls.__bases__ = tuple(new_bases) + (debug_object, )

    def __init__(self, *args, **kwargs) -> None:
        # TypeError: object.__init__() takes no parameters
        if len(args):
            print(f"[type(self).__module].{type(self).__name__}.__init__: "
                  f"unexpected arguments: {args}")
            self.debug_mro()
        if len(kwargs):
            print(f"{type(self)}.__init__: "
                  f"unexpected keyword arguments: {kwargs}")
            self.debug_mro()
        super().__init__(*args, **kwargs)

    def __getattribute__(self, attr: str) -> Any:
        value = super().__getattribute__(attr)
        return (debug(value)
                if callable(value) and hasattr(self, '_debug')
                else value)

    def debug_on(self) -> None:
        self._debug = True

    def debug_off(self) -> None:
        if hasattr(self, '_debug'):
            del self._debug

    def debug_mro(self) -> None:
        print("Bases:", [cls.__module__ + '.' + cls.__name__
                         for cls in type(self).__bases__])
        print("MRO:", [cls.__module__ + '.' + cls.__name__
                       for cls in type(self).__mro__])
