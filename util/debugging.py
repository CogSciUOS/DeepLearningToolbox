"""
.. moduleauthor:: Rasmus Diederichsen, Ulf Krumnack

.. module:: util.debug

This module collects miscellaneous debugging utilities.

"""

import subprocess

from . import error

def debug(func):
    """A decorator to output exceptions raised in a function.
    """
    def closure(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BaseException as exception:
            error.handle_exception(exception)
            raise exception
    return closure

def edit(filename: str, lineno: int=None) -> int:
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
