"""base: a module providing base functionality for the deep
visualization toolbox.

This module should not depend on other modules from the toolbox.
"""

class Runner:
    """Base class for runner objects which must be provided for each
    controller/user interface.

    """

    def __init__(self) -> None:
        pass

    def runTask(self, function, *args, **kwargs) -> None:
        """Schedule the execution of a function.
        This is equivalent to running::

            fn(*args, **kwargs)

        asynchronously.

        Parameters
        ----------
        fn  :   function
                Function to run asynchronously
        args    :   list
                    Non-keyword args to ``fn``
        kwargs  :   dict
                    keyword args to ``fn``
        """
        function(*args, **kwargs)


from .observer import Observable, BusyObservable, change
from .config import Config
from .controller import Controller, View, run
