""".. moduleauthor:: Ulf Krumnack

.. module:: datasource.dummy

This module provides a dummy datasource :py:class:`Dummy` that can be
used for tests.  It especially focuses on timing issues, introducing
artificial delays allowing to examine the behavior of user interfaces
in such situations.

"""

# standard imports
import time

# toolbox imports
from .noise import Noise

# The delay for operations in seconds
DELAY = 5.0

# simulate delay during import of module
time.sleep(DELAY)


class Dummy(Noise):
    # pylint: disable=too-many-ancestors
    """A :py:class:`Dummy` is a :py:class:`Datasource` that is not meant
    basically for test purposes.  It implements certain methods to just
    sleep for some seconds, allowing to investigate the behaviour in
    multi-threaded environments. This includes:
    * importing the module
    * initializing an instance of the class (:py:meth:`__init__`)
    * preparing an instance of the class (:py:meth:`prepare`)

    """

    def __init__(self, key: str = "Dummy", delay: float = DELAY,
                 **kwargs) -> None:
        """Create a new :py:class:`Dummy`
        """
        description = f"Dummy Datasource with {delay}s delay."
        super().__init__(key=key, description=description, **kwargs)
        self._delay = delay
        self._dummy_prepared = False
        time.sleep(delay)

    def __str__(self):
        return "Dummy"

    #
    # Preparation
    #

    def _prepared(self) -> bool:
        """Check if this :py:class:`Dummy` was prepared.
        """
        return super()._prepared() and self._dummy_prepared

    def _prepare(self) -> None:
        """Preparation of this :py:class:`Dummy` datasource. Preparation
        may need some time.
        """
        super()._prepare()
        time.sleep(self._delay)
        self._dummy_prepared = True

    def _unprepare(self) -> None:
        """Relase of resource acquired by the preparation of this
        :py:class:`Dummy` datasource. This should be done quickly.
        """
        self._dummy_prepared = False
        super()._unprepare()
