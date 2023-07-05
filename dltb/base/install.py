"""Abstract base class for resources.
"""
# FIXME[old/todo]: there are at least 3 different Installable classes
#  -> combine them into someting useful

# standard imports
from abc import ABC, abstractmethod
import os
import sys
import time
import logging

# toolbox imports
from .busy import BusyObservable
from .fail import Failable
from ..util.importer import importable, import_module

# logging
LOG = logging.getLogger(__name__)


# FIXME[todo]: merge with other Resource classes
# (in util.resources ans thirdparty)...
class Installable(BusyObservable, Failable):
    """An installable is some resource may have some requirements to be used.
    It may provide methods to install such requirements.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._requirements = {}

    #
    # Requirements
    #

    def _add_requirement(self, name, what, *data) -> None:
        """Add a requirement for this :py:class:`Tool`.
        """
        self._requirements[name] = (what,) + data

    #
    # Preparable
    #

    def _preparable(self) -> bool:
        """Check if required resources are available.
        """
        for name, requirement in self._requirements.items():
            if requirement[0] == 'file':
                if not os.path.exists(requirement[1]):
                    LOG.warning("File requirement '%s' (filename='%s') "
                                "for resource '%s' (%s) not found.", name,
                                requirement[1], self.key, type(self).__name__)
                    return False
            if requirement[0] == 'module':
                if requirement[1] in sys.modules:
                    continue
                if not importable(requirement[1]):
                    LOG.warning("Module requirement '%s' (module=%s) "
                                "for resource '%s' (%s) not found.", name,
                                requirement[1], self.key, type(self).__name__)
                    return False
        return True

    def _prepare(self, install: bool = False, **kwargs):
        # pylint: disable=arguments-differ
        """Load the required resources.
        """
        super()._prepare(**kwargs)

        # FIXME[concept]:
        # In some situations, one requirement has to be prepared in
        # order to check for other requirements.
        # Example: checking the availability of an OpenCV data file
        # may require the 'cv2' module to be loaded in order to construct
        # the full path to that file.

        for requirement in self._requirements.values():
            if requirement[0] == 'module' and requirement[1] not in globals():
                globals()[requirement[1]] = import_module(requirement[1])

        if not self.preparable:
            if install:
                self.install()
            else:
                raise RuntimeError("Resources required to prepare '" +
                                   type(self).__name__ +
                                   "' are not installed.")

    #
    # Installation
    #

    def install(self) -> None:
        """Install the resources required for this module.
        """
        LOG.info("Installing requirements for resource '%s'.",
                 self.__class__.__name__)
        start = time.time()
        self._install()
        end = time.time()
        LOG.info("Installation of requirements for resource '%s' "
                 "finished after %.2fs",
                 self.__class__.__name__, end-start)

    def _install(self) -> None:
        # FIXME[concept]: what is this method supposed to do
        # and which (sub)classes should implement this method.
        """Actual implementation of the installation procedure.
        """
        # to be implemented by subclasses

        # raise NotImplementedError("Installation of resources for '" +
        #                           type(self).__name__ +
        #                           "' is not implemented (yet).")


class Installable2(ABC):  # FIXME[todo]: yet another implementation
    """A class depending on third party resources that can be installed.
    """

    @property
    def installed(self) -> bool:
        """A property indicating whether the resource is fully
        installed and can be used.
        """
        return self._installed()

    @abstractmethod
    def _installed(self) -> bool:
        """The actual check if the resource has been installed that
        is to be implemented by subclasses.
        """
        return True

    def install(self, **kwargs) -> None:
        """Install the resource.
        """
        if not self.installed:
            self._install(**kwargs)

    @abstractmethod
    def _install(self, **kwargs) -> None:
        """Do the actual installation. This method has to be implemented
        by subclasses.
        """
