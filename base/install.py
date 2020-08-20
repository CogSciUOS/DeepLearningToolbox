"""Support for installation of resources.
"""

# standard imports
from abc import ABC, abstractmethod

class Installable(ABC):
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
