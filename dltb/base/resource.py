"""General functionality for managing resources.

FIXME[todo]: this code is still subject to development.
  It is currently only used in qtgui/panels/internals.py

FIXME[concept]: the current implementation seems to mix several ideas:
 - a `Resource` as something that can or can not be available
   -> different types of resources (package, module, repository, file,
      directory, model, maybe in a specific version)
   -> checking availability of resources can be part of checking
      if a `Preparable` object can be prepared
 - an `Installable` as something that can be installed.
   - checking if it is installed
   - perform installation und deinstallation (maybe asynchronous)
   - downloading and unpacking can be a way to install something
     (but there may be other ways, like using a library or a package
     manager)
   - all Installables are Resources - they are available if installed.
- registering resources, so that they are available by key
  -> introduce a `Resourcelike` as Union[Resource, ] type that can be passed a argument.
  -> Resources should be `Registrable`.
  -> no need for the `Metaclass`

FIXME[design]: there is also a resource related stuff in other modules:
 - dltb/util/importer.py: function managed_resource
 - dltb/util/resource


FIXME[design]: what is the difference between installation and preparation?
 - both processes aim at making an unavailable resource available
 - both processes are considered computational expensive
   (though installation is typically on a different scale than
   e.g., importing a module).
 - both processes may fail
 - in case of preparation, one may check if the process
     is likely to succeed, that is, if the object is preparable.
     typically by checking if certain dependencies are satisfied.
       -> if these are resources that can be made available
          then the preparability can be changed.
   in case of installation, one assumes that it will succeed,
     but there is little more to do if it fails.
 - preparation is typically an internal process, while
   installation is more external:
      needs contact to the outside world
      can be performed by external tools

FIXME[design]: what is the relation between the ``Preparable``
resource defined in this module and the ``Preparable`` from
module ``prepare``?  Can these two classes merged?
  - should all ``Preparables`` be made resources?



Resources can include hardware and software resources, including
libraries, models, and datasets.

A resource should provide a test to check if it is available.


Availability can mean different things in different contexts. We may
need to clarify this. Some ambiguities I noted so far are:

  * availability of a Python class can mean that (the module defining
    that) class can be imported (the module is available), or that
    the class definition is already loaded (its module has already
    been imported).

  * availablity of a Python module (similar to classes): the module
    can be imported (is installed) or the model has already been
    imported.

  * availability of a Preparable object: (a) the object is
    instantiated, (b) the object is instantiated and is preparable
    or (c) the object is prepared

Less controversial are the following cases:

  * a python package is available, if it is installed on the
    system

  * files are available, if they are present (usually on the local
    file system) and can be opened.

  * hardware (like a GPU) is available, if it is physically present
    in the system (it may still be unavailable, if it is used by
    some other program).

A resource can be installable, meaning that it provides a method to
install it.  Installation can include downloading, unpacking to a
desired location.

"""

# standard imports
from types import ModuleType
import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import Iterable, Union, Optional

# toolbox imports
from ..util.importer import importable, import_module
from .observer import Observable
from .busy import BusyObservable, busy
from .register import RegisterEntry, RegisterClass
from .info import Info, InfoSource

# logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


class Resource(Observable, RegisterEntry, register=True,
               method='resource_changed', changes={'availability_changed'}):
    """A :py:class:`Resource` is everything that may be required by a tool.
    This includes specific hardware, software, or data.

    The base ``Resource`` class only supports checking for
    availability, and allows interested parties to observe the
    ``Resource`` to be notified when the availability changes
    (subclasses may offer additional methods to make the ``Resource``
    available, e.g., by download or installation, but the base class
    does not assume that this is possible).

    As a :py:class:``Registrable``, the ``Resource`` has a unique key
    that allows to identify a resource.  The ``Resource`` class also
    provides a register of all instances.

    A ``Resource`` can also be annotated with a human readable
    ``description``.

    """
    _description: str = None

    def __init__(self, description: str = None, **kwargs) -> None:
        """Create a new resource.

        Arguments
        ---------
        description: str
            A description of the resource. This text may be presented
            to the user.
        """
        super().__init__(**kwargs)
        self.description = description

    @property
    def description(self) -> str:
        """A verbal description of this resource.
        """
        return (("No description has been provided "
                 f"for Resource '{self.key}'.")
                if self._description is None else self._description)

    @description.setter
    def description(self, description: str) -> None:
        if description is not None:
            self._description = description
        elif self._description is not None:
            del self._description

    @property
    def available(self) -> bool:
        """Check the availability of this :py:class:`Resource`.  If ``True``,
        the resource can be used directly, otherwise it is (currently)
        not usable.

        """
        return self._available()

    @staticmethod
    def _available() -> bool:
        return True  # to be implemented by subclasses

    def availability_changed(self) -> None:
        """Notify observers that the availability of this ``Resource`` has
        changed.

        """
        self.change(availability_changed=True)


class Dependant:
    """A `Dependant` is depending on the availability of some `Resource`.
    Dependencies may be obligatory (`required=True`) or optional
    (`required=False`).  A `Dependant` can only be used (prepared), if
    all required `Resources` are available.

    A `Dependant` provides the `add_dependency` method to add
    dependencies.  This should usually be done upon initialization,
    before the object is prepared for use.

    """
    # FIXME[todo]: maybe a better name - e.g. "Consumer" (but it does not
    # really consume the resource), "Dependant" (a dependant is someone
    # who relies on someone else) - add_dependency(resource, required=True)

    _dependencies: Iterable[Resource] = []

    @classmethod
    def add_class_dependency(cls, resource: Resource) -> None:
        """Add a class resource, that is a resource to be used
        by the class itself or all instances of the class.
        A typical example would be a third-party software package.
        """
        cls._dependencies.append(resource)

    def add_instance_dependency(self, resource: Resource) -> None:
        """Add an instance resource, that is a resource to be used by one
        specific instances of the class.  A typical example would be a
        model file.
        """
        self._dependencies.append(resource)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._dependencies = []

    def dependencies(self) -> Iterable[Resource]:
        """Iterates over the resources on which this ``Dependant`
        depends.
        """
        for resource in type(self)._dependencies:
            yield resource
        for resource in self._dependencies:
            yield resource

    def resources_available(self) -> bool:
        """Check if all resources required to use this class are available.
        """
        for resource in self.dependencies():
            if not resource.available():
                return False
        return True

    def install_dependencies(self) -> None:
        """Make sure that all (installable) resources required for using
        this class are available and install those that are missing.
        """
        for resource in self.dependencies():
            if isinstance(resource, Installable):
                resource.install()


class Preparable(Resource, BusyObservable):
    """A :py:class:``Resource`` that can be prepared.  Preparation can
    make an unavailable resource available.

    """
    # FIXME[concept]: clarify relation to class `dltb.base.prepare.Preparable`

    @property
    def prepared(self) -> bool:
        """Check if the :py:class:`Resource` has been prepared.  If True, the
        resource can be used directly otherwise some preparation is be
        necessary, which may delay further operation.
        """
        return self._prepared()

    @staticmethod
    def _prepared() -> bool:
        return True

    @busy("preparing")
    def prepare(self):
        """Prepare this :py:class:`Resource`.  This may require some time.
        After preparation, the :py:class:`Resource` should be usable
        without any delay.
        """
        self._prepare()
        self.availability_changed()

    def _available(self) -> bool:
        return self.prepared


class Installable(Resource, BusyObservable):
    """A :py:class:``Resource`` that can be installed.  Installation can
    make an unavailable resource available.  An `Installable` object
    provides methods to check if it is installed, and to install or
    uninstall it.

    """

    @property
    def installed(self) -> bool:
        """A flag indicating if this py:class:`Installable` is installed.
        """
        return self._installed()

    def _installed(self) -> bool:
        """A method to check if the py:class:`Installable` is installed.
        This should be implemented by subclasses and do the actual test.
        """
        return False

    @busy("installing")
    def install(self, **kwargs) -> None:
        """Install the :py:class:`Installable`. After successful installation,
        the :py:class:`Resource` should be :py:meth:`available`.
        """
        self._install(**kwargs)
        self.availability_changed()

    def _install(self, **kwargs) -> None:
        raise NotImplementedError(f"Installation of resource '{self.key}' "
                                  "is not implemented (yet), sorry!")

    @busy("uninstalling")
    def uninstall(self) -> None:
        """Uninstall the :py:class:`Installable`.
        """
        self._uninstall()

    def _uninstall(self) -> None:
        """Do the actuall deinstallation. This method should be
        implemented by subclasses.
        """

    def _available(self) -> bool:
        return self.installed


class GitRepository(Installable):
    """A `GitRepository` provides source code managed by the git version
    control system.  The source code can be cloned from the repository,
    allowing to either install it or directly use it in place.
    """
    # FIXME[todo]

    def _install(self, **kwargs) -> None:
        """Clone the git repository.
        """


class Downloadable(Installable):
    """A `Downloadable` resource can be installed by downloading and
    unpacking it into a suitable location.
    """
    _file: Path = None
    _url: str = None
    _checksum: str = None

    def _install(self, **kwargs) -> None:
        """Download and unpack to target location
        """
