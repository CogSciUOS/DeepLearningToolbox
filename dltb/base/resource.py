# standard imports
import sys
import logging
import importlib
from pathlib import Path

# toolbox imports
from .busy import BusyObservable, busy


# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Metaclass(type):
    """A metaclass for the :py:class:`Resource` class. It assigns a unique
    identifier to each instance of this class and collects all object
    of the instance class(es), allowing iteration and index access.

    """
    _base_class = None

    def __new__(cls, clsname, superclasses, attributedict, **kwargs):
        """Create a new instance class of this :py:class:`Metaclass`.
        """
        logger.debug("Creating new instance class of meta class"
                     f"{cls.__name__}: {clsname} "
                     f"(superclasses: {superclasses})")
        new_class = type.__new__(cls, clsname, superclasses, attributedict,
                                 **kwargs)
        if new_class._base_class is None:
            original_init = new_class.__init__

            def init_wrapper(self, id: str, **kwargs) -> None:
                self._id = id
                original_init(self, **kwargs)

            def repr(self) -> str:
                return self._id

            new_class._instances = {}
            new_class.__init__ = init_wrapper
            new_class.__repr__ = repr
            new_class._base_class = new_class
            logger.debug(f"Added instance dictionary to class {clsname}")
        return new_class

    def __call__(cls, id: str=None, *args, **kwargs):
        """Create a new object of the instance class. The new object
        will get an id and will be inserted into the dictionary
        of instances of the base class.

        Attributes
        ----------
        id: str
            The unique identifier of the instance. If no id is provided,
            a generic id will be generated.

        Raises
        ------
        RuntimeError:
            If the id is already in ues.
        """
        if id is None:
            id = cls._base_class.__name__ + str(len(cls._instances))
        if id in cls._base_class._instances:
            raise RuntimeError(f"Ambigouos use of {cls._base_class.__name__} "
                               f" identifier '{id}'.")
        instance = super(Metaclass, cls).__call__(id=id, *args, **kwargs)
        cls._base_class._instances[id] = instance
        return instance
        
        
    def __getitem__(cls, id):
        instance = cls._base_class._instances[id]
        if not isinstance(instance, cls):
            raise TypeError(f"{cls._base_class.__name__} '{id}' has "
                            "inappropriate type "
                            f"'{instance.__class__.__name__}' while it was "
                            f"accessd as '{cls.__name__}'.")
        return instance

    def __iter__(cls):
        for instance in cls._base_class._instances.values():
            if isinstance(instance, cls):
                yield instance

    def __len__(cls) -> int:
        len = 0
        for instance in cls._base_class._instances.values():
            if isinstance(instance, cls):
                len += 1
        return len



class Resource(BusyObservable, method='resource_changed',
               changes={'status_changed'}, metaclass=Metaclass):
    """A :py:class:`Resource` is everything that may be required by a tool.
    This includes specific hardware, software, or data.

    The class supports checking for availability, offers methods to
    install or update a resource and provides information on the
    resource.
    """
    _label: str = None
    _description: str = None

    def __init__(self, label: str = None, description: str = None,
                 **kwargs) -> None:
        """Create a new resource.

        Arguments
        ---------
        name: str
            A name for the :py:class:`Resource`. It should be globally
            unique so that it can be used as an identify for this
            :py:class:`Resource`.
        description: str
            A description of the resource. This text may be presented
            to the user.
        """
        super().__init__(**kwargs)
        self.label = label
        self.description = description

    @property
    def label(self) -> str:
        return self._id if self._label is None else self._label

    @label.setter
    def label(self, label: str) -> None:
        if label is not None:
            self._label = label
        elif self._label is not None:
            del self._label

    @property
    def description(self) -> str:
        return (("No description has been provided "
                f"for Resource '{self.label}'.")
                if self._description is None else self._description)

    @description.setter
    def description(self, description: str) -> None:
        if description is not None:
            self._description = description
        elif self._description is not None:
            del self._description

    @property
    def available(self) -> bool:
        """Check the availability of this :py:class:`Resource`.  If True, the
        resource can be prepared or used directly, otherwise it is
        necessary needs to be requires installation.
        """
        return True

    def install(self) -> None:
        """Install the :py:class:`Resource`. After successful installation,
        the :py:class:`Resource` should be :py:meth:`available` (but not
        necessary :py:meth:`prepared`).
        """
        raise NotImplementedError(f"Installation of resource '{self._id}' "
                                  "is not implemented (yet), sorry!")

    def update(self) -> None:
        """Updates an installed :py:class:`Resource`.
        """
        raise NotImplementedError(f"Update of resource '{self._id}' "
                                  "is not implemented (yet), sorry!")

    @property
    def prepared(self) -> bool:
        """Check if the :py:class:`Resource` has been prepared.  If True, the
        resource can be used directly otherwise some preparation is be
        necessary, which may delay further operation.
        """
        return False

    def prepare(self, **kwargs):
        """Prepare this :py:class:`Resource`.  This may require some time.
        After preparation, the :py:class:`Resource` should be usable
        without any delay.
        """
        pass


class DummyResource(Resource):
    pass


class ModuleResource(Resource):
    """
    """
    _module: str = None
    _conda: dict = None
    _pip: str = None
    _prefix: str = None

    def __init__(self, *args, module: str = None, prefix: str = None,
                 conda: str = None, conda_channel: str='', **kwargs) -> None:
        """Initialize a new :py:class:`ModuleResource`.

        Arguments
        ---------
        module: str
            The fully qualified module name. If none is provided,
            the name of this resource will be used.
        prefix: str
            The namespace prefix to be used for the module.
        conda: str
            The conda name of this module.
        conda_channel: str
            The conda channel from which the module should
            be installed. If none is provided, the default
            channel will be used.
        """
        super().__init__(*args, **kwargs)
        self._module = self._id if module is None else module
        self._prefix = prefix
        if conda is not None:
            self.add_conda_source(conda, channel=conda_channel)

    def add_conda_source(self, name: str, channel: str='') -> None:
        """Add a conda source allowing to install the module.

        Attributes
        ----------
        name: str
            The name of the conda package.
        channel: str
            The conda channel from which the package can be installed.
        """
        if self._conda is None:
            self._conda = {}
        self._conda[channel] = name

    @property
    def prepared(self) -> bool:
        """Check, if this :py:class:`ModuleResource` was prepared,
        that is, if the module was imported.

        Returns
        ------
        True if the module was already imported, False otherwise.
        """
        return self._module in sys.modules

    @busy("Import module")
    def prepare(self):
        """Prepare this :py:class:`ModuleResource`, that is,
        import the module.
        """
        importlib.import_module(self._module)
        self.change(status_changed=True)

    @property
    def available(self) -> bool:
        """Check if this :py:class:`ModuleResource` is installed.
        """
        # It may happen that a module is loaded (in sys.modules), but
        # does not provide a __spec__ attribute, or that this
        # attribute is None. I these cases, importlib.util.find_spec
        # raises a ValueError. Hence we check beforhand, if the module
        # is loaded, which is sufficient for us (we do not need the
        # module spec) und only refer to importlib in case it is not
        # loaded.
        if self.module in sys.modules:
            return sys.modules[self.module] is not None
        return importlib.util.find_spec(self.module) is not None

    @property
    def module(self) -> str:
        return self._module
            
    @property
    def version(self) -> str:
        """Check the module version.
        """
        if not self.prepared:
            raise RuntimeError(f"Module '{self._module}' was not imported. "
                               "No version information available")
        
        module = sys.modules[self.module]
        if hasattr(module, '__version__'):
            version = str(module.__version__)
        elif self._module == "PyQt5":
            version = module.QtCore.QT_VERSION_STR
        else:
            version = "loaded, no version"
        return version

    @busy("Install module")
    def install(self, method: str='auto', channel: str=''):
        """Install this :py:class:`ModuleResource`

        Arguments
        ---------
        method: str
            The installation method to use. Supported are
            'pip' for installation with pip and 'conda' for
            conda installation. 'auto' tries to autoamtically
            determine the best installation method.
        """
        # FIXME[todo]: as installation may take some time, it should
        # be run in a background thread ...
        
        # are we using conda?
        # FIXME[todo]: provide a general function for this check
        # FIXME[hack]: this does only work in conda environments,
        # but not if the conda base environment is used
        if 'CONDA_PREFIX' in os.environ and self._conda:
            import subprocess
            command = ['conda', 'install']
            
            if channel in self._conda:
                command += '-c', channel
            command.append(self._conda[channel])
            process = subprocess.Popen(command,
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            #stdout, stderr

            # FIXME[todo]: there is a python 'conda' module
            # which may be used ...
        else:
            # use pip
            #
            pass
        self.change(status_changed=True)



class Installable(Resource, BusyObservable):

    @property
    def installed(self) -> bool:
        return self._installed()

    def _installed(self) -> bool:
        """A method to check if the py:class:`Installable` is installed.
        This should be implemented by subclasses and do the actual test.
        """

    @busy("installing")
    def install(self) -> None:
        """Install the :py:class:`Installable`.
        """
        self._install()

    def _install(self) -> None:
        """Do the actuall installation. This method should be
        implemented by subclasses.
        """

    @busy("uninstalling")
    def uninstall(self) -> None:
        """Uninstall the :py:class:`Installable`.
        """
        self._uninstall()

    def _uninstall(self) -> None:
        """Do the actuall deinstallation. This method should be
        implemented by subclasses.
        """

class Package(Installable):
    """
    """
    _name: str
    _module_name: str
    _pip_name: str
    _conda_name: str

    def __init__(self, name: str, module: str = None,
                 pip: str = None,
                 conda: str = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._name = name
        self._module_name = module or name
        self._pip_name = pip or name
        self._conda_name = conda or name

    def _installed(self) -> bool:
        """Check if the package is installed using `importlib`.
        """
        return importlib.util.find_loader(self._module_name) is not None

    def _install(self) -> None:
        """Do the actuall installation. This method should be
        implemented by subclasses.
        """

    def pip_install(self) -> None:
        """Install package via pip.
        """

    def conda_install(self) -> None:
        """Install package with conda.
        """


class GitRepository(Installable):
    """
    """
    # FIXME[todo]


class Downloadable(Installable):
    """
    """
    _file: Path = None
    _url: str = None
    _checksum: str = None


class ResourceUser:

    @classmethod
    def add_class_resource(cls, resource: Resource) -> None:
        """Add a class resource, that is a resource to be used
        by the class itself or all instances of the class.
        A typical example would be a third-party software package.
        """
        cls._resources.append(resource)

    def add_instance_resource(self, resource: Resource) -> None:
        """Add an instance resource, that is a resource to be used by one
        specific instances of the class.  A typical example would be a
        model file.
        """
        self._resources.append(resource)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._resources = []

    def resources(self) -> Iterable[Resource]:
        for resource in type(self)._resources:
            yield resource
        for resource in self._resources:
            yield resource

    def resources_available(self) -> bool:
        for resource in self.resources():
            if not resource.available():
                return False
        return True

    def install_resources(self) -> None:
        for resource in self.resources():
            if isinstance(resource, Installable):
                resource.install()

##############################################################################

Resource(id='test')
Resource(id='test2', label='abc')
Resource(id='test3', description='xyz')

DummyResource(description='abc')

ModuleResource(id='numpy', prefix='np', label="NumPy",
               description="NumPy is the fundamental package for "
               "scientific computing with Python.");
ModuleResource(id='tensorflow', prefix='tf', label="TensorFlow")
ModuleResource(id='keras', label="Keras")
ModuleResource(id='appsdir')
ModuleResource(id='matplotlib')
ModuleResource(id='opencv', module='cv2', label="OpenCV")
ModuleResource(id='caffe')
ModuleResource(id='qt', module='PyQt5', label="Qt")
ModuleResource(id='pycuda', conda='pycuda', conda_channel='lukepfister')
ModuleResource(id='lucid',
               description="A collection of infrastructure and tools "
               "for research in neural network interpretability.")
ModuleResource(id='imutils',
               description='A series of convenience functions to make '
               'basic image processing functions such as translation, '
               'rotation, resizing, skeletonization, displaying '
               'Matplotlib images, sorting contours, detecting edges, '
               'and much more easier with OpenCV and both '
                'Python 2.7 and Python 3.')
ModuleResource(id='dlib',
               description='Dlib is a modern C++ toolkit containing '
               'machine learning algorithms and tools '
               'for creating complex software to solve real world problems.')
ModuleResource(id='ikkuna',
               description='A tool for monitoring neural network training.')
ModuleResource(id='sklearn',
               description='Machine Learning in Python.')

##############################################################################

