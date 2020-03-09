#
# Hardware/CUDA stuff
#

import resource
import subprocess
import re

try:
    from py3nvml import py3nvml
    try:
        py3nvml.nvmlInit()
    except py3nvml.NVMLError_LibRmVersionMismatch as e:
        # "RM has detected an NVML/RM version mismatch."
        py3nvml = None
except ModuleNotFoundError as e:
    # "No module named 'py3nvml'"
    py3nvml = None
except ImportError as e:
    # "cannot import name 'py3nvml'"
    py3nvml = None

class _component(dict):
    def __init__(self, name = None):
        if name is not None:
            self['name'] = name

    def __getattr__(self, key):
        if key in self:
            return self[key]

    def __setattr__(self, key, value):
        self[key] = value


cuda = _component()
try:
    cuda.nvidia_smi = subprocess.check_output(["nvidia-smi"],
                                              universal_newlines=True)
    cuda.nvidia_smi_l = subprocess.check_output(["nvidia-smi", "-L"],
                                                universal_newlines=True)
    cuda.nvidia_smi_q = subprocess.check_output(["nvidia-smi", "-q"],
                                                universal_newlines=True)
    # -q = query: GPU temperature, memory, ...

    match = re.search('Driver Version: ([^ ]*)', cuda.nvidia_smi)
    if match:
        cuda.driver_version = match.group(1)
    else:
        cuda.driver_version = '?'

    match = re.search('CUDA Version: ([^ ]*)', cuda.nvidia_smi)
    if match:
        cuda_version = match.group(1)
    else:
        nvcc_v = subprocess.check_output(["nvcc", "--version"],
                                         universal_newlines=True)
        match = re.search(', V([0-9.]*)', nvcc_v)
        if match:
            cuda_version = match.group(1)
        else:
            cuda_version = '?'

    match = re.search('GPU [\d]+: (.*) \(UUID', cuda.nvidia_smi_l)
    if match:
        gpus = [_component(match.group(1))]
    else:
        gpus = [_component('?')]


except FileNotFoundError as e:
    cuda = None
    gpus = []

except subprocess.CalledProcessError as e:
    # Command '['nvidia-smi']' returned non-zero exit status 231.
    cuda = None
    gpus = []


#
# processor name
#
import platform
import subprocess
_processor_name = '?'
if platform.system() == "Windows":
    _processor_name = platform.processor()
elif platform.system() == "Darwin":
    os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
    command ="sysctl -n machdep.cpu.brand_string"
    _processor_name = subprocess.check_output(command).strip()
elif platform.system() == "Linux":
    command = "cat /proc/cpuinfo"
    all_info = str(subprocess.check_output(command, shell=True,
                                           universal_newlines=True))
    for line in all_info.split("\n"):
        if "model name" in line:
            _processor_name = re.sub( ".*model name.*:", "", line,1)
cpus = [_component(_processor_name)]

mem = _component()

def update(initialize:bool = False):
    global gpus, mem
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    # The Python docs aren't clear on what the units are exactly,
    # but the Mac OS X man page for getrusage(2) describes the
    # units as bytes. The Linux man page isn't clear, but it seems
    # to be equivalent to the information from /proc/self/status,
    # which is in kilobytes.
    mem.shared = rusage.ru_ixrss
    mem.unshared = rusage.ru_idrss
    mem.peak = rusage.ru_maxrss

    if cuda is not None:
        gpu = -1
        mem_section = False
        for line in subprocess.check_output(["nvidia-smi", "-q"],
                                            universal_newlines=True).splitlines():
            if re.match('GPU ', line):
                gpu += 1; continue
            elif gpu < 0:
                continue

            match = re.match(' *GPU Current Temp *: ([^ ]*) C', line)
            if match: gpus[gpu].temperature = match.group(1); continue
            if initialize:
                match = re.match(' *GPU Shutdown Temp *: ([^ ]*) C', line)
                if match: gpus[gpu].temperature_max = match.group(1); continue

            match = re.match('  *(\w*) Memory Usage', line)
            if match: mem_section = (match.group(1) == 'FB'); continue

            if mem_section:
                match = re.match(' *Used *: ([^ ]*) MiB', line)
                if match: gpus[gpu].mem = int(match.group(1)); continue

                if initialize:
                    match = re.match(' *Total *: ([^ ]*) MiB', line)
                    if match:
                        gpus[gpu].mem_total = int(match.group(1)); continue

update(True)

import logging
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
            

from base.observer import BusyObservable, busy_message, change

class Resource(BusyObservable, method='resource_changed',
               changes=['status_changed'],
               metaclass=Metaclass):
    """A :py:class:`Resource` is everything that may be required by a tool.
    This includes specific hardware, software, or data.

    The class supports checking for availability, offers methods to
    install or update a resource and provides information on the
    resource.
    """
    _label: str = None
    _description: str = None
   
    def __init__(self, label: str=None, description: str=None,
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

from base import View as BaseView, Controller as BaseController, run

class View(BaseView, view_type=Resource):
    """View on :py:class:`Resource`.

    Attributes
    ----------
    _resource: Resource
        The resource viewed by this View.

    """

    def __init__(self, resource: Resource=None, **kwargs):
        super().__init__(observable=resource, **kwargs)


class Controller(View, BaseController):
    """Controller for :py:class:`Resource`.
    This class allows to control a :py:class:`Resource` in a
    multi-threaded environment.

    """
    
    def __init__(self, resource: Resource, **kwargs) -> None:
        """
        Parameters
        ----------
        resource: Resource
        """
        super().__init__(resource=resource, **kwargs)
        self._next_image = None

    @run
    def install(self, **kwargs):
        """Install :py:class:`Resource` controlled by this
        :py:class:`Controller` resource. After successful
        installation, the :py:class:`Resource` should be
        available (but not necessary prepared).
        """
        self._resource.install(**kwargs)

    @run
    def prepare(self, **kwargs):
        """Prepare :py:class:`Resource` controlled by this
        :py:class:`Controller` resource. After preparation,
        the :py:class:`Resource` should be ready for use.
        """
        self._resource.prepare(**kwargs)


class DummyResource(Resource):
    pass

import sys
import importlib
    
class ModuleResource(Resource):
    """
    """
    _module: str = None
    _conda: dict = None
    _pip: str = None
    _prefix: str = None

    def __init__(self, *args, module: str=None, prefix: str=None,
                 conda: str=None, conda_channel: str='', **kwargs) -> None:
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

        Result
        ------
        True if the module was already imported, False otherwise.
        """
        return self._module in sys.modules

    @busy_message("Import module")
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

    @busy_message("Install module")
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


