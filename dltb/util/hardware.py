"""Functions to access hardware and system resources.
"""

# standard imports
from typing import Iterable, Tuple, Optional, Union, overload
from types import ModuleType
from pathlib import Path
import re
import os
import sys
import time
import platform
import resource
import itertools
import importlib
import subprocess

# toolbox imports
from ..typing import Literal
from ..base.types import classproperty
from ..base.info import InfoSource, Unit
from ..base.meta import Postinitializable
from ..base.implementation import Implementable
from .importer import Importer


#
# Hardware
#

class HardwareInfo(InfoSource):
    """Information on the system hardware.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        #
        # CPU
        #
        self.processor_name = '?'
        if platform.system() == "Windows":
            self.processor_name = platform.processor()
        elif platform.system() == "Darwin":
            os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
            command = "sysctl -n machdep.cpu.brand_string"
            self.processor_name = subprocess.check_output(command).strip()
        elif platform.system() == "Linux":
            command = "cat /proc/cpuinfo"
            all_info = str(subprocess.check_output(command, shell=True,
                                                   universal_newlines=True))
            for line in all_info.split("\n"):
                if "model name" in line:
                    self.processor_name = \
                        re.sub(".*model name.*:", "", line, 1)
        #
        # Memory
        #

        # memory size in bytes
        self.memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        # SC_PAGE_SIZE is often 4096.
        # SC_PAGESIZE and SC_PAGE_SIZE are equal.

    def initialize_info(self) -> None:
        """Register hardware specific information.
        """
        super().initialize_info()

        self.add_info('cpu', self.processor_name,
                      title="CPU")
        # for idx, cpu in enumerate(cpus):
        #     self.add_info('cpu{idx}', cpu,
        #                   title="{idx+1}. CPU")

        # for idx, gpu in enumerate(gpus):
        #     self.add_info('cpu{idx}', gpu,
        #                   title="{idx+1}. GPU")

        self.add_info('memory', self.memory >> 20,
                      title="Memory", unit=Unit.MEMORY)

#
# Resources
#

class ResourceInfo(InfoSource):
    """Information on the system resources.
    """
    # MINIMAL_UPDATE_INTERVAL: minimal update interval in seconds
    MINIMAL_UPDATE_INTERVAL: float = 1

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._last_update = 0
        self._rusage = None
        self._process = None
        self.update()

    def initialize_info(self) -> None:
        """Register resource information.
        """
        super().initialize_info()

        #
        # resource
        #

        # 'resource' is a standard library module.
        #
        # For Unixes (Linux, Mac OS X, Solaris) you can use the
        # getrusage() function from the standard library module
        # resource. The resulting object has the attribute ru_maxrss,
        # which gives peak memory usage for the calling process.
        #
        # The Python docs aren't clear on what the units are exactly,
        # but the Mac OS X man page for getrusage(2) describes the
        # units as bytes. The Linux man page isn't clear, but it seems
        # to be equivalent to the information from /proc/self/status,
        # which is in kilobytes.
        self.add_info('mem_peak', lambda: self.peak_memory,
                      title="Peak memory usage", unit=Unit.MEMORY)
        self.add_info('mem_shared', lambda: self.shared_memory,
                      title="shared memory", unit=Unit.MEMORY)
        self.add_info('mem_unshared', lambda: self.unshared_memory,
                      title="Unshared memory", unit=Unit.MEMORY)

        #
        # psutil
        #

        # Memory (2)
        # a useful solution that works for various operating systems,
        # including Linux, Windows 7, etc.:
        try:
            psutil = importlib.import_module('psutil')
            mem = psutil.virtual_memory()
            # mem.total: total physical memory available
            self.add_info('mem_total', mem.total,
                          title="Total memory", unit=Unit.MEMORY)

            self._process = psutil.Process(os.getpid())
            self.add_info('mem_usage', lambda: self.memory,
                          title="Memory usage", unit=Unit.MEMORY)
        except ModuleNotFoundError:
            pass

    def update(self) -> None:
        """Update the resource information
        """
        if time.time() - self._last_update < self.MINIMAL_UPDATE_INTERVAL:
            return False  # avoid too frequent updates

        self._rusage = resource.getrusage(resource.RUSAGE_SELF)

    @property
    def shared_memory(self) -> int:
        """Amount of shared memory (ru_ixrss).
        """
        return self._rusage.ru_ixrss

    @property
    def unshared_memory(self) -> int:
        """Amount of unshared memory (ru_ixrss).
        """
        return self._rusage.ru_idrss

    @property
    def peak_memory(self) -> int:
        """Peak memory usage (ru_maxrss).
        """
        self.update()
        return self._rusage.ru_maxrss

    @property
    def memory(self) -> int:
        """Memory usage (rss).
        """
        return -1 if self._process is None else self._process.memory_info().rss

#
# System
#

class SystemInfo(InfoSource):
    """Information on the system.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def initialize_info(self) -> None:
        """Register system specific information.
        """
        super().initialize_info()

        # uname is a tuple (system, node, release, version, machine, processor)
        # self.add_info('uname', platform.uname(), title="Uname")
        self.add_info('node', platform.node(), title="Node")
        self.add_info('system', platform.system(), title="System")
        self.add_info('release', platform.release(), title="Release")
        # The version string can be very long: e.g.
        # '#36~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Fri Feb 17 15:17:25 UTC 2'
        # self.add_info('version', platform.version(), title="Version")
        # machine and processor seem to be the same
        self.add_info('machine', platform.machine(), title="Machine")
        # self.add_info('processor', platform.processor(), title="Processor")


#
# Python
#

class PythonInfo(InfoSource):
    """Information on the Python installation.
    """

    def initialize_info(self) -> None:
        """Register Python specific information.
        """
        super().initialize_info()

        self.add_info('version', sys.version,
                      title="Python version")
        self.add_info('platform', sys.platform,
                      title="Platform")
        self.add_info('prefix', sys.prefix,
                      title="Prefix")
        self.add_info('executable', sys.executable,
                      title="Executable")

#
# GPU/CUDA support
#

class CudaDeviceInfo(InfoSource):
    """An object providing information on an individual CUDA devie.
    """
    _device_number: int

    def __new__(cls, device_number: Optional[int] = None,
                gpu: Optional[int] = None, **kwargs) -> None:
        if gpu is None and device_number is None:
            raise TypeError("Constructing a CudaDeviceInfo requires "
                            "a numerical 'device_number' argument.")
        return super().__new__(cls, **kwargs)

    def __init__(self, device_number: Optional[int] = None,
                 gpu: Optional[int] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._device_number = gpu if device_number is None else device_number

    @property
    def device_number(self) -> int:
        """The GPU number, start counting at `0`.
        """
        return self._device_number

    @property
    def pci_device(self) -> str:
        """A string identifying the PCI device providing the GPU.
        """
        return "?"


class CudaInfo(Implementable, InfoSource, Postinitializable):
    """Information on CUDA and CUDA devices.

    There are different python bindings for the NVIDIA Management
    Library (NVML):
     * nvidia-ml-py [4], referred to as "the original", by NVIDIA, seems to
       be automatically generated, lacks docstrings and Python3 support,
       last change 2017
     * pynvml [1], I started from scratch (not adapting some automatically
       generated Python2 bingings), but the number of supported functions
       is low. This module is also used by `torch.cuda`. Last change 2017, 
     * nvidia-ml-py3 [2], last change 2019
     * py3nvml [3], last change 2022
     * pycuda [5]
    
    Class Attributes
    ----------------

    nvml_module: Optional[ModuleType]
        A reference to the NVIDIA Management Library.

    References
    ----------
    [1] https://github.com/lukeyeager/pynvml
    [2] https://github.com/nicolargo/nvidia-ml-py3
    [5] https://github.com/inducer/pycuda
    """
    # FIXME[todo]: we should include some code that checks for a version
    # mismatch between the version of the operating system's NVIDIA driver
    # and the driver version expected by the CUDA toolkit.
    # Such a situation can occur if the CUDA is updated on a Linux/Ubuntu
    # system but the system was not rebooted yet.

    nvml_module: Optional[ModuleType] = None

    @staticmethod
    def initialize_nvml_module() -> Optional[ModuleType]:
        """Initialize the Python bindings module.
        """
        return None

    def __new__(cls, **kwargs) -> 'CudaInfo':
        if cls.nvml_module is None:
            try:
                cls.nvml_module = cls.initialize_nvml_module()
            except ValueError as error:
                raise TypeError("No NVML module available to initialize "
                                f"instance of {cls}") from error
        return super().__new__(cls, **kwargs)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._devices = []

    def __post_init__(self) -> None:
        super().__post_init__()
        self._initialize_devices()

    def _initialize_devices(self) -> None:
        """Initialize the list of device infos.
        """
        # to be implemented by subclasses

    @classproperty
    def cuda_available(cls) -> bool:
        # pylint: disable=no-self-argument
        """Check if CUDA functionality is available.
        """
        return cls.nvml_module is not None and cls.have_nvidia_driver()

    @classmethod
    def have_nvidia_driver(cls) -> bool:
        # pylint: disable=no-self-argument
        """Check if the system has an NVIDIA kernel driver.
        """
        return False

    @property
    def number_of_devices(self) -> int:
        """Number of CUDA devices available.
        """
        return len(self._devices)

    def device_infos(self) -> Iterable[CudaDeviceInfo]:
        """Iterate the available CUDA devices. 
        """
        return ()

    def __getitem__(self, idx: int) -> CudaDeviceInfo:
        """Index accesss to individual device infos.
        """
        return self._devices[idx]

    def __iter__(self) -> Iterable[CudaDeviceInfo]:
        """Iterate the available devices.
        """
        return iter(self._devices)

    def initialize_info(self) -> None:
        """Register CUDA specific information.
        """
        super().initialize_info()

        self.add_info('cuda_available', self.cuda_available,
                      title="CUDA is available")


#
# nvidia-smi based implementation
#
class NvidiaSmiDeviceInfo(CudaDeviceInfo):
    # pylint: disable=too-many-instance-attributes
    """A `CudaDeviceInfo` based on the output of `nvidia-smi`.
    """
    def __init__(self, cuda_info: Optional['CudaInfo'] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = '?'
        self._temperature = -1
        self.slow_temperature = -1
        self.max_temperature = -1
        self.free_memory = -1
        self.used_memory = -1
        self.reserved_memory = -1
        self.total_memory = -1
        self._cuda_info = cuda_info
        self._initialize_properties()

    def _initialize_properties(self) -> None:
        gpus = (self.device_number, )
        sections = ('', 'FB Memory Usage', 'Temperature')
        attributes = ('Product Name', 'Total',
                      'GPU Slowdown Temp', 'GPU Shutdown Temp')
        for _, section, attribute, value in \
                self._cuda_info.nvidia_query_devices(gpus=gpus,
                                                     sections=sections,
                                                     attributes=attributes):
            if not section:
                if attribute == 'Product Name':
                    self.name = value
            elif section =='FB Memory Usage':
                if attribute == 'Total':
                    self.total_memory = int(value.split(' ')[0])
            elif section =='Temperature':
                if attribute == 'GPU Slowdown Temp':
                    self.slow_temperature = int(value.split(' ')[0])
                elif attribute == 'GPU Shutdown Temp':
                    self.max_temperature = int(value.split(' ')[0])
        self.update_properties(force=True)

    def update_properties(self, force: bool = False) -> None:
        """Update the dynamic properties of this `NvidiaSmiDeviceInfo`.

        This will call :py:meth:`NvidiaSmiInfo.update_nvidia_smi` to
        obtain new data and then parses the dynamic attributes.

        Attributes
        ----------
        force:
            Usually the parsing will be skipped if no new nvidia-smi
            information are available.  Setting this to `True` will
            force parsing independent of the nvidia-smi call. This
            may be useful durign initialization.
        """
        if not self._cuda_info.update_nvidia_smi(flags='-q') and force:
            return  # no new data available
        gpus = (self.device_number, )
        sections = ('FB Memory Usage', 'Temperature')
        for _, section, attribute, value in self.\
                _cuda_info.nvidia_query_devices(gpus=gpus, sections=sections):
            if section == 'Temperature':
                if attribute == 'GPU Current Temp':
                    self._temperature = int(value.split(' ')[0])
            elif section == 'FB Memory Usage':
                if attribute == 'Reserved':
                    self.reserved_memory = int(value.split(' ')[0])
                if attribute == 'Used':
                    self.used_memory = int(value.split(' ')[0])
                if attribute == 'Free':
                    self.free_memory = int(value.split(' ')[0])

    def initialize_info(self) -> None:
        super().initialize_info()
        self.add_info('name', self.name, title='Device Name')
        self.add_info('temp', lambda: self.temperature,
                      title='Temperature', unit=Unit.TEMPERATURE)
        self.add_info('max_temp', self.max_temperature,
                      title='Maximal temperature', unit=Unit.TEMPERATURE)

        self.add_info('free_mem', lambda: self.free_memory,
                      title='Free memory', unit=Unit.MEMORY)
        self.add_info('used_mem', lambda: self.used_memory,
                      title='Used memory', unit=Unit.MEMORY)
        self.add_info('reserved_mem', lambda: self.reserved_memory,
                      title='Reserved memory', unit=Unit.MEMORY)
        self.add_info('total_mem', self.total_memory,
                      title='Free memory', unit=Unit.MEMORY)

    @property
    def temperature(self) -> float:
        """The device temperature in degrees Celsius.
        """
        self.update_properties()
        return self._temperature


class NvidiaSmiInfo(CudaInfo):
    """Implementation of :py:class:`CudaDeviceInfo` based on parsing
    `nvidia-smi` output.

    This is a fallback solution that should only be used if no real
    NVML wrapper is installed.

    """

    # MINIMAL_UPDATE_INTERVAL: minimal update interval in seconds
    MINIMAL_UPDATE_INTERVAL: float = 1

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._last_update = 0
        self.nvidia_smi = ''
        self.nvidia_smi_l = ''
        self.nvidia_smi_q = ''
        self.update_nvidia_smi()

        self.driver_version = '?'
        self.cuda_version = '?'
        self.num_gpus = -1
        self._initialize_properties()

    def _initialize_devices(self) -> None:
        """Initialize the list of device infos.
        """
        # use "nvidia-smi -L"
        for line in self.nvidia_smi_l.splitlines():
            match = re.search(r'GPU ([\d]+): (.*) \(UUID: (GPU-[-0-9a-f]+)\)',
                              line)
            idx, _name, _uuid = match.group(1), match.group(2), match.group(3)
            device = NvidiaSmiDeviceInfo(gpu=int(idx), cuda_info=self)
            self._devices.append(device)

    def update_nvidia_smi(self,
                          flags: Union[None, str, Tuple[str]] = None) -> bool:
        """Update the information by calling `nvidia-smi` again.
        """
        if time.time() - self._last_update < self.MINIMAL_UPDATE_INTERVAL:
            return False  # avoid too frequent updates

        if flags is None:
            flags = ('', '-L', '-q')
        elif isinstance(flags, str):
            flags = (flags, )
        for flag in flags:
            if flag == '-L':
                # -L, --list-gpus: List each of the NVIDIA GPUs in the
                #                  system, along with their UUIDs.
                cmd, attr = ['-L'], 'nvidia_smi_l'
            elif flag == '-q':
                # -q = query: GPU temperature, memory, ...
                cmd, attr = ['-q'], 'nvidia_smi_q'
            elif flag == '':
                cmd, attr = [], 'nvidia_smi'
            else:
                raise ValueError(f"Unsupported flag '{flag}' for calling "
                                 "nvidia-smi.")
            try:
                output = subprocess.check_output(["nvidia-smi"] + cmd,
                                                 universal_newlines=True)
                setattr(self, attr, output)
            except (FileNotFoundError, subprocess.CalledProcessError) as error:
                # Command '['nvidia-smi']' returned non-zero exit status 231.
                #
                # In case of a driver version mismatch, `nvidia-smi` command
                # line tool fails with the following error message:
                # "Failed to initialize NVML: Driver/library version mismatch
                raise RuntimeError("Invoking 'nvidia-smi' failed") from error

        self._last_update = time.time()
        for device in self._devices:
            device.update_properties()
        return True  # update successful

    def _initialize_properties(self) -> None:
        gpus = (-1, )  # general properties
        attributes = ('Driver Version', 'CUDA Version', 'Attached GPUs')
        for _gpu, _section, attribute, value in self.\
                nvidia_query_devices(gpus=gpus, attributes=attributes):
            if attribute == 'Driver Version':
                self.driver_version = value
            elif attribute == 'CUDA Version':
                self.cuda_version = value
            elif attribute == 'Attached GPUs':
                self.num_gpus = value

    def nvidia_query_devices(self, gpus: Optional[Tuple[int]] = None,
                             sections: Optional[Tuple[str]] = None,
                             attributes: Optional[Tuple[str]] = None
                             ) -> Iterable[Tuple[int, str, str]]:
        """Iterate the device sections from an `nvidia-smi` query (`-q`).
        The output will consist of triples (gpu, section, line).
        """
        gpu = -1
        section = ''
        if attributes is None:
            re_attributes = '^ *([^:]+[^ ;]) *: *(.*)'
        elif isinstance(attributes, str):
            re_attributes = '^ *(' + attributes + ') *: *(.*)'
        else:
            re_attributes = '^ *(' + ('|'.join(attributes)) + ') *: *(.*)'
        for line in self.nvidia_smi_q.splitlines():
            if re.match('^GPU ', line):
                gpu += 1
                section = ''
                continue
            if gpu < 0 or (gpus is not None and gpu not in gpus):
                continue

            match = re.match('^    ([^:]+)$', line)
            if match:
                section = match.group(1)
                continue
            if re.match('^    ([^ ])', line):
                section = ''
            if sections is not None and section not in sections:
                continue

            match = re.match(re_attributes, line)
            if not match:
                continue
            yield (gpu, section, match.group(1), match.group(2))


#
# pycuda implementation
#

class PycudaDeviceInfo(CudaDeviceInfo):
    """Pycuda implementation of :py:class:`CudaDeviceInfo`.
    """

    def __init__(self, cuda_info: Optional['PycudaInfo'] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._cuda_info = cuda_info
        self._device = cuda_info.nvml_module.driver.Device(self.device_number)

    def initialize_info(self) -> None:
        super().initialize_info()

        attrs = self._device.get_attributes()
        for key, value in attrs.items():
            self.add_info(key, value)


class PycudaInfo(CudaInfo):
    """A :py:class:`CudaInfo` implementation based on the third-party
    package `pycuda`.

    """

    @staticmethod
    def initialize_nvml_module() -> Optional[ModuleType]:
        """Initialize the `pycuda` module.
        """
        try:
            Importer.import_module('pycuda.autoinit')
            Importer.import_module('pycuda.driver')
        except ImportError as error:
            # ImportError: libcurand.so.8.0
            # The problem occurs with the current anaconda version
            # (2017.1, "conda install -c lukepfister pycuda").
            # The dynamic library "_driver.cpython-36m-x86_64-linux-gnu.so"
            # is linked against "libcurand.so.8.0". However, the cudatookit
            # installed by anaconda is verion 9.0.
            raise ValueError("Error importing third-party module 'pycuda'") \
                from error
        return sys.modules['pycuda']

    def _initialize_devices(self) -> None:
        """Initialize the list of device infos.
        """
        for idx in range(self.nvml_module.driver.Device.count()):
            self._devices.append(PycudaDeviceInfo(cuda_info=self, gpu=idx))

    def initialize_info(self) -> None:
        """Register CUDA specific information.
        """
        super().initialize_info()
        pycuda = self.nvml_module
        pycuda_driver = pycuda.driver

        self.add_info('cuda_driver_version', pycuda_driver.driver_version,
                      title="NVIDIA Kernel driver")
        self.add_info('cuda_toolkit_version', pycuda_driver.toolkit_version,
                      title="NVIDIA Kernel driver")
        self.add_info('number_of_devices', pycuda_driver.Device.count(),
                      title="NVIDIA Kernel driver")

        # text = str(self._cuda.nvidia_smi) + str(cuda.nvidia_smi_l)

        #
        # Global GPU Memory
        #
        free, total = pycuda_driver.mem_get_info()
        self.add_info('total_gpu_memory', total,
                      title="Total GPU memory", unit=Unit.MEMORY)
        self.add_info('free_gpu_memory', free,
                      title="Free GPU memory", unit=Unit.MEMORY,
                      part_of='total_gpu_memory')
        self.add_info('cuda_device_count', self.number_of_devices,
                      title="Number of CUDA devices")



class Py3nvmlInfo(CudaInfo):
    """A :py:class:`CudaInfo` implementation based on the third-party
    pacakge `py3nvml` which are essentially Python bindings for the
    NVIDIA `nvml` library.

    Attributes
    ----------

    _deviceCount: int
        The number of NVIDA devices.
    _handle:
        An NVML handle for the current device. None if no device
        is selected.
    """

    @staticmethod
    def initialize_nvml_module() -> Optional[ModuleType]:
        """Initialize the `pycuda` module.
        """
        try:
            py3nvml = Importer.import_module('py3nvml')
        except ModuleNotFoundError as error:
            # "No module named 'py3nvml'"
            raise ValueError("Error importing module 'py3nvml' - probably "
                             "the package is not installed.") from error
        except ImportError as error:
            raise ValueError("Error importing module 'py3nvml' - package "
                             "seems to be installed, so probably there "
                             "is some CUDA problem (no driver)") from error

        try:
            py3nvml.nvmlInit()
        except py3nvml.NVMLError_LibRmVersionMismatch:
            # "RM has detected an NVML/RM version mismatch."
            raise ValueError("Error initalizing NVML.") from error
        return py3nvml

    def _initialize_devices(self) -> None:
        """Initialize the list of device infos.
        """
        # for idx in range(NUMBER_OF_DEVICES):
        #     self._devices.append(PycudaDeviceInfo(idx))

    def initialize_info(self) -> None:
        """Register CUDA specific information.
        """
        super().initialize_info()
        py3nvml = self.nvml_module

        self.add_info("have_py3nvml", py3nvml is not None,
                      title="py3nvml available")
        if py3nvml is None:
            return  # no further information available



#
# Linux specific implementations
#

class LinuxCudaInfo(CudaInfo):
    """Linux implementation of the :py:class:`CudaInfo` API.
    """

    driver_path: Path = Path('/proc/driver/nvidia')

    @classmethod
    def have_nvidia_driver(cls) -> bool:
        """Check if the system has an NVIDIA kernel driver.
        """
        return cls.driver_path.exists()

    @classmethod
    def nvidia_driver_version(cls) -> Tuple[str, str]:
        """Get the NVIDIA kernel driver version.

        Result
        ------
        driver_version:
            The version of the NVIDIA kernel driver.
        gcc_version:
            The version of the GCC compiler used to compile the driver.
        """
        version_path = cls.driver_path / 'version'
        if not version_path.exists():
            raise ValueError("Cannot determine NVIDIA kernel driver version "
                             f"(no file '{version_path}')")

        with open(version_path, encoding='utf-8') as version_file:
            driver_info = version_file.read()
        match = re.search('Kernel Module +([^ ]*)', driver_info)
        if match:
            nvidia_version = match.group(1)
        match = re.search('gcc version +([^ ]*)', driver_info)
        if match:
            gcc_version = match.group(1)
        return nvidia_version, gcc_version

    def initialize_info(self) -> None:
        """Register CUDA specific information.
        """
        super().initialize_info()

        try:
            driver_version, compiler_version = self.nvidia_driver_version()
        except ValueError:
            driver_version, compiler_version = 'N/A', 'N/A'

        self.add_info('nvidia_driver_version', driver_version,
                      title="NVIDIA driver version")
        self.add_info('nvidia_driver_compiler', compiler_version,
                      title="Driver compiler version")

    @overload
    @classmethod
    def get_nvidia_devices(cls, pci: Literal[False] = ...) -> Iterable[int]:
        ...

    @overload
    @classmethod
    def get_nvidia_devices(cls, pci: Literal[True]) -> Iterable[str]:
        ...

    @classmethod
    def get_nvidia_devices(cls, pci: bool):
        """Iterate over the detected NVIDIA devices.

        Arguments
        ---------
        pci:
            A flag indicating if devices should be listed by a
            numerical identifier (`False`) of their PCI identifier
            (`True`).
        """
        gpus_path = cls.driver_path / 'gpus'
        if pci:
            return gpus_path.iterdir()

        return range(len(gpus_path.iterdir()))


class LinuxCudaDeviceInfo(CudaDeviceInfo):
    """Linux implementation of the `CudaDeviceInfo`.

    Arguments
    ---------
    gpu:
        A value identifying the GPU.  Either an integer identifier
        or a string, which (currently) can only be the PCI identifier.
    
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        nvidia_devices = LinuxCudaInfo.get_nvidia_devices(pci=True)
        self._pci_device = \
            next(itertools.islice(nvidia_devices, self.device_number, None))
        if not self.pci_device_path.is_dir():
            raise ValueError("No PCI device path in '/sys/bus/pci/devices' "
                             f"for GPU number {self.device_number}")

    @property
    def pci_device(self) -> str:
        """The PCI Device identifier."""
        return self._pci_device

    @property
    def pci_device_path(self) -> Path:
        """Path to the Liux PCI device directory, like
        `/sys/bus/pci/devices/...`
        """
        return Path('/sys/bus/pci/devices') / self.pci_device

    #
    # NUMA
    #

    @property
    def nvidia_numa_node_file(self) -> Path:
        """The path to the NUMA node file for the given GPU.
        """
        numa_node_file = self.pci_device_path / 'numa_node'
        if not numa_node_file.is_file():
            raise ValueError("No NUMA node information for "
                             f"GPU {self.device_number}.")
        return numa_node_file

    def check_nvidia_device(self) -> bool:
        """Check if an NVIDIA device (GPU) is configured correctly.

        Arguments
        ---------
        gpu:
           The GPU device to check.
        """
        return self.nvidia_numa_node != -1

    @property
    def nvidia_numa_node(self) -> int:
        """Get the NUMA node (CPU-memory cluster) to which a GPU is
        associated.

        A NUMA node is a cluster of CPU(s) and memory.  In most
        desktop computers there is only one CPU and hence only one
        NUMA node, but in servers there may be multiple NUMA nodes.  A
        GPU is usually associated to one such NUMA node.

        The NUMA node value is read from the special `numa_node` file
        in the device directory of the GPU device (in the SysFS
        filesystem).

        Arguments
        ---------
        gpu:
           The number of the GPU (0 for first GPU, 1 for second GPU, etc.).

        Result
        ------
        numa:
            The NUMA node to which the GPU is associated.  A value
            of `-1` means that there is no NUMA node information
            available for that device.
        """
        return int(self.nvidia_numa_node_file.read_text(encoding='utf-8'))

    @nvidia_numa_node.setter
    def nvidia_numa_node(self, numa: int) -> None:
        """Set the NUMA node value to which a GPU is associated.  This is done
        by writing that value into the respective SysFS device file.

        Usually setting this value should not be necessary, as it is
        initialized from the firmware settings during system boot.
        However, with buggy firmware, this value may be incorrect
        (reporting `-1`).  Software like TensorFlow will detect this
        mismatch and issue a message ("successful NUMA node read from
        SysFS had negative value (-1), but there must be at least one
        NUMA node, so returning NUMA node zero").  Setting the NUMA
        node to the correct value, will silence such messages.

        Remark: Setting this value requires superuser privileges, as
        it affects the system configuration (not just for the current
        user).

        Remark 2: Setting this value should be considerd a hack
        required to fix a bug in the firmware of the syste. In a sane
        system, the firmware should provide the correct value.  If the
        SysFS `numa_node` file contains the wrong NUMA node number,
        update your firmware, and if that doesn't help inform your
        mainboard manufactor on this bug.

        Arguments
        ---------
        numa:
            The number of the NUMA node to which the GPU is associated.
            A value of `-1` means that the GPU is not associated to a
            NUMA node (which usually makes no sense).
        """
        try:
            self.nvidia_numa_node_file.write_text(str(numa), encoding='utf-8')
        except PermissionError as exc:
            raise RuntimeError("Setting the NUMA node for a GPU requires "
                               "superuser priveleges. To do it manually, "
                               "run the following command: echo {numa} | "
                               f"sudo tee -a {self.nvidia_numa_node_file}") \
                 from exc


running_on_linux: bool = sys.platform == 'linux'

