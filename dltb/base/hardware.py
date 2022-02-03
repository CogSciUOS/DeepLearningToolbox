"""Hardware related code.
"""

# standard imports
from typing import Iterable, Union, overload
from pathlib import Path
import os
import re
import platform
import resource
import subprocess
import itertools

# third-party imports
try:
    from py3nvml import py3nvml
    try:
        py3nvml.nvmlInit()
    except py3nvml.NVMLError_LibRmVersionMismatch:
        # "RM has detected an NVML/RM version mismatch."
        py3nvml = None
except ModuleNotFoundError:
    # "No module named 'py3nvml'"
    py3nvml = None
except ImportError:
    # "cannot import name 'py3nvml'"
    py3nvml = None

# toolbox imports
from dltb.typing import Literal
    

class _component(dict):
    def __init__(self, name: str = None):
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

    match = re.search(r'GPU [\d]+: (.*) \(UUID', cuda.nvidia_smi_l)
    if match:
        gpus = [_component(match.group(1))]
    else:
        gpus = [_component('?')]

except FileNotFoundError:
    cuda = None
    gpus = []

except subprocess.CalledProcessError:
    # Command '['nvidia-smi']' returned non-zero exit status 231.
    cuda = None
    gpus = []


#
# processor name
#

_processor_name = '?'
if platform.system() == "Windows":
    _processor_name = platform.processor()
elif platform.system() == "Darwin":
    os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
    command = "sysctl -n machdep.cpu.brand_string"
    _processor_name = subprocess.check_output(command).strip()
elif platform.system() == "Linux":
    command = "cat /proc/cpuinfo"
    all_info = str(subprocess.check_output(command, shell=True,
                                           universal_newlines=True))
    for line in all_info.split("\n"):
        if "model name" in line:
            _processor_name = re.sub(".*model name.*:", "", line, 1)
cpus = [_component(_processor_name)]

mem = _component()


def update(initialize: bool = False):
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
        output = subprocess.check_output(["nvidia-smi", "-q"],
                                         universal_newlines=True)
        for line in output.splitlines():
            if re.match('GPU ', line):
                gpu += 1
                continue
            if gpu < 0:
                continue

            match = re.match(' *GPU Current Temp *: ([^ ]*) C', line)
            if match:
                gpus[gpu].temperature = match.group(1)
                continue
            if initialize:
                match = re.match(' *GPU Shutdown Temp *: ([^ ]*) C', line)
                if match:
                    gpus[gpu].temperature_max = match.group(1)
                    continue

            match = re.match(r'  *(\w*) Memory Usage', line)
            if match:
                mem_section = (match.group(1) == 'FB')
                continue

            if mem_section:
                match = re.match(' *Used *: ([^ ]*) MiB', line)
                if match:
                    gpus[gpu].mem = int(match.group(1))
                    continue

                if initialize:
                    match = re.match(' *Total *: ([^ ]*) MiB', line)
                    if match:
                        gpus[gpu].mem_total = int(match.group(1))
                        continue


update(True)


#
# GPU/CUDA support
#


@overload
def get_nvidia_devices(pci: Literal[False] = ...) -> Iterable[int]:
    ...


@overload
def get_nvidia_devices(pci: Literal[True]) -> Iterable[str]:
    ...


def get_nvidia_devices(pci: bool):
    """Iterate over the detected NVIDIA devices.

    Arguments
    ---------
    pci:
        A flag indicating if devices should be listed by a
        numerical identifier (`False`) of their PCI identifier
        (`True`).
    """
    if pci:
        return os.listdir('/proc/driver/nvidia/gpus/')

    # FIXME[hack]
    return range(len(os.listdir('/proc/driver/nvidia/gpus/')))


def _linux_nvidia_numa_node_file(gpu: Union[int, str]) -> Path:
    pci_device = linux_nvidia_pci_device(gpu)
    numa_node_file = Path(f'/sys/bus/pci/devices/{pci_device}/numa_node')
    if not numa_node_file.is_file():
        raise ValueError(f"No NUMA node information for gpu {gpu}.")
    return numa_node_file

def linux_nvidia_pci_device(gpu: Union[int, str]) -> str:
    """Get the Linux PCI identifier for the given NVIDIA GPU.

    Arguments
    ---------
    gpu:
        A value identifying the GPU.  Either an integer identifier
        or a string, which (currently) can only be the PCI identifier.

    Result
    ------
    pci:
        The PCI identifier.
    """
    if isinstance(gpu, str):
        return gpu
    return next(itertools.islice(get_nvidia_devices(True), gpu, None))


def linux_get_gpu_numa_node(gpu: Union[int, str]) -> int:
    """Get the NUMA node (CPU-memory cluster) to which a GPU is associated.

    A NUMA node is a cluster of CPU(s) and memory.  In most desktop
    computers there is only one CPU and hence only one NUMA node,
    but in servers there may be multiple NUMA nodes.  A GPU is usually
    associated to one such NUMA node.

    The NUMA node value is read from the special `numa_node` file in
    the device directory of the GPU device (in the SysFS filesystem).

    Arguments
    ---------
    gpu:
        The number of the GPU (0 for first GPU, 1 for second GPU, etc.).

    Result
    ------
    numa:
        The NUMA node to which the GPU is associated.  A value of `-1` means
        that there is no NUMA node information available for that device.
    """
    numa_node_file = _linux_nvidia_numa_node_file(gpu)
    return int(numa_node_file.read_text())


def linux_set_gpu_numa_node(gpu: Union[int, str], numa: int) -> None:
    """Set the NUMA node value to which a GPU is associated.
    This is done by writing that value into the respective SysFS
    device file.

    Usually setting this value should not be necessary, as it is
    initialized from the firmware settings during system boot.
    However, with buggy firmware, this value may be incorrect
    (reporting `-1`).  Software like TensorFlow will detect this
    mismatch and issue a message ("successful NUMA node read from
    SysFS had negative value (-1), but there must be at least one NUMA
    node, so returning NUMA node zero").  Setting the NUMA node to the
    correct value, will silence such messages.

    Remark: Setting this value requires superuser privileges, as it
    affects the system configuration (not just for the current user).

    Remark 2: Setting this value should be considerd a hack required to
    fix a bug in the firmware of the syste. In a sane system, the firmware
    should provide the correct value.  If the SysFS `numa_node` file
    contains the wrong NUMA node number, update your firmware, and if
    that doesn't help inform your mainboard manufactor on this bug.

    Arguments
    ---------
    gpu:
        The GPU device to check.
    numa:
        The number of the NUMA node to which the GPU is associated.
        A value of `-1` means that the GPU is not associated to a
        NUMA node (which usually makes no sense).
    """
    numa_node_file = _linux_nvidia_numa_node_file(gpu)
    try:
        numa_node_file.write_text(str(numa))
    except PermissionError:
        raise RuntimeError("Setting the NUMA node for a GPU requires "
                           "superuser priveleges. To do it manually, "
                           "run the following command: "
                           f"echo {numa} | sudo tee -a {numa_node_file}")


def check_nvidia_device(gpu: Union[int, str]) -> bool:
    """Check if an NVIDIA device (GPU) is configured correctly.

    Arguments
    ---------
    gpu:
        The GPU device to check.
    """
    return linux_get_gpu_numa_node(gpu) != -1
