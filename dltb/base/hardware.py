"""Hardware related code.
"""

# standard imports
import os
import re
import platform
import resource
import subprocess

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
            elif gpu < 0:
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
