#
# Hardware/CUDA stuff
#

class _component(dict):
    def __init__(self, name = None):
        if name is not None:
            self['name'] = name

    def __getattr__(self, key):
        if key in self:
            return self[key]

    def __setattr__(self, key, value):
        self[key] = value

import subprocess
import re

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
