"""Collect information on the torch installation.
"""
from typing import Optional
from importlib import import_module
import sys


def get_module(name: str, load: bool = True) -> Optional:
    print(f"Module '{name}': ", end='')
    mod = sys.modules.get(name, None)
    if mod is None and load:
       try:
           mod = import_module(name)
       except ImportError as ex:
           print("*not installed*")
           return None
       except OSError as ex:
           print("*installation error*")
           return None
    if mod is None:
        print("*not installed*")
    else:
        print(mod.__version__)
    return mod


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def info_torch(load: bool = True) -> None:
    """Output information on the Torchvision module.
    """
    torch = get_module('torch', load=load)
    if torch is None:
       False

    if torch.cuda.is_available():
        print("* compiled with CUDA Support:")
        print("  - CUDA version:", torch.version.cuda)
        print("  - initialized:", torch.cuda.is_initialized())
        print("  - arch list:", torch.cuda.get_arch_list())
        print("  - device count:", torch.cuda.device_count())
        print("  - current device:", torch.cuda.current_device(), torch.cuda.get_device_name())
        have_pynvml = True
        if have_pynvml:
            print("  - global GPU memory usage:", torch.cuda.memory_usage())
            print("  - global GPU utilization:", torch.cuda.utilization())
        print(torch.cuda.get_device_properties(torch.cuda.current_device()))
    else:
        print("* compiled without CUDA Support")

    if torch.backends.cuda.is_built():
        print("* CUDA backend is built:")
        print("  - preferred linalg backend:", torch.backends.cuda.preferred_linalg_library())
    else:
        print("* CUDA backend is not built")

    if torch.backends.cudnn.is_available():
        print("* CuDNN backend is available:")
        print("  - cudnn version:", torch.backends.cudnn.version())
        print("  - cudnn enabled:", torch.backends.cudnn.enabled)
    else:
        print("* CuDNN backend is available")


    if torch.backends.mps.is_available():
        print("* MPS backend is available")
    else:
        print("* MPS backend is not available")

    if torch.backends.mkl.is_available():
        print("* MKL backend is available")
    else:
        print("* MKL backend is not available")

    if torch.backends.openmp.is_available():
        print("* OpenMP backend is available")
    else:
        print("* OpenMP backend is not available")

    #
    # Device information
    #
    print("GPU device infos:")
    # torch.cuda.list_gpu_processes:
    #   a human-readable printout of the running processes and their
    #   GPU memory use for a given device.
    print(" - GPU processes:", torch.cuda.list_gpu_processes())

    # the global free and total GPU memory occupied for a given device
    # using cudaMemGetInfo:
    mem_info = torch.cuda.mem_get_info()
    print(" - memory information: "
          f"{sizeof_fmt(mem_info[0])}/{sizeof_fmt(mem_info[1])}"
          f" ({int(mem_info[0]/mem_info[1]*100)}%)")
    # torch.cuda.memory_usage:
    #   the percent of time over the past sample period during which
    #   global (device) memory was being read or written. as given by
    #   nvidia-smi.
    print(" - active memory usage: "
          f"{torch.cuda.memory_usage()}")


def info_torchvision(load: bool = True) -> None:
    """Output information on the Torchvision module.
    """
    torchvision = get_module('torchvision', load=load)


def info_torchaudio(load: bool = True) -> None:
    """Output information on the Torchaudio module.
    """
    torchvision = get_module('torchaudio', load=load)


def info_torchtext(load: bool = True) -> None:
    """Output information on the Torchtext module.
    """
    torchvision = get_module('torchtext', load=load)


def info_accimage(load: bool = True) -> None:
    """Output information on the accimage module.
    """
    accimage = get_module('accimage', load=load)
    

def info_pynvml(load: bool = True) -> None:
    """Output information on the accimage module.
    """
    pynvml = get_module('pynvml', load=load)


if __name__ == '__main__':
    info_torch()
    info_torchvision()
    info_torchaudio()
    info_torchtext()
    # info_accimage()
    info_pynvml()
    import time
    time.sleep(10)
