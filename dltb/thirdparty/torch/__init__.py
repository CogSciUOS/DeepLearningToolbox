"""Integration of the torch library. Torch provides its
own array types, similar to numpy.

This module is to be included when the `torch` package is loaded.

This module adds some hooks to work with torch:

* adapt :py:func:`dltb.base.image.Data` to allow transformation of
  `Datalike` objects to :py:class:`torch.Tensor` (as_torch) as well
  as and from :py:class:`PIL.Image.Image` to other formats.

* add a data kind 'torch' and an associated loader to the
  :py:class:`Datasource` class.

"""

# standard imports
import os
import sys
import logging

# third party imports
import numpy as np
import torch

# toolbox imports
from ...base.info import Info, InfoSource
from ...base.data import Data, Datalike
from ...base.package import Package
from ...util.importer import Importer
from ...datasource import Datasource

# logging
LOG = logging.getLogger(__name__)


def as_torch(data: Datalike, copy: bool = False) -> torch.Tensor:
    """Get a :py:class:`torch.Tensor` from a :py:class:`Datalike`
    object.
    """
    if isinstance(data, torch.Tensor):
        return data.clone().detach() if copy else data

    if isinstance(data, str) and data.endswith('.pt'):
        return torch.load(data)

    if isinstance(data, Data):
        if not hasattr(data, 'torch'):
            data.add_attribute('torch', Data.as_torch(data.array, copy=copy))
        return data.torch

    if not isinstance(data, np.ndarray):
        data = Data.as_array(data)

    # from_numpy() will use the same data as the numpy array, that
    # is changing the torch.Tensor will also change the numpy.ndarray.
    # On the other hand, torch.tensor() will always copy the data.

    # pylint: disable=not-callable
    # torch bug: https://github.com/pytorch/pytorch/issues/24807
    return torch.tensor(data) if copy else torch.from_numpy(data)


LOG.info("Adapting dltb.base.data.Data: adding static method 'as_torch'")
Data.as_torch = staticmethod(as_torch)

# add a loader for torch data: typical suffix is '.pt' (pytorch)
Datasource.add_loader('torch', torch.load)


class TorchPackage(Package):
    """An extended :py:class:`Package` for providing specific Torch
    information.

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(key='torch', **kwargs)

    def initialize_info(self) -> None:
        """Register torch specific information.
        """
        super().initialize_info()

        # for some CUDA related information, torch relies on `pynvml`,
        # which is independent of torch and may not be available.
        pynvml_installed = Importer.importable('pynvml')

        cuda_support = torch.backends.cuda.is_built()
        self.add_info('cuda_support', torch.backends.cuda.is_built(),
                      title='CUDA backend is built')

        if cuda_support:
            cuda_available = torch.cuda.is_available()
            self.add_info('cuda_available', torch.cuda.is_available(),
                          title='CUDA available:')
            # self.add_info('cuda_backend',
            #               torch.backends.cuda.preferred_linalg_library(),
            #               title='CUDA Linalg backend:')
        else:
            cuda_available = False

        if cuda_available:
            self.add_info('cuda_initialized', torch.cuda.is_initialized,
                          title='CUDA Initialized')
            self.add_info('cuda_version', torch.version.cuda,
                          title='CUDA version')
            self.add_info('cuda_arch_list', torch.cuda.get_arch_list(),
                          title='CUDA architectures')
            if pynvml_installed:
                # torch.cuda.memory_usage():
                #   the percent of time over the past sample period during
                #   which global (device) memory was being read or written.
                #   as given by nvidia-smi.
                self.add_info('cuda_memory', torch.cuda.memory_usage,
                              title='Global CUDA memory usage')
                # torch.cuda.list_gpu_processes():
                #   a human-readable printout of the running processes and
                #   their GPU memory use for a given device.
                self.add_info('cuda_utilization', torch.cuda.utilization,
                              title='Global CUDA memory usage')
            # else: [no CUDA information as pynvml is not installed]

        if torch is not None:
            self.add_info('cudnn_available',
                          torch.backends.cudnn.is_available(),
                          title='CuDNN backend available')
            if torch.backends.cudnn.is_available():
                self.add_info('cudnn_enabled', torch.backends.cudnn.enabled,
                              title='CuDNN backend enabled')
                self.add_info('cudnn_version',
                              torch.backends.cudnn.version(),
                              title='CuDNN backend version')               

            self.add_info('mps_available', torch.backends.mps.is_available(),
                          title='MPS backend available')
            
            self.add_info('mkl_available', torch.backends.mkl.is_available(),
                          title='MKL backend available')
            
            self.add_info('openmp_available',
                          torch.backends.openmp.is_available(),
                          title='OpenMP backend available')
        #
        # Device information
        #
        if cuda_available:
            # GPU device information
            self.add_info('gpu_count', torch.cuda.device_count(),
                          title="Number of GPUs")
            self.add_info('gpu_current', torch.cuda.current_device(),
                          title="Current GPU")
            for idx in range(torch.cuda.device_count()):
                self.add_info(f'gpu{idx}_name',
                              torch.cuda.get_device_name(idx),
                              title=f"GPU-{idx} name")
                self.add_info(f'gpu{idx}_properties',
                              torch.cuda.get_device_properties(idx),
                              title=f"GPU-{idx} details")

                # the global free and total GPU memory occupied for
                # a given device using cudaMemGetInfo:
                # - mem_info[0]: the free GPU memory on the device, that
                #                is the memory that has not been allocated
                #                by any process (torch and others). This will
                #                always be less or equal to total GPU memory
                #                reported by mem_info[1].
                # - mem_info[1]: the total GPU device memory
                #                (allocated and free).
                #                That is, on a 4GB card, this will report 4GB.
                mem_info = torch.cuda.mem_get_info(idx)
                self.add_info(f'gpu{idx}_mem_free', mem_info[0],
                              title=f"GPU-{idx} mem free")
                self.add_info(f'gpu{idx}_mem_total', mem_info[1],
                              title=f"GPU-{idx} mem total")
                # f"{sizeof_fmt(mem_info[0])}/{sizeof_fmt(mem_info[1])}"
                # f" ({int(mem_info[0]/mem_info[1]*100)}%)")

        #
        # Environment
        #

        # TORCH_HOME:
        #    if set, several subdirectories will be created:
        #      checkpoints/
        #      hub/
        #         checkpoints/
        #         pytorch_vision_v0.6.0/
        #         torchaudio/
        self.add_info('env_torch_home',
                      os.environ.get('TORCH_HOME', '*undefined*'),
                      title="TORCH_HOME")
        self.add_info('env_xdg_cache_home',
                      os.environ.get('XDG_CACHE_HOME', '*undefined*'),
                      title="XDG_CACHE_HOME")
        # default for XDG_CACHE_HOME is '~/.cache'

        #
        # Directories
        # 

        # torch.hub: the location can be changed as follows:
        #  1. Calling hub.set_dir(<PATH_TO_HUB_DIR>)
        #  2. $TORCH_HOME/hub, if environment variable TORCH_HOME is set.
        #  3. $XDG_CACHE_HOME/torch/hub, if environment variable XDG_CACHE_HOME
        #     is set.
        #  4. ~/.cache/torch/hub
        if torch is not None:
            self.add_info('dir_torch_hub', torch.hub.get_dir(),
                          title="torch.hub")

TorchPackage()
