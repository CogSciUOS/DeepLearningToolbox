"""A StyleGAN2-ADA demonstration.

from dltb.thirdparty.nvlabs.stylegan2ada import StyleGAN2Ada
gan = StyleGAN2Ada()
gan.prepare()

from dltb.util.image import imshow
for i in range(5):
    img, d = gan.test()
    imshow(img, title=f"d={d}")


[1] https://github.com/NVlabs/stylegan2-ada-pytorch
"""

# standard imports
import sys
import logging

# third-party imports
import numpy as np
import torch

# Toolbox imports
from ...base import Preparable
from ...config import config
from ...tool.generator import ImageGAN

# logging
LOG = logging.getLogger(__name__)

# Configuration
config.add_property('nvlabs_stylegan2ada_pytorch_directory',
                    default=lambda c: c.github_directory /
                    'stylegan2-ada-pytorch')


class StyleGAN2Ada(ImageGAN, Preparable):

    stylegan_dir = config.nvlabs_stylegan2ada_pytorch_directory
    stylegan_github = 'https://github.com/NVlabs/stylegan.git'

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._generator = None
        self._discriminator = None
        self._device = None
        self._rnd = None

    def _prepared(self) -> bool:
        return self._generator is not None and super()._prepared()

    def _prepare(self) -> None:
        """Preparing :py:class:`StyleGAN` includes importing the
        required modules and loading model data.
        """
        super()._prepare()

        # Step 1: Import modules from stylegan repository
        print("importing stylegan2ada from "
              f"'{config.nvlabs_stylegan2ada_pytorch_directory}'")
        nvlabs_directory = str(config.nvlabs_stylegan2ada_pytorch_directory)
        if nvlabs_directory not in sys.path:
            sys.path.insert(0, nvlabs_directory)

        global dnnlib
        import legacy  # This requires torch 1.7.1
        import dnnlib

        sys.path.remove(nvlabs_directory)

        self._device = torch.device('cuda')
        network_pkl = \
            str(config.nvlabs_stylegan2ada_pytorch_directory / "metfaces.pkl")

        with dnnlib.util.open_url(network_pkl) as f:
            network_pkl = legacy.load_network_pkl(f)

        print(list(network_pkl))
        # ['G', 'D', 'G_ema', 'training_set_kwargs', 'augment_pipe']
        self._generator = network_pkl['G_ema'].to(self._device)
        self._discriminator = network_pkl['D'].to(self._device)

    def test(self, seed: int = 0):
        label = torch.zeros([1, self._generator.c_dim], device=self._device)
        truncation_psi = 1.
        noise_mode = 'const'

        if self._rnd is None or seed:
            self._rnd = np.random.RandomState(seed)
        z = self._rnd.randn(1, self._generator.z_dim)
        z = torch.from_numpy(z).to(self._device)
        img = self._generator(z, label, truncation_psi=truncation_psi,
                              noise_mode=noise_mode)
        d = self._discriminator(img, label)
        img = img.permute(0, 2, 3, 1)
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        first_img = img[0].cpu().numpy()

        return first_img, float(d)
