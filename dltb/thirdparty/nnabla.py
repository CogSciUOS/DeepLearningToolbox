"""The Sony Neural Network Libraries (nnabla) is a deep learning
framework intended to be used in research, development, and
production.  It consist of a core package (`nnabla`, [1]) and several
extensions like CUDA extensions (`nabla-ext-cuda` [2]), examples [3],
a C-runtime library for inference neural networks [4], a hardware-aware
neural architecture search [5], and a Windows GUI app [6].

Installation
------------

The core package `nnabla` can be installed directly with pip.

  pip install nnabla

Installing the CUDA extension package `nnabla-ext-cuda` requires
knowing the version of the CUDA toolkit installed (install, for example,
pip package `nnabla-ext-cuda101` for CUDA toolkit 10.1).

  pip install nnabla-ext-cuda101

There is no pip package for `nabla-examples`. For running the
examples, the `nnabla-examples` repository [3] should be cloned.

  git clone https://github.com/sony/nnabla-examples.git

That repository is split into several subdirectories, many with
specific requirements specified by a `requirements.txt` file. These
can be install by `pip -r example_subdirectory/requirements.txt`.

The package `nnabla` has several third-party dependencies, like
* `boto3` and `s3transfer`:
  libraries that allow access to Amazon Web Services (AWS)
* `onnx`:
  Open Neural Network Exchange


Examples
--------



References
----------

[1] https://github.com/sony/nnabla/
[2] https://github.com/sony/nnabla-ext-cuda
[3] https://github.com/sony/nnabla-examples
[4] https://github.com/sony/nnabla-c-runtime
[5] https://github.com/sony/nnabla-nas
[6] https://dl.sony.com/

"""

# standard imports
import os
import sys
import logging

# third-party imports
import numpy as np

# FIXME[hack]: importing nnabla.logging (automatically imported by nnabla)
# will set the global log level to
# nnabla_config.get('LOG', 'log_console_level')
# which is set to 'INFO' in the default 'nnabla.conf' configuration
# file.
logging.basicConfig(level=logging.WARNING)

import nnabla as nn
from nnabla.ext_utils import get_extension_context
# nnabla_ext_name = "nnabla-ext-cuda{cudatoolkit_version_number}"
# !pip install nnabla-ext-cuda100
# !pip install nnabla-ext-cuda101
import nnabla_ext.cuda

# Toolbox imports
from ..config import config
from ..base import Preparable
from ..base.busy import BusyObservable, busy
from ..base.image import Image
from ..tool.generator import ImageGAN
from ..util.download import download

# Logging
LOG = logging.getLogger(__name__)


# Configuration
config.add_property('nnabla_examples_directory',
                    default=lambda c: c.github_directory / 'nnabla-examples',
                    description="Path to a directory into which the "
                    "`nnabla-examples` repository "
                    "(https://github.com/sony/nnabla-examples.git) "
                    "was cloned.")


class NNabla(Preparable, BusyObservable):
    """Abstract base class for networks realized in `nnabla`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._nnabla_extension_context = None

    def _prepared(self) -> bool:
        return (self._nnabla_extension_context is not None and
                super()._prepared())

    def _prepare(self) -> None:
        super()._prepare()

        # init gpu
        # FIXME[todo]: make this conditional
        self._nnabla_extension_context = get_extension_context("cudnn")
        nn.set_default_context(self._nnabla_extension_context)

    @staticmethod
    def convert_images_to_uint8(images, drange=[-1, 1]) -> np.ndarray:
        """
        convert float32 -> uint8
        """
        # to numpy.ndarray
        if isinstance(images, nn.Variable):
            images = images.d

        # addapt range from [drange[0], drange[1]] to [0, 255]
        scale = 255 / (drange[1] - drange[0])
        images = images * scale + (0.5 - drange[0] * scale)

        # to uint8
        images = np.uint8(np.clip(images, 0, 255))

        # (batch, channel, height, width) -> (batch, height, width, channel)
        images = images.transpose(0, 2, 3, 1)

        return images


class NNablaExample(NNabla):

    # FIXME[hack]: we need a better info API
    def info(self):
        print(f"NNabla example directory: {config.nnabla_examples_directory}")


class StyleGAN2(ImageGAN, NNablaExample):
    """
    Ths StyleGAN2 demo directory (`nnabla-examples/GANs/stylegan2/`)
    contains the following files:

    `generate.py`:
        Defines the functions `generate` and `synthesis` (and `main`).
        The function `synthesis` defines a convolutional generator,
        mapping a latent vector w through a chain of activation maps,
        starting from (*, 512, 4, 4) -> (*, 512, 8, 8) ->
        (*, 512, 16, 16) -> (*, 512, 32, 32) -> (*, 512, 64, 64) ->
        (*, 256, 128, 128) -> (*, 128, 256, 256) -> (*, 64, 512, 512) ->
        (*, 32, 32, 512).
    
    `networks.py`:
        Defines `mapping_network`

    `ops.py`:
        Several utility functions:

    The nnabla StyleGAN2 demo seems to only contain the generator, no
    discriminator.
    """
    # Import functions from the example script
    # {nnabla_examples}/GANs/stylegan2/generate.py

    # version trouble: there are some changes to the repository,
    # that make it hard to use this consistently:
    #
    # * git checkout d646fd9cc58f0914db0f5c5a4043c4756209f9d4:
    #   - no relative imports
    #     -> import directly from the 'GANs/stylegan2/' directory,
    #        not from 'GANs/' directory
    #   - moderate use of the 'inplace' argument
    #     -> does work with nnabla-1.7.0
    #
    # * newer versions

    # FIXME[hack]: some proper error handling is needed here!
    # (notice that this code is executed during import)
    assert os.path.isdir(config.nnabla_examples_directory)
    nnabla_stylegan2_directory = \
        os.path.join(config.nnabla_examples_directory, 'GANs', 'stylegan2')
    #    os.path.join(config.nnabla_examples_directory, 'GANs')
    if nnabla_stylegan2_directory not in sys.path:
        sys.path.insert(0, nnabla_stylegan2_directory)

    LOG.debug("Importing functions 'synthesis' and 'generate' from "
              "'%s/generate.py'", nnabla_stylegan2_directory)

    # FIXME[problem]: this will place a module 'generate' in sys.modules
    # FIXME[bug]: with "Python 3.6.10 :: Anaconda, Inc."
    #   ImportError: attempted relative import with no known parent package
    import generate as _nnabla_stylegan2
    # from stylegan2 import generate as _nnabla_stylegan2

    sys.path.remove(nnabla_stylegan2_directory)

    def __init__(self, model: str = None,
                 filename: str = None, **kwargs) -> None:
        # FIXME[hack]: model argument is not needed, but provided
        # by demos/dl-gan.py invocation. Repair
        super().__init__(**kwargs)
        self.weights_file = filename or \
            os.path.join(config.model_directory, 'styleGAN2_G_params.h5')
        self._rgb_output = None

    def _prepared(self) -> bool:
        return self._rgb_output is not None and super()._prepared()

    def _prepare(self) -> None:
        super()._prepare()
        LOG.info("prepare: starting nnabla stylegan2 preparation")
        self._prepare_model()
        self._prepare_nnabla_generator()
        # self._prepare_nnabla_generator2()
        LOG.info("prepare: finished nnabla stylegan2 preparation")

    def _prepare_model(self) -> None:
        #
        # Load the model data
        #

        # make sure the weights file is available
        if not os.path.isfile(self.weights_file):
            url = ('https://nnabla.org/pretrained-models/nnabla-examples/'
                   'GANs/stylegan2/styleGAN2_G_params.h5')
            LOG.debug("prepare: downloading model data: %s", url)
            download(url, self.weights_file)

        # load the weights file
        LOG.debug("prepare: loading the weights: %s", self.weights_file)
        nn.load_parameters(self.weights_file)

    def _prepare_nnabla_generator(self) -> None:
        #
        # Prepare computations
        #

        # The number of images to generate
        batch_size: int = 1

        # The seed for noise input **z**. (This drastically changes
        # the result)
        latent_seed: int = 300  # min: 0, max: 1000

        # The value for truncation trick.
        truncation_psi: float = 0.32  # min: 0.0, max: 1.0
        # truncation_psi: float = 0.5

        # The seed for stochasticity input.  (This slightly changes
        # the result)
        noise_seed: int = 500  # min: 0, max: 1000

        # The seed for stochasticity input.  Number of
        # layers to inject noise (a.k.a stochastic variations)
        # into. *This seems to change very little. Default: 18*
        num_layers: int = 18  # min: 0, max: 500

        # FIXME[hack]: find a way to determine the number of feature
        # dimensions from the network
        self._feature_dimensions = 512

        # rnd = np.random.RandomState(latent_seed)
        # rnd.randn(batch_size, self._feature_dimensions)
        z = self.random_features(seed=latent_seed)

        self._style_noises = [nn.Variable((batch_size, 512)).apply(d=z)
                              for _ in range(num_layers)]

        # Create a new nnabla._variable.Variable called rgb_output
        #
        # The function self._nnabla_stylegan2.generate performs the
        # following steps:
        #   1. obtain a normalized version of self._style_noises
        #   2. get the actual latent code "w" by sending the
        #      normalized style noise through the mapping_network
        #      (defined in self._nnabla_stylegan2.networks)
        #   3. apply the truncation trick
        #   4. define some constant
        #   5. create the synthesis graph using the function
        #      self._nnabla_stylegan2.synthesis
        self._rgb_output = \
            self._nnabla_stylegan2.generate(batch_size, self._style_noises,
                                            noise_seed, truncation_psi)

    #@busy("generating")
    #def generate(self, **kwargs) -> Image:
    #    return super().generate(**kwargs)

    def _generate_single(self, features: np.ndarray) -> np.ndarray:
        """
        """
        for style_noise in self._style_noises:
            style_noise.d = features[np.newaxis, ...]

        # rgb_output.forward()  # RuntimeError: memory error in alloc
        self._rgb_output.forward(clear_buffer=True)
        # nnabla._variable.Variable
        # shape: (1, 3, 1024, 1024) as (BATCH, CHANNEL, HEIGHT, WIDTH)
        # min: -0.99, max: 0.91

        return self.convert_images_to_uint8(self._rgb_output, drange=[-1, 1])

    def mix(self, seed1: int = 300, seed2: int = 444,
            mix: int = 7) -> np.ndarray:
        """Mix the image from two seeds.

        Arguments
        ---------
        seed1:
            Seed for the first random feature vector.
        seed2:
            Seed for the second random feature vector.
        mix:
            The index of the layer where to mix in the second features.
        """

        if not 0 <= mix <= len(self._style_noises):
            raise ValueError(f"Invalid mix index {mix}, should be between 0 "
                             f"and {len(self._style_noises)}")

        z1 = self.random_features(seed=seed1)
        z2 = self.random_features(seed=seed2)

        for index, style_noise in enumerate(self._style_noises):
            style_noise.d = z1 if index < mix else z2

        self._rgb_output.forward(clear_buffer=True)
        images = self.convert_images_to_uint8(self._rgb_output, drange=[-1, 1])
        return images[0]

    def _data_shape(self):
        # swap color channel
        #return (self._generator.output_shape[2:] +
        #        self._generator.output_shape[1:2])
        return (1024, 1024, 3)  # FIXME[hack]

    def info(self) -> None:
        print(f"Info: {self} ({type(self)})")
        super().info()
