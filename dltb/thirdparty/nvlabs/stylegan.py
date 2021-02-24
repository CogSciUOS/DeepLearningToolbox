"""A StyleGAN1 demonstration. Code for StyleGAN2 is also available in
the face recognition seminar notebooks.

[1] https://github.com/NVlabs/stylegan
"""

# standard imports
from typing import Iterable, Union
import os
import sys
import pickle
import logging

# third-party imports
import numpy as np

# Toolbox imports
from ...config import config
from ...tool.generator import ImageGAN
from ..tensorflow.v1 import tensorflow as tf
from ..tensorflow.keras import KerasTensorflowModel

# logging
LOG = logging.getLogger(__name__)


class StyleGAN(ImageGAN, KerasTensorflowModel):
    """A StyleGAN1 demonstration. This class is based on the NVIDIA labs
    StyleGAN demo [1].


    Implementation
    --------------
    The StyleGAN repository provides a module called `dnnlib`. That module
    is structured in three submodules: `dnnlib.tflib`, `dnnlib.util`, and
    `dnnlib.submission`.

    The `dnnlib.util` package defines two helper classes,  the `EasyDict`
    (a dictionary allowing attribute syntax), and a `Logger`,
    as well as several function.

    For accessing the pretrained StyleGAN model, only the `dnnlib.tflib`
    interface is used: the function `dnnlib.tflib.init_tf` can be
    used to initialize TensorFlow.

    Pretrained models
    -----------------
    There are five pretrained models available:

    Each model is provided as Python 3 pickle file (which can simply
    be loaded with `pickle.load`), containing a triple
    (generator_snapshot, discriminator_snapshot, generator_stabil).

    References
    ----------
    [1] https://github.com/NVlabs/stylegan
    """
    config.set_default_value('nvlabs_stylegan_directory',
                             os.path.join(config.github_directory,
                                          'stylegan'))

    stylegan_dir = config.nvlabs_stylegan_directory
    stylegan_github = 'https://github.com/NVlabs/stylegan.git'

    # FIXME[hack]: we need a better import mechanism ...
    # stylegan_directory:
    #    path to a directory into which the NVlabs 'StyleGAN' repository
    #    (https://github.com/NVlabs/stylegan.git) was cloned.
    config.set_default_value('stylegan_directory',
                             os.path.join(config.github_directory, 'stylegan'))
    print(f"Stylegan directory: {config.stylegan_directory}")
    # assert os.path.isdir(stylegan_directory)

    # Prepare import of stylegan package
    if not os.path.isdir(stylegan_dir):
        LOG.error("stylegan directory not found in '%s'", stylegan_dir)
        LOG.info("install it by typing: git clone %s %s",
                 stylegan_github, stylegan_dir)
        sys.exit(1)  # FIXME[hack]: do not exit - find a better exception handling mechanism

    model_dir = '/space/home/ulf/share/tensorflow'

    # URLs for downloading model files (filesize is roughly 300M per file).
    model_urls = {
        'karras2019stylegan-ffhq-1024x1024.pkl':
        'https://myshare.uni-osnabrueck.de/f/148ad26e871243cda4d2/?dl=1',
        'karras2019stylegan-celebahq-1024x1024.pkl':
        'https://myshare.uni-osnabrueck.de/f/52b1da1fbf044834bfa1/?dl=1'
    }

    models = {
        'bedrooms': 'karras2019stylegan-bedrooms-256x256.pkl',
        'cars': 'karras2019stylegan-cars-512x384.pkl',
        'cats': 'karras2019stylegan-cats-256x256.pkl',
        'celebahq': 'karras2019stylegan-celebahq-1024x1024.pkl',
        'ffhq': 'karras2019stylegan-ffhq-1024x1024.pkl'
    }

    def __init__(self, model: str = None, filename: str = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if (model or filename) is None:
            model = 'ffhq'  # default

        if filename is not None:
            self._model = model
            self._filename = filename
        elif model is not None:
            self._model = model
            self._filename = self.models[model]

        self._generator = None
        self._generator_snapshot = None
        self._discriminator_snapshot = None
        self._generator_stabil = None

    @property
    def model(self) -> str:
        return self._model

    def _prepared(self) -> bool:
        """The :py:class:`StyleGAN` preparation is finished by assigning
        the `generator` property.
        """
        return (self._generator is not None) and super()._prepared()

    def _prepare(self) -> None:
        """Preparing :py:class:`StyleGAN` includes importing the
        required modules and loading model data.
        """
        # Step 0: Initialize TensorFlow (version 1)
        # -------
        # Implemented in super class.
        super()._prepare()

        # Step 1: Import modules from stylegan repository
        # import dnnlib
        if config.stylegan_dir not in sys.path:
            sys.path.insert(0, config.stylegan_dir)

        global tflib
        import dnnlib.tflib as tflib

        sys.path.remove(config.stylegan2_directory)

        # The config module from the stylegan repository defines the
        # following values:
        #   config.result_dir = 'results'
        #   config.data_dir = 'datasets'
        #   config.cache_dir = 'cache'
        #   config.run_dir_ignore = ['results', 'datasets', 'cache']
        #
        # import config

        # Step 2: Inititialize Tensorflow  (version 2)
        #
        # The function `tflib.tfutil.init_tf` will just initialize the
        # tensorflow default session, using the options given as argument
        # to that function: This function allows to pass one argument:
        #   config_dict: dict
        # Following options may be set:
        #   cfg["gpu_options.allow_growth"] = True
        #    - False = Allocate all GPU memory at the beginning.
        #    - True = Allocate only as much GPU memory as needed.
        #
        # Note: init_tf will check if there is already a default session
        # set up (that is tf.get_default_session() is not None) and will
        # then use that session.
        #
        # Note 2: as the we will initialize a `KerasTensorflowModel` class
        # utilize a specific Session (and Graph) for each model, there
        # is no need to use the default session mechanism and hence
        # it will be skipped here:
        run_tflib_init_tf = False
        if run_tflib_init_tf:
            LOG.info("default session already initialized: %s",
                     tf.get_default_session() is not None)
            tflib.init_tf({
                'gpu_options.allow_growth': False
            })

        # Step 3: Load the network
        #
        # The graph should be loaded into a graph/session, hence
        # `_prepare_network` is run in an appropriate context:
        self.run_tensorflow(self._prepare_model)

    def _prepare_model(self) -> None:
        """Prepare the model by loading the data and initialize
        the computation.
        """
        # Step 1: Loading model data
        # -------
        # Model data are loaded from a pickle file. Such a file is
        # expected to contain three models:
        # 1. generator_snapshot: Instantaneous snapshot of the generator.
        #    Mainly useful for resuming a previous training run.
        # 2. discriminator_snapshot: Instantaneous snapshot of the
        #    discriminator. Mainly useful for resuming a previous
        #    training run.
        # 3. generator_stabil: Long-term average of the generator.
        #    Yields higher-quality results than the instantaneous snapshot.
        #
        # Note: TensorFlow (2.2.0) emits the following warning:
        #    calling BaseResourceVariable.__init__
        #    (from tensorflow.python.ops.resource_variable_ops)
        #    with constraint is deprecated and will be removed in
        #    a future version.
        #
        #    Instructions for updating:
        #    If using Keras pass *_constraint arguments to layers.
        filename = os.path.join(self.model_dir, self._filename)
        LOG.info("Initializing StyleGAN models from '%s'", filename)
        with open(filename, 'rb') as file:
            self._generator_snapshot, self._discriminator_snapshot, \
                self._generator_stabil = pickle.load(file)
        print("stylegan._prepare_model: "
              f"generator_snapshot: {type(self._generator_snapshot)}")
        print("stylegan._prepare_model: "
              f"discriminator_snapshot: {type(self._discriminator_snapshot)}")
        print("stylegan._prepare_model: "
              f"generator_stabil: {type(self._generator_stabil)}")

        # Select the generator
        generator = self._generator_stabil
        self._feature_dimensions = generator.input_shape[1]

        # Step 2: initialize the generator
        # -------
        # run generator for the first time to finish preparation
        features = self.random_features(batch=1, seed=0)
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        LOG.debug("Warm-up generator by running on dummy input of shape %s.",
                  features.shape)
        result = generator.run(features, None,
                               truncation_psi=0.7, randomize_noise=True,
                               output_transform=fmt)
        LOG.debug("Done running generator with output shape %s",
                  result.shape)

        # Step 3: finish preparation
        # -------
        # Providing the new generator as property finishes the preparation.
        # The model is now considered `prepared`.
        self._generator = generator

    def _unprepare(self) -> None:
        self._generator = None
        self._feature_dimensions = None
        self._generator_stabil = None
        self._generator_snapshot = None
        self._discriminator_snapshot = None
        super()._unprepare()

    def _generate_batch(self, features: np.ndarray) -> np.ndarray:
        """Generate a image(s) using the generator network of
        this :py:class:`StyleGAN`.

        Arguments
        ---------
        features:
            A batch of feature vectors with shape (BATCH_SIZE, FEATURE_DIMS)
            used for the generation.

        Result
        ------
        generatum:
            An array containing the generated images.
        """
        return self.run_tensorflow(self._generate_tensorflow, features)

    def _generate_tensorflow(self, features: np.ndarray) -> np.ndarray:
        """Generate a batch of images from a batch of feature vectors
        using the generator model of this :py:class:`StyleGAN`
        It is assumed that method has to be called in a suitable
        TensorFlow context, meaning that Graph and Session have
        been set up correctly.
        """
        # Some sort of necessary output transform configuration
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        # FIXME[bug]: my run out of memory
        # generated = \
        #    self._generator.run(features, None, truncation_psi=0.7,
        #                        randomize_noise=True, output_transform=fmt)
        batch_size = len(features)
        output_shape = tuple(self._generator.output_shape[1:])
        generated = np.ndarray((batch_size,) + output_shape[1:] +
                               output_shape[:1], dtype=np.uint8)
        # generated2 = np.ndarray((batch_size,) + output_shape)
        # evaluation = np.ndarray((batch_size,) + (1,))
        mini = 4
        LOG.debug("Running generator on %s", features.shape)
        batches = range(0, batch_size, mini)
        if batch_size // mini > 1:
            import tqdm  # FIXME[hack]
            batches = tqdm.tqdm(batches)
        for index in batches:
            end = min(index+mini, batch_size)
            generated[index:end] = \
                self._generator.run(features[index:end], None,
                                    truncation_psi=0.7,
                                    randomize_noise=True,
                                    output_transform=fmt)
            #generated[index:end] = \
            #    self._generator.run(features[index:end], None,
            #                        truncation_psi=0.7,
            #                        randomize_noise=True,
            #                         output_transform=fmt)
            #generated2[index:end] = \
            #    self._generator.run(features[index:end], None,
            #                        truncation_psi=0.7,
            #                        randomize_noise=True)
            #evaluation[index:end] = \
            #    self._discriminator_snapshot.run(generated2[index:end], None)

        LOG.debug("generated images: %s of %s",
                  generated.shape, generated.dtype)
        return generated

    #
    # FIXME[todo]: we need a better info mechanism
    #

    @staticmethod
    def _model_info(model, title: str):
        print(f"{title}:")
        print("-" * len(title))
        print(f"model: {type(model)}")
        if model is not None:
            print(f"model input shape: {model.input_shape}")
            print(f"model output shape: {model.output_shape}")
        print()

    def info(self) -> None:
        """Output StyleGAN model information.
        """
        self._model_info(self._generator_snapshot,
                         'Generator Snapshot')
        self._model_info(self._discriminator_snapshot,
                         'Discriminator Snapshot')
        self._model_info(self._generator_stabil,
                         'Generator Stable')
