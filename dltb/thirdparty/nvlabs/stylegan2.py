"""StyleGAN2 [1] is an improved version of StyleGAN.

[1] https://github.com/NVlabs/stylegan2
"""
# FIXME[bug]: there are some changes that have to be done to the
# code in the stylegan2 github repository in order to make it work
# with tensorflow2:
# 1. Remove tensorflow.contrib (seems not to be used anyhow)
#    sed -i '/tensorflow.contrib/s/^[^#]/#&/' stylegan2/dnnlib/tflib/tfutil.py
# 2. In the file 'stylegan2/dnnlib/tflib/custom_ops.py', which is
#    responsible for running 'nvcc' to compile additional cuda modules,
#    certain adaptations have to be made (depending on the cuda system):
# 2a: Compilation fails with the message "C++ versions less than C++11
#     are not supported." This may be due to an old compiler version. Add
#     a suitable flag to the nvcc invocation by inserting a line into
#     the function '':
#         '-std=c++11'
#

# standard imports
import os
import sys
import pickle
import logging

# third-party imports
import numpy as np

# Toolbox imports
from ...config import config
from ...util.download import download
from ...tool.generator import ImageGAN
from ..tensorflow.v1 import tensorflow as tf
from ..tensorflow.keras import KerasTensorflowModel

# Logging
LOG = logging.getLogger(__name__)

LOG.setLevel(logging.DEBUG)


class StyleGAN2(ImageGAN, KerasTensorflowModel):
    """A StyleGAN1 demonstration. This class is based on the NVIDIA labs
    StyleGAN demo [1].

    Pretrained models
    -----------------
    There are several pretrained models available:

    Each model is provided as Python 3 pickle file (which can simply
    be loaded with `pickle.load`), containing a triple
    (generator_snapshot, discriminator_snapshot, generator_stabil).

    References
    ----------
    [1] https://github.com/NVlabs/stylegan2
    """

    stylegan2_github = 'https://github.com/NVlabs/stylegan2.git'

    # stylegan2_directory:
    #     path to a directory, into which the `stylegan2` repository
    #     (https://github.com/NVlabs/stylegan2.git) was cloned.
    config.set_default_value('nvlabs_stylegan2_directory',
                             os.path.join(config.github_directory,
                                          'stylegan2'))

    # Prepare import of stylegan package
    if not os.path.isdir(config.nvlabs_stylegan2_directory):
        LOG.error("nvlabs stylegan2 directory not found in '%s'",
                  config.nvlabs_stylegan2_directory)
        LOG.info("install it by typing: git clone %s %s",
                 stylegan2_github, config.nvlabs_stylegan2_directory)
        sys.exit(1)  # FIXME[hack]: do not exit - find a better exception handling mechanism

    model_dict = {
        'ffhq': 'gdrive:networks/stylegan2-ffhq-config-f.pkl',
        'church': 'gdrive:networks/stylegan2-church-config-f.pkl'
    }
    models = list(model_dict.keys())

    def __init__(self, model: str = None, filename: str = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._generator = None
        self._generator_snapshot = None
        self._discriminator_snapshot = None
        self._generator_stabil = None
        self._noise_vars = None

        self._model = model
        self._model_filename = filename
        self._network_pkl = None
        if (model or filename) is None:
            self.model = 'ffhq'  # default

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, model: str) -> None:
        if self._model == model:
            return  # nothing todo

        self.unprepare()
        self._model = model
        self._network_pkl = self.model_dict[model]
        self._model_filename = \
            os.path.join(config.model_directory,
                         self._network_pkl.rsplit('/', maxsplit=1)[-1])

    def _prepared(self) -> bool:
        """The :py:class:`StyleGAN2` preparation is finished by assigning
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
        if config.nvlabs_stylegan2_directory not in sys.path:
            sys.path.insert(0, config.nvlabs_stylegan2_directory)
        LOG.debug("Importing module 'dnnlib' from '%s'",
                  config.nvlabs_stylegan2_directory)

        # dnnlib is contained in the stylegan2 repository
        global tflib
        import dnnlib.tflib as tflib
        import pretrained_networks

        sys.path.remove(config.nvlabs_stylegan2_directory)

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

        #
        # Step 3: Load the network
        #

        # download model file if not exist
        if not os.path.exists(self._model_filename):
            url = pretrained_networks.gdrive_urls[self._network_pkl]
            download(url, self._model_filename)

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
        # NVCC - dnnlib/tflib/custom_ops.py issues:
        # Loading the StyleGAN2 pickle file triggers the compilation
        # of some CUDA code ('fused_bias_act.cu' and 'upfirdn_2d.cu')
        # which is initiated by 'dnnlib/tflib/custom_ops.py'. There
        # are several issues with that code (especially with TensorFlow 2.x):
        #
        # 1. in TensorFlow v1 compatibility mode, the method
        #    tf.sysconfig.get_lib() returns
        #    'SITE_PACKAGES/tensorflow/_api/v2/compat/v1' instead of
        #    'SITE_PACKAGES/tensorflow', which prevent the shared library
        #    '_pywrap_tensorflow_internal.so' from being found.
        #    Solution: use
        #      compile_opts += '"%s"' % os.path.join(
        #         tf.__file__.rsplit('tensorflow', maxsplit=1)[0],
        #         'tensorflow', 'python', '_pywrap_tensorflow_internal.so')
        #
        # 2. Modern tensorflow (>=1.15) may require the compiler flag
        #    "-D_GLIBCXX_USE_CXX11_ABI=1" instead of the one defined
        #    in 'custom_ops.py' (which is '-D_GLIBCXX_USE_CXX11_ABI=0')
        #
        # 3. For some reason, the CUDA code is compiled everytime I
        #    load the pickle file. There should be some possibility to
        #    keep the result, but I was not able to find it yet.

        LOG.info("Initializing StyleGAN2 models from '%s'",
                 self._model_filename)
        try:
            # FIXME[todo]: check if chdir is necessary or can be omitted
            print(f"*** Changing to StyleGAN2 directory: '{config.nvlabs_stylegan2_directory}'")
            LOG.debug("Changing to StyleGAN2 directory: '%s'",
                      config.nvlabs_stylegan2_directory)
            cwd = os.getcwd()
            os.chdir(config.nvlabs_stylegan2_directory)
            os.environ['CPATH'] = '/space/conda/user/ulf/envs/dl-toolbox/lib/python3.7/site-packages/tensorflow/include'
            with open(self._model_filename, 'rb') as file:
                self._generator_snapshot, self._discriminator_snapshot, \
                    self._generator_stabil = pickle.load(file,
                                                         encoding='latin1')
        finally:
            print(f"*** Changing back to old directory: '{cwd}'")
            LOG.debug("Changing back to old working directory: '%s'", cwd)
            os.chdir(cwd)

        # Select the generator
        generator = self._generator_stabil
        self._feature_dimensions = generator.input_shape[1]
        self._noise_vars = \
            [var for name, var in generator.components.synthesis.vars.items()
             if name.startswith('noise')]

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
        self._noise_vars = None
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
        truncation_psi: float = 0.7

        from dnnlib import EasyDict
        Gs_kwargs = EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8,
                                          nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        Gs_kwargs.truncation_psi = truncation_psi

        # print(f"StyleGAN2 Features: {features.shape}, {features.dtype}")
        if False:
            # FIXME[bug]: for "stylegan2-church-config-f.pkl"
            # KeyError: "The name 'G_synthesis_1/noise0/setter:0' refers
            #            to a Tensor which does not exist. The operation,
            #            'G_synthesis_1/noise0/setter', exists but only
            #            has 0 outputs."
            noise_rnd = np.random.RandomState(1)  # fix noise
            tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list())
                            for var in self._noise_vars})  # [height, width]

        # [minibatch, height, width, channel]
        images = self._generator.run(features, None, **Gs_kwargs)
        # print(f"StyleGAN2 images: {images.shape}, {images.dtype}")
        # PIL.Image.fromarray(images[0], 'RGB'))
        return images

        return self.generate_images([features])[0]

    def generate_images_in_w_space(self, dlatents, truncation_psi):
        """Generates a list of images, based on a list of latent vectors (Z),
        and a list (or a single constant) of truncation_psi's.

        """
        from dnnlib import EasyDict
        Gs_kwargs = EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8,
                                          nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        Gs_kwargs.truncation_psi = truncation_psi
        dlatent_avg = self._generator.get_var('dlatent_avg')  # [component]

        imgs = []
        for row, dlatent in enumerate(dlatents):
            # row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * \
            #    np.reshape(truncation_psi, [-1, 1, 1]) + dlatent_avg
            dl = (dlatent-dlatent_avg) * truncation_psi + dlatent_avg
            row_images = \
                self._generator.components.synthesis.run(dlatent, **Gs_kwargs)
            # imgs.append(PIL.Image.fromarray(row_images[0], 'RGB'))
            imgs.append(row_images[0])
        return imgs

    def generate_images(self, features_list, truncation_psi: float = 0.7):
        """Generate array of images.
        """
        from dnnlib import EasyDict
        Gs_kwargs = EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8,
                                          nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        if not isinstance(truncation_psi, list):
            truncation_psi = [truncation_psi] * len(features_list)

        imgs = []
        for index, features in enumerate(features_list):
            Gs_kwargs.truncation_psi = truncation_psi[index]
            noise_rnd = np.random.RandomState(1)  # fix noise
            tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list())
                            for var in self._noise_vars})  # [height, width]

            # [minibatch, height, width, channel]
            images = self._generator.run(features, None, **Gs_kwargs)
            # imgs.append(PIL.Image.fromarray(images[0], 'RGB'))
            imgs.append(images[0])
        return imgs

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

    def _data_shape(self):
        # swap color channel
        return (self._generator.output_shape[2:] +
                self._generator.output_shape[1:2])
