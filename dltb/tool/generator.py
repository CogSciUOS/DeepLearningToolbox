"""Abstract interface for generative models.

"""

# standard imports
from typing import Iterable, Union, Tuple
from collections.abc import Iterable as AbstractIterable
import os
import logging

# third-party imports
import numpy as np

# Toolbox imports
from dltb.base.busy import BusyObservable, busy
from dltb.base.data import Data
from dltb.base.image import Image, ImageObservable
from dltb.base.implementation import Implementable

# logging
LOG = logging.getLogger(__name__)


class Generator:
    """A :py:class:`Generator` represents a generative model.
    In general terms, a generative model describes a data distribution
    and allows to sample from that distribution. Sampling can be
    seen as a creative process, generating new data.  The
    :py:class:`Generator` class allows to sample by calling the
    :py:meth:`generate` method.

    Currently this class aims at generative models realized as random
    variable, that is a mapping from a feature space Z into the
    dataspace X, or short: a function X=f(Z).

    A subclass implementing a generative model should overwrite at
    least one of the methdos :py:class:`_generate_single` (sample a
    single datapoint) or :py:class:`_generate_batch` (sample a batch
    of data points).

    In the context of deep learning, prominent approaches to
    generative models are generative adversarial networks (GAN),
    variational autoencoders (VAE), autoregressive architectures, and
    normalized flow. There are subclasses providing specialized
    interfaces to these models.

    """
    _feature_dimensions: int = None

    Data: type = Data

    @property
    def feature_dimensions(self) -> int:
        """The dimensionality of the feature vector.
        """
        return self._feature_dimensions

    @property
    def data_shape(self) -> Tuple[int]:
        """The dimensionality of the data vector.
        """
        return self._data_shape()

    def random_features(self, batch: int = None,
                        seed: Union[int, Iterable[int]] = None) -> np.ndarray:
        """Create random features that can be used to generate data.

        The features will be randomly sampled from the distribution
        the generator is was trained on.  Notice that different
        generators may be trained on different distributions and
        hence using features from the correct distribution is crucial
        for obtaining best generation results.

        The default implementation will create feature vectors,
        sampled from a normal distribution.  Subclasses may overwrite
        this method to realize other distributions.

        Arguments
        ---------
        batch:
            The batch size specifying how many feature vectors
            to create. If `None`, no batch but a single feature vector
            is returned.
        seed:
            Seed for the pseudo random number generator. If an `Iterable`,
            a batch of feature vectors is created, with each vector
            created after the random number generator was seeded with
            the next seed.

        Results
        -------
        features:
            A numpy array, either a single feature vector
            of shape (FEATURE_DIMS,) if `batch` is `None` or a batch
            of feature vectors of shape (`batch`, FEATURE_DIMS) if
            `batch` is an integer.

        """
        if isinstance(seed, AbstractIterable):
            result = []
            for next_seed in seed:
                result.append(self.random_features(seed=next_seed))
            if batch is not None and len(result) != batch:
                raise ValueError(f"Inconsistent batch size ({batch}) "
                                 f"and number of seeds ({len(result)}).")
            return np.asarray(result)

        shape = ((self._feature_dimensions,) if not batch else
                 (batch, self._feature_dimensions))
        return self._randn(shape, seed)

    def _randn(self, shape, seed=0):
        rnd = np.random.RandomState(seed)
        return rnd.randn(*shape)

    def _check_arguments(self, features: np.ndarray,
                         seed: Union[int, Iterable[int]],
                         batch: int) -> Tuple[np.ndarray,
                                              Union[int, Tuple[int]]]:

        if features is not None:
            if seed is not None:
                raise ValueError("Cannot access both, features and seed.")
            if batch is not None:
                if features.ndim == 1:
                    raise ValueError("No batch size is accepted, when "
                                     "specifying a single feature vector.")
                if len(features) != batch:
                    raise ValueError("Inconsistent batch size: "
                                     f"{batch} (batch) vs. "
                                     f"{len(features)} (features).")
            return features, None

        if seed is not None and not isinstance(seed, int):
            seed = tuple(seed)
            if batch is None:
                batch = len(seed)
            elif batch != len(seed):
                raise ValueError("Inconsistent batch size: "
                                 f"{batch} (batch) vs. {len(seed)} (seed).")

        return self.random_features(batch=batch, seed=seed), seed

    def generate(self, features: np.ndarray = None,
                 seed: Union[int, Iterable[int]] = None,
                 batch: int = None) -> Data:
        """Generate data by sampling from the distribution described
        by this :py:class:`Generator`.

        Arguments
        ---------
        features:
            A single feature vector with shape (FEATURE_DIMS,) or a batch
            of feature vectors with shape (BATCH_SIZE, FEATURE_DIMS)
            used for the generation.
        seed:
            Seed for creating a random feature vector. If iteratable,
            a batch of feature vectors will be created.
        batch:
            If not `None` this should be an integer specifying the
            batch size, that is the number of data points to be
            generated (sampled). If `features` or `seed` are also provided,
            the implied batch size should be compatible with the `batch`
            argument.

        Result
        ------
        generatum:
            A :py:class:`Data` object containing the generated data
            as its `array` attribute.  The features used for generation
            will be provided in the `features` attribute. If a seed
            was provided, it will be stored in the `seed` attribute.
        """
        features, seed = self._check_arguments(features, seed, batch)
        array = self._generate_array(features)
        generatum = self.Data(array, batch=(features.ndim > 1))
        generatum.add_attribute('features', value=features, batch=True)
        if seed is not None:
            generatum.add_attribute('seed', value=seed,
                                    batch=not isinstance(seed, int))
        return generatum

    def generate_array(self, features: np.ndarray = None,
                       seed: Union[int, Iterable[int]] = None,
                       batch: int = None) -> np.ndarray:
        """Generate data as (numpy) array.
        """
        features, _seed = self._check_arguments(features, seed, batch)
        return self._generate_array(features)

    def _generate_array(self, features: np.ndarray) -> np.ndarray:
        # check if we have a single feature vector (batch == False)
        # or a batch of feature vectors (batch == True)
        generate_batch = (features.ndim > 1)

        return (self._generate_batch(features) if generate_batch else
                self._generate_single(features))

    def _generate_single(self, features: np.ndarray) -> np.ndarray:
        """
        """
        if type(self)._generate_batch is Generator._generate_batch:
            raise NotImplementedError("At least one of _generate_single "
                                      "or _generate_batch has to be "
                                      "implemented by subclass "
                                      f"{type(self)} of Generator.")

        # generate single by generating a batch of size 1
        return self._generate_batch(features[np.newaxis, ...])[0]

    def _generate_batch(self, features: np.ndarray) -> np.ndarray:
        """Actual implementation of the generator method of this
        :py:class:`Generator`.  Generate a batch of data from
        a batch of feature vectors.

        Arguments
        ---------
        features:
            A batch of feature vectors of shape (BATCH_SIZE, FEATURE_DIMS)
            used for the generation.

        Result
        ------
        generatum:
            An array of shape (BATCH_SIZE, DATA_SHAPE) containing
            the generated data.
        """
        if type(self)._generate_single is Generator._generate_single:
            raise NotImplementedError("At least one of _generate_single "
                                      "or _generate_batch has to be "
                                      "implemented by subclass "
                                      f"{type(self)} of Generator.")

        # generate batch by generating each single element
        first = self._generate_single(features[0])
        batch = np.ndarray((len(features),) + first.shape, dtype=first.dtype)
        batch[0] = first
        for index, feature in enumerate(features[1:]):
            batch[index+1] = self._generate_single(feature)
        return batch

    def transition(self, seed1: int, seed2: int, steps: int = 100):
        """Generate a transition between to feature vectors.
        """
        # 400,602
        # seed1 = 400
        # seed2 = 42
        # STEPS = 100

        features = self.random_features(seed=[seed1, seed2])

        # Take the difference between to vectors and divide it by the
        # number of steps to obtain the transition
        transition = np.linspace(features[0], features[1], steps)
        generatum = self.generate(features=transition)
        generatum.add_attribute('transition', value=(seed1, seed2))
        return generatum


class ImageGenerator(Generator, Implementable):
    """An :py:class:`ImageGenerator` is a :py:class:`Generator`
    specialized on creating images.
    """
    Data: type = Image

    def store_images(self, seeds, directory):
        """Generate a series of images and store them in a directory.
        """

        os.makedirs(directory, exist_ok=True)

        # For each seed, generate the respective image
        seeds = list(seeds)
        for seed_idx, seed in enumerate(seeds):
            LOG.debug("Generating image for seed %d/%d ...",
                      seed_idx, len(seeds))
            image = self.generate_image(seed)

            # Save image
            path = os.path.join(directory, f"image{seed_idx}.png")
            # FIXME[todo]: should by imsave
            import PIL.Image
            PIL.Image.fromarray(image, 'RGB').save(path)

        LOG.debug("Generation complete!")


class ImageGeneratorWorker(ImageObservable, BusyObservable):
    # FIXME[todo]: batch generator
    """An :py:class:`ImageGeneratorWorker` uses an
    :py:class:`ImageGenerator` to (asynchronously) perform image
    generation operation. It will inform :py:class:`ImageObservers`
    whenever a new result was generated.

    """

    _generator: ImageGenerator = None

    def __init__(self, generator: ImageGenerator = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._image = None
        self._next_features = None
        self.generator = generator

    @property
    def generator(self) -> ImageGenerator:
        """The generator applied by this Worker.
        """
        return self._generator

    @generator.setter
    def generator(self, generator: Generator) -> None:
        """Set the :py:class:`Generator` to be used by this Worker.
        """
        self._generator = generator
        self._data = None

    @property
    def image(self) -> Image:
        return self._image

    def generate(self, features: np.ndarray) -> None:
        """Generate data using the generator.
        """
        self._next_features = features
        if not self.busy:
            self._generate()  # FIXME[todo]: self._work()

    @busy("generating")
    # FIXME[hack/bug]: if queueing is enabled, we are not really busy ...
    # (that is we are busy, but nevertheless accepting more work)
    def _generate(self):
        while self._next_features is not None:
            features = self._next_features
            self._next_features = None
            self._image = self._generator.generate(features)
            self.change('image_changed')

    def random(self, seed: int = None) -> None:
        """Generate random data.
        """
        self._image = Image(self._generator.random(seed))
        self.change('data_changed')

    # FIXME[todo]: make some broader video concept
    def make_video(self):
        """Generate a sequence of images into a video stream.
        """
        # Link the images into a video.
        # !ffmpeg -r 30 -i {config.result_dir}/image%d.png -vcodec mpeg4 -y movie.mp4

    # FIXME[todo]:

    @property
    def old_batch(self) -> Image:
        """old? - if not used -> remove
        """
        return self._image


class GAN(Generator):
    """A generative adversarial network.
    """

    @property
    def generator(self) -> Generator:
        pass

    @property
    def discriminator(self):  # FIXME[todo] -> Network:
        pass


class ImageGAN(GAN, ImageGenerator, Implementable):
    """A generative adversarial network (GAN) for generating images.
    """
