"""The MNIST dataset.

Demo 1: basic usage
-------------------

from dltb.thirdparty.datasource.mnist import MNIST
mnist = MNIST()
assert len(mnist), 60000

mnist_test = MNIST(section='test')
assert len(mnist_test), 10000


Demo 2: Implementable
---------------------

from dltb.datasource import Datasource
mnist = Datasource(module='mnist', section='test')


Demo 3: one-hot encoded labels
------------------------------

from dltb.datasource import Datasource
mnist = Datasource(module='mnist', section='test', one_hot=True)
mnist[0, ('label', 'label')]


Demo 4: downloading the data
----------------------------

from pathlib import Path
import shutil
from dltb.datasource import Datasource
mnist_directory = Path('./MNIST')

try:
   mnist_directory.mkdir()  # may raise FileExistsError
   mnist = Datasource(module='mnist', directory=mnist_directory)
   shutil.rmtree(mnist_directory)
except FileExistsError:
   print(f"Directory {mnist_directory} already exists")


"""

# standard imports
from typing import Iterator, Tuple
import logging
import gzip

# thirdparty imports
import numpy as np

# toolbox imports
from dltb import config
from dltb.util.download import download
from dltb.types import Path, Pathlike, as_path
from dltb.datasource import Imagesource
from dltb.datasource.array import LabeledArray
from dltb.tool.classifier import ClassScheme

# logging
LOG = logging.getLogger(__name__)

# configuration
config.add_property('mnist_directory',
                    default=lambda c: c.data_directory / 'mnist',
                    description="Directory where MNIST data are stored."
                    "The files {train,t10k}-images-idx3-ubyte.gz and "
                    "{train,t10k}-labels-idx1-ubyte.gz can be found here.")


class MNIST(LabeledArray, Imagesource):
    """Datasource interface to the famous MNIST hand-written digit
    dataset.
    """

    url = "http://yann.lecun.com/exdb/mnist/"

    def __init__(self, section: str = 'train', validation_split: bool = True,
                 directory: Pathlike = None,
                 **kwargs) -> None:
        """
        """
        self._section = section
        self._validation_split = validation_split
        shape = (28, 28, 1)
        scheme = ClassScheme(10)
        description = f"The MNIST handwritten number dataset ({section})"
        # The array will only be provided by prepare()
        super().__init__(array=None, labels=None,
                         shape=shape, scheme=scheme,
                         description=description, **kwargs)
        self._mnist_archive_directory = config.mnist_directory \
            if directory is None else as_path(directory)

    def _prepare(self) -> None:
        # Obtain paths to data files (download if necessary)
        if self._section in ('train', 'valid'):
            images_path = self._maybe_download('train-images-idx3-ubyte.gz')
            labels_path = self._maybe_download('train-labels-idx1-ubyte.gz')
            n_images = 60000
            if self._validation_split:
                indices = slice(0, 50000) if self._section == 'train' else \
                    slice(50000, 60000)
            else:
                indices = slice(0, 60000)
        else:
            images_path = self._maybe_download('t10k-images-idx3-ubyte.gz')
            labels_path = self._maybe_download('t10k-labels-idx1-ubyte.gz')
            n_images = 10000
            indices = slice(0, 10000)

        # load the images (uint8)
        all_images = self._extract_mnist_images(images_path, n_images)
        self._images = all_images[indices]

        # preprocessing: obtain data array: dtype=float, min/max=0.0/1.0
        self._array = self._images/255
        # alternative preprocessing: centering, normalization, ...
        # mean = np.mean(self._images, axis=(0, 1, 2))
        # std = np.mean(self._images, axis=(0, 1, 2))
        # self._array = self._images/255

        # labels: uint8, min/max=0/9
        all_labels = self._extract_mnist_labels(labels_path, n_images)
        self._labels = all_labels[indices]

        super()._prepare()

    def _maybe_download(self, filename: str) -> Path:
        """Download the dataset if not yet downloaded.

        Arguments
        ---------
        filename:
            The basename of the file.

        Result
        ------
        filepath:
            `Path` object refering to the downloaded data file.
        """
        filepath = self._mnist_archive_directory / filename
        url = self.url + filename
        download(url, filepath, skip_if_exists=True)

        LOG.info("Have file '%s' with %d bytes.",
                 filepath, filepath.stat().st_size)
        return filepath

    def _extract_mnist_images(self, filepath: Path,
                              n_images: int) -> np.ndarray:
        """Extract the MNIST labels from a zipped image
        file (`...-images-idx1-ubyte.gz`).
        """
        LOG.info("Extracting and Reading %s", filepath)

        height, width, channels = self.shape
        with gzip.open(filepath) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(height * width * n_images * channels)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = np.reshape(data, [n_images, height, width, channels])
        return data

    @staticmethod
    def _extract_mnist_labels(filepath: Path,
                              n_images: int) -> np.ndarray:
        """Extract the MNIST labels from a zipped label
        file (`...-labels-idx1-ubyte.gz`).
        """
        LOG.info("Extracting and Reading %s", filepath)

        with gzip.open(filepath) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1*n_images)
            labels = np.frombuffer(buf, dtype=np.uint8)
        return labels

