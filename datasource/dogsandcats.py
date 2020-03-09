from . import Datasource, Labeled, Random

from util.image import imread

import os
import random
import numpy as np

import logging
logger = logging.getLogger(__name__)


class DogsAndCats(Random, Labeled):
    """An interface to the Cats & Dogs dataset.

    The training archive contains 25,000 images of dogs and
    cats. Train your algorithm on these files and predict the labels
    for test1.zip (1 = dog, 0 = cat).
    
    [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats)

    kaggle competitions download -c dogs-vs-cats
    
    In this competition, you'll write an algorithm to classify whether
    images contain either a dog or a cat.  This is easy for humans,
    dogs, and cats. Your computer will find it a bit more difficult.

    Asirra (Animal Species Image Recognition for Restricting Access)
    is a HIP that works by asking users to identify photographs of
    cats and dogs. This task is difficult for computers, but studies
    have shown that people can accomplish it quickly and
    accurately. Many even think it's fun! Here is an example of the
    Asirra interface:

    Asirra is unique because of its partnership with Petfinder.com,
    the world's largest site devoted to finding homes for homeless
    pets. They've provided Microsoft Research with over three million
    images of cats and dogs, manually classified by people at
    thousands of animal shelters across the United States. Kaggle is
    fortunate to offer a subset of this data for fun and research.


    'test.zip' is currently not available to me - try to download ...


    Attributes
    ----------
    _directory: str
    _label: int
        The category of the current image: either "Cat" or "Dog".
    _image: np.ndarray
    """

    _directory: str = None
    _label: int = None
    _image: np.ndarray = None
    
    def __init__(self, prefix=None, section='train', **kwargs):
        super().__init__(id=f"cats-and-dogs-{section}",
                         description=f"Dogs vs. Cats", **kwargs)
        self._image = None

    @property
    def prepared(self) -> bool:
        """Report if this Datasource prepared for use.
        A Datasource has to be prepared before it can be used.
        """
        return self._directory is not None

    def _prepare_data(self):
        """Prepare this Datasource for use.
        """
        logger.info(f"PREPARING Dogs vs. Cats dataset ...")
        directory = os.getenv('DOGSANDCATS_DATA', '.')
        if not os.path.isdir(directory):
            logger.info(f"PREPARING Dogs vs. Cats: failed: no directory")
            raise RuntimeError("Cannot locate Dogs vs. Cats directory!")
        self._directory = directory
        self.add_label_format('str', ['Cat', 'Dog'])
        logger.info(f"PREPARING Dogs vs. Cats: success: {self._directory}")

    @property
    def number_of_labels(self) -> int:
        """There are two labels for this dataset: Cats and Dogs.
        """
        return 2

    def _fetch(self, **kwargs):
        """Fetch a random image from the dataset.
        """
        self._fetch_random(**kwargs)

    def _fetch_random(self, **kwargs):
        """Fetch a random image from the dataset.
        """
        label = random.randint(0,1)
        filename = str(random.randint(0,12499)) + '.jpg'
        img_file = os.path.join(self._directory,
                                self.format_labels(label, 'str'),
                                filename)
        self._image = imread(img_file)
        self._label = label

    def fetched(self):
        """Fetch a random image from the dataset.
        """
        return self._image is not None

    def _get_data(self):
        """The actual implementation of the :py:meth:`data` property
        to be overwritten by subclasses.

        It can be assumed that a data point has been fetched when this
        method is invoked.
        """
        return self._image

    def _get_label(self):
        return self._label

    def __str__(self):
        return "Dogs vs. Cats"
