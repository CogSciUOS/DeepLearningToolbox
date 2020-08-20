"""The dogs and cats dataset.
"""

# standard imports
import os

# toolbox imports
from .data import ClassScheme
from .datasource import Imagesource
from .directory import DataDirectory


class DogsAndCats(DataDirectory, Imagesource):
    # pylint: disable=too-many-ancestors
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
    """

    def __init__(self, key: str = None, **kwargs):
        """Initialize the :py:class:`DogsAndCats` dataset.
        """
        key = key or f"cats-and-dogs"
        description = f"Dogs vs. Cats"
        directory = os.getenv('DOGSANDCATS_DATA', '.')
        scheme = ClassScheme(2, key='dogsandcats')
        scheme.add_labels('text', ['Cat', 'Dog'])
        super().__init__(key=key, description=description,
                         directory=directory, scheme=scheme, **kwargs)

    def __str__(self):
        return "Dogs vs. Cats"
