"""Tools for supporting iterations (iterable, iterators, generators, etc.).
"""

# standard imports
from typing import Generator, Protocol, Iterable


class SizedIterable(Protocol):
    """An :py:class:`Iterable` with a known length.
    """

    def __len__(self):
        pass

    def __iter__(self):
        pass


class SizedGenerator:
    """A generator with a known length.
    """
    def __init__(self, gen: Generator, length: int):
        self.gen = gen
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Generator:
        return self.gen

    def __next__(self):
        return next(self.gen)


def ignore_errors(iterable: Iterable, error_type: type,
                  logger=None) -> Iterable:
    while True:
        try:
            yield next(iterable)
        except error_type as error:
            if logger is not None:
                logger.warning("Ignoring error: %s", error)
        except StopIteration:
            break
