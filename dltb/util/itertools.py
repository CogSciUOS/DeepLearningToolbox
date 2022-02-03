"""Tools for supporting iterations (iterable, iterators, generators, etc.).
"""

# standard imports
from typing import Generator, Iterable
import os

# toolbox imports
from ..typing import Protocol

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
    """A wraper that allows to ignore exceptions raised during
    an iteration.

    Arguments
    ---------
    iterable:
        The `Iterable` to wrap.
    error_type:
        The exception to catch.
    logger:
        A logger to which catched exceptions are logged.
    """
    while True:
        try:
            yield next(iterable)
        except error_type as error:
            if logger is not None:
                logger.warning("Ignoring error: %s", error)
        except StopIteration:
            break


class Selection(Iterable[int]):
    """Iterate over a selection of numbers, specified as text.  A
    selection can be specified by multiple ranges, separated by
    commas.  Each range can either be a singleton, or a pair of first
    and last element, separated by hyhpen.

    The class supports some special cases:

    sge-task:
        The numbers are taken from the sun grid engine task parameters.
        These can be provided when submitting the command with `qsub` as
        an array job, using the -t m[-n[:s]] command line option.
        The task parameters are then reported via the SGE_TASK_* environment
        variables.
    """

    def __init__(self, selection: str):
        self._selection = selection

    def __iter__(self):
        if not self._selection:
            yield None
        elif self._selection == 'sge-task':
            # Use value from the sun grid engine task parameters
            # These can be provided with -t m[-n[:s]] and are reported
            # via the SGE_TASK_* environment variables.
            if os.getenv('SGE_TASK_ID', 'undefined') == 'undefined':
                raise ValueError("Selection was specified as 'sge-task', but no "
                                 "task id (SGE_TASK_ID) could be determined.")

            # logging.debug("Running an array job:")
            # logging.debug(" SGE_TASK_FIRST=%s", os.getenv('SGE_TASK_ID'))
            # logging.debug(" SGE_TASK_LAST=%s}", os.getenv('SGE_TASK_LAST'))
            # logging.debug(" SGE_TASK_STEPSIZE=%s",
            #               os.getenv('SGE_TASK_STEPSIZE'))
            # logging.debug(" SGE_TASK_ID=%s", os.getenv('SGE_TASK_ID'))

            task_id = int(os.getenv('SGE_TASK_ID')) - 1
            # _task_first = int(os.getenv('SGE_TASK_FIRST'))
            task_last = int(os.getenv('SGE_TASK_LAST'))
            task_stepsize = int(os.getenv('SGE_TASK_STEPSIZE'))
            for number in range(task_id, min(task_last, task_id + task_stepsize)):
                yield number
        else:
            for part in self._selection.split(','):
                if '-' not in part:
                    yield int(part)
                else:
                    first, last = part.split('-', maxsplit=1)
                    for number in range(int(first), int(last)+1):
                        yield number
