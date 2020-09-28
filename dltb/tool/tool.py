"""Abstract base class for tools.
"""

# standard imports
from typing import Any, Union, Iterator
import time
import logging

# third party imports
import numpy as np

# toolbox imports
from base import RegisterClass
from ..base.resource import Resource
from ..base.data import Data, BatchWrapper, BatchDataItem

# logging
LOG = logging.getLogger(__name__)


Datalike = Union[Data, np.ndarray]


class Tool(Resource, metaclass=RegisterClass, method='tool_changed'):
    # pylint: disable=too-many-ancestors
    """:py:class:`Tool` is an abstract base class for tools.
    A Tool can be applied to process data. This abstract base class
    does not define any specific method for processing data,
    this should be done by subclasses (e.g., a Detector provides a
    detect method).

    In addition, :py:class:`Tool` provides some support for processing
    data with a Processor.

    This is an abstract base class and subclasses can (or have to)
    implement at least some of the following methods:

    :py:meth:`_preprocess_data`:
        Preprocess the given data. This method will be invoked
        before processing to bring the data into an appropriate
        format. This method should be overwritten if the Tool expects
        data to be in a specific format.
    :py:meth:`_process_data` and :py:meth:`_process_batch`:
        Do the actal processing and store results as tool-specific
        attributes in the given :py:class:`Data` object. Either
        `_process_data` or `_process_batch` have to be overwritten
        (but it is also allowed to owverwrite both).

    Attributes
    ----------
    timer: bool
        A flag indicating if timing information should be added
        to the :py:class:`Data` object during processing.
    """

    def __init__(self, timer: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.timer = timer
        LOG.info("New tool: %s (%s)", self.key, type(self).__name__)

    def __str__(self) -> str:
        return f"{type(self).__name__}[{self.key}]"

    def _prepare(self, **kwargs):
        # pylint: disable=arguments-differ
        """Prepare this tool.
        """
        super()._prepare(**kwargs)

    def preprocess(self, data: Datalike, **kwargs) -> np.ndarray:
        """Perform preprocessing necessary to obtain data suitable for
        for processing with this tool.
        """
        if isinstance(data, np.ndarray):
            return self._preprocess(data, **kwargs)

        if not isinstance(data, Data):
            ValueError(f"Argment data is of invalid type {type(data)}.")

        array = self.preprocessed(data)
        if array is not None:
            return array

        return self._preprocess(data.array, **kwargs)

    def _preprocess(self, array: np.ndarray, **_kwargs) -> np.ndarray:
        # pylint: disable=no-self-use
        """Preprocess the given data. The default implementation does nothing,
        it just returns the input data. The method may be implemented
        by subclasses to do the actual preprocessing.

        Arguments
        ---------
        data: np.ndarray
            The data to be preprocessed.

        Result
        ------
        array: np.ndarray
            The preprocessed data.
        """
        return array

    def add_data_attribute(self, data: Data, name: str, value: Any = None,
                           batch: bool = True) -> None:
        """Add a tool specific attribute to a data object.
        """
        data.add_attribute(self.key + '_' + name, value, batch=batch)

    def set_data_attribute(self, data: Data, name: str, value: Any,
                           index: int = None) -> None:
        """Set a tool specific attribute in a data object.
        """
        setattr(data if index is None else data[index],
                self.key + '_' + name, value)

    def get_data_attribute(self, data: Data, name: str, default: Any = None,
                           index: int = None) -> Any:
        """Get a tool specific attribute from a data object.
        """
        value = getattr(data if index is None else data[index],
                        self.key + '_' + name, None)
        return default if value is None else value

    #
    # data Preprocessing
    #

    def preprocess_data(self, data: Data, **kwargs) -> None:
        """Preprocess the data given data. The preprocessed version of
        the data is stored in the tool-specific 'preprocessed' attribute
        of the :py:class:`Data` object.

        Arguments
        ---------
        data: Data
            A :py:class:`Data` object that can be a batch or a single
            data item.

        """
        self._preprocess_data(data, **kwargs)

    def _preprocess_data(self, data: Data, **kwargs) -> None:
        """This method does the actual preprocessing.

        This method may be overwritten by subclasses to add attributes
        to the data object, without assigning values yet (this can be
        done in :py:meth:`_process_data` or
        :py:meth:`_process_batch`). This method may set the data
        object and notify observers, allowing them to observe how the
        data object gets filled during processing.
        """
        self.add_data_attribute(data, 'preprocessed')
        self.set_data_attribute(data, 'preprocessed',
                                self.preprocess(data, **kwargs))
        if self.timer:
            self.add_data_attribute(data, 'duration')

    def preprocessed(self, data: Data) -> np.ndarray:
        """Get the tool-specific preprocessed data from a
        :py:class:`Data` object.  This may be `None` if the data object
        has not yet undergone preprocessing.
        """
        return self.get_data_attribute(data, 'preprocessed', data.array)

    #
    # Processing
    #

    def process(self, data: Data, **kwargs) -> None:
        """Process the data and adapt the :py:class:`Data` object to
        hold the results.
        This method is to be implemented by subclasses.
        Subclasses may also augment the given data object with
        intermediate results if requested.

        Arguments
        ---------
        data: Data
            A :py:class:`Data` object that is guaranteed not to be a batch.
        """
        self.preprocess_data(data, **kwargs)
        if self.timer:
            start = time.time()
        if data.is_batch:
            self._process_batch(data, **kwargs)
        else:
            self._process_data(data, **kwargs)
        if self.timer:
            end = time.time()
            self.set_data_attribute(data, 'duration', end - start)
        self._postprocess_data(data, **kwargs)

    def _process_data(self, data: Data, **kwargs) -> None:
        """Adapt the data.
        To be implemented by subclasses. Subclasses may augment
        the given data object with the result of their processing.

        Arguments
        ---------
        data: Data
            A :py:class:`Data` object that is guaranteed not to be a batch
            and to provide preprocessed data in its tool-specific
            'preprocessed' attribute.
        """
        if isinstance(data, BatchDataItem):
            raise ValueError("Use _process_batch to process batch data.")
        self._process_batch(BatchWrapper(data))

    def _process_batch(self, batch: Data, **kwargs) -> None:
        """Process batch data. The base implementation just
        processes each batch element individually. Subclasses may
        overwrite this to do real batch processing.

        Arguments
        ---------
        batch: Data
            A :py:class:`Data` object that is guaranteed to be a batch
            and to provide preprocessed data in its tool-specific
            'preprocessed' attribute.
        """
        if isinstance(batch, BatchWrapper):
            raise ValueError("Use _process_data to process non-batch data.")
        for data in batch:
            self._process_data(data)

    def duration(self, data) -> float:
        """Provide the duration (in seconds) the tool needed for processing
        the given data. This property is only available after processing,
        and only if the timer was activated.
        """
        return self.get_data_attribute(data, 'duration')

    def _postprocess_data(self, data: Data, **kwargs) -> None:
        """Postprocess the data. This may result in adding tool specific
        attributes to the data object.  It is assumed that the data
        object was successfully processed by :py:meth:`_process_data`
        before this method is called.

        Arguments
        ---------
        data: Data
            The data object to be postprocessd.
        """
        # to be extended by subclasses


class IterativeTool(Tool):
    """An iterative tool performs its operation as an iterative process.
    """

    #
    # public interface
    #

    def __call__(self, data: Datalike, **kwargs) -> Datalike:
        """Apply the tool to the given data. The input data will
        not be modified. The tool specific result will be returned.
        """
        internal = self._preprocess(data)
        arguments = self._preprocess_arguments(**kwargs)
        for internal in self._steps(internal, **arguments):
            pass
        return self._postprocess(internal, data)

    def step(self, data: Datalike, **kwargs) -> Datalike:
        """Perform a single step of the iterative process.
        """
        internal = self._preprocess(data)
        arguments = self._preprocess_arguments(**kwargs)
        return self._postprocess(self._step(internal, **arguments), data)

    def steps(self, data: Datalike, **kwargs) -> Iterator[Datalike]:
        """Iterate the steps of the iterative process, providing
        an intermediate result at every step.
        """
        internal = self.preprocess(data)
        arguments = self._preprocess_arguments(**kwargs)
        for internal in self._steps(internal, **arguments):
            yield self._postprocess(internal, data)

    #
    # private (internal) methods
    #

    def _step(self, data: Any, **kwargs) -> Any:
        # to be implemented by subclasses
        raise NotImplementedError()

    def _steps(self, data: Any, **kwargs) -> Iterator[Any]:
        while True:  # stop criterien (kwargs)
            data = self._step(data, **kwargs)
            yield data

    def process(self, data: Data, **kwargs) -> None:
        """Perform iterative processing on the data object.
        After each step, the relevant attributes of the data object
        are updated (and interested observers are informed.

        """
        self.preprocess_data(data)
        preprocessed = self.get_data_attribute(data, 'preprocessed')
        arguments = self._preprocess_arguments(**kwargs)
        # FIXME[concept]: there should be a way to interrupt the process
        for intermediate in self._steps(preprocessed, **arguments):
            self._postprocess_data(data, intermediate)
