"""Abstract base class for tools.
"""

# standard imports
from typing import Any, Union, Tuple, Iterator
import time
import logging
from threading import Event

# third party imports

# toolbox imports
from ..base.data import Datalike
from ..base.register import RegisterClass
from ..base.prepare import Preparable
from ..base.data import Data, BatchWrapper, BatchDataItem

# logging
LOG = logging.getLogger(__name__)


# FIXME[todo]: specify the observable interface: there should be
# at least one change 'tool_changed' indicating that the tool
# has changed in a way that it will now yield different results
# to prior application. This may be caused by a change of configuration
# parameters, exchange of the underlying engine, or the tool
# becoming perpared ...
class Tool(Preparable, metaclass=RegisterClass, method='tool_changed'):
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


    Class Attributes
    ----------------
    external_result: Tuple[str]
        A tuple naming the values to be returned by an application
        of the :py:class:`Tool`. These values will be constructed
        by calling :py:meth:`_postprocess` on the intermediate values.
    internal_arguments: Tuple[str] = ()
        A tuple naming the positional arguments for calling the internal
        processing function :py:meth:`_process`. Values for these
        names will be taken from the intermediate data structure,
        which should be filled by :py:class:`_preprocess`
    internal_result: Tuple[str] = ()
        A name tuple naming the results of the internal processing.
        Values will be stored under this name in the intermediate
        data structure.

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

    #
    # Application API
    #

    external_result: Tuple[str] = ()
    internal_arguments: Tuple[str] = ()
    internal_result: Tuple[str] = ()

    def __call__(self, *args, batch: bool = False, internal: bool = False,
                 result: Union[str, Tuple[str]] = None,
                 **kwargs) -> Any:
        """Apply the tool to the given arguments.

        Arguments
        ---------
        batch: bool
            A flag indicating that arguments are given a batch,
            do batch_processing.
        internal: bool
            A flag indicating that arguments are given in internal format,
            no preprocessing of the given data is required and no
            postprocessing is applied.
        result: Union[str, Tuple[str]]
            A description of the result values to return. If None is
            given, the Tools default will be used (if the `internal` flag
            is set, the internal result will be returned without
            postprocessing)

        Result
        ------
        result:
            The result of applying this :py:class:`Tool`. This is usually
            a postprocessed result, as specified by the property
            :py:prop:`external_result`.  If other return values are
            desired, those can be specified by the argument `result`.
            If the argument `internal` is `True`, instead of the
            postprocessed results, the internal result is returned,
            as specified by the property `internal` and computed
            by the :py:meth:`_process`.
        """
        # FIXME[todo]: batch processing:
        #  - a tool may implement (either or both)
        #    _process_single()
        #    _process_batch()
        # preprocessing, processing and postprocessing have to
        # deal with this

        # preprocessing
        data = self._do_preprocess(*args, internal=internal, batch=batch,
                                   result=result, **kwargs)
        internal_arguments = \
            self._get_attributes(data, self.internal_arguments)

        result = data.result_

        # processing
        if data.is_batch:
            internal_result = self._process(*internal_arguments)
            # self._process_batch(data, **kwargs)
        else:
            internal_result = self._process(*internal_arguments)
            # self._process_data(data, **kwargs)

        # postprocessing
        if internal and result is None:
            return internal_result
        self._add_attributes(data, self.internal_result,
                             internal_result, simplify=True)
        return self._do_postprocess(data)

    def _preprocess_data(self, data: Data, **kwargs) -> None:
        """Prepare a data object.  This method may be overwritten by
        subclasses to add attributes to a data object, without
        assigning values yet (this can be done in
        :py:meth:`_process_data` or :py:meth:`_process_batch`). This
        method may set the data object and notify observers, allowing
        them to observe how the data object gets filled during
        processing.
        """

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

    def _do_preprocess(self, *args, internal: bool = False,
                       batch: bool = False,
                       result: Union[str, Tuple[str]] = None,
                       **kwargs) -> Data:
        """Perform preprocessing of arguments provided when invoking the
        tool.  The actual preprocessing steps depend on the specific
        tool and the format of the data presented.

        The main idea of this method is to create a new auxiliary(!)
        :py:class:`Data` object, intended to hold internal and
        external arguments and results as attributes. The idea is,
        that these attributes are to be incrementally filled during
        processing.  At the end of processing the relvant values are
        extracted from that object and the object is then deleted.

        The base implementation of this function will fill in the
        internal arguments required for calling the underlying tool
        implementation. If preprocessing is required (if the arguments
        are not already given in an internal format), the method
        :py:meth:`_preprocess` is invoked to perform the
        preprocessing.  In addition, the attribute `result_` will be
        added and hold the names (as a list of str) of the values to
        return. If a `'duration'` is to be returned, also the start
        time will be inserted in the attribute `'start_'`.

        Arguments
        ---------
        *args:
            The original arguments passed when invoking the tool.
        internal:
            A flag indicating that the data are already provided in
            the internal format. No additional preprocessing is required.
        batch:
            A flag indicating if processing is to be done on an individual
            datum or on a batch of data.
        result:
            The names of the values to be returned by the tool.
            If this argument is not provided, the default list,
            defined by the property :py:prop:`external_result` is used.

        """
        if result is None and not internal:
            result = self.external_result
        elif isinstance(result, str):
            result = (result, )

        if internal:
            context = Data(batch=batch)
            for name, arg in zip(self.internal_arguments, args):
                context.add_attribute(name, arg)
        else:
            context = self._preprocess(*args, batch=batch, **kwargs)
        context.add_attribute('result_', result)

        if 'duration' in result:
            context.add_attribute('start_', time.time())
        return context

    def _do_postprocess(self, data: Data) -> Any:
        """Perform postprocessing and provide return value(s).

        Arguments
        ---------
        data:
            The (auxiliary) :py:class:`Data` object used for storing
            internal values.
        """
        result = data.result_
        for name in result:
            self._postprocess(data, name)

        return self._get_attributes(data, result, simplify=True)

    @staticmethod
    def _get_attributes(data: Data, what: Tuple[str],
                        simplify: bool = False) -> Any:
        if simplify:
            if len(what) == 0:
                return None
            if len(what) == 1:
                return getattr(data, what[0])
        return tuple(getattr(data, arg) for arg in what)

    @staticmethod
    def _add_attributes(data: Data, what: Tuple[str], args,
                        simplify: bool = False) -> None:
        if simplify and len(what) == 0:
            pass
        elif simplify and len(what) == 1:
            data.add_attribute(what[0], args)
        else:
            for name, value in zip(what, args):
                data.add_attribute(name, value)

    #
    # Private interface (to be implemented by subclasses):
    #

    def _preprocess(self, *arg, batch: bool = False, **kwargs) -> Data:
        """Perform preprocessing of arguments. Preprocessed arguments
        are stored as attributes of an auxiliary :py:class:`Data` object.

        This method should be extended by subclasses to incorporate
        additional preprocessing operations.

        Arguments
        ---------
        batch:
        *args:
             positional arguments passed when invoking the tool.
        **kwargs:
             keyword arguments passed when invoking the tool.

        Result
        ------
        data:
            The auxiliary :py:class:`Data` object to which preprocessed
            information have been added.  Alle intermediate results
            obtained during application of the :py:class:`Tool`
            should also be stored in that object.
        """
        context = Data(batch=batch)
        if self.timer:
            context.add_attribute('start', time.time())
        return context

    def _process(self, *args, **kwargs) -> Any:
        """Do the actual processing.

        To be implemented by subclasses.

        Arguments
        ---------
        *args:
            The positional arguments of the functions. These are
            the (preprocessed) arguments named by the property
            :py:prop:`internal_arguments`.
        **kwargs:
            Optional keyword arguments passed when invoking the tool.
        """
        raise NotImplementedError()

    def _postprocess(self, data: Data, what: str) -> None:
        """Perform postprocessing.

        This method is intended to be extended by subclasses. If doing
        so, such methods should call `super()._postprocess(data, what)`
        in case they can not handle the property `what` themself.

        Arguments
        ---------
        data:
            The auxiliary :py:class:`Data` object to which results
            of the postprocessing should be stored.
        what:
            An identifier specifying what kind of postprocessing operation
            should be performed on `data`.  If the method can handle
            this type of processing, it should store the results as
            propery `what` in the `data` object and return. If the
            method can not handle that type of processing, it should
            call the method :py:meth:`_postprocess` of the super class.
        """
        if what == 'duration':
            data.add_attribute('duration', time.time() - data.start_,
                               batch=False)
        elif not hasattr(data, what):
            raise ValueError(f"Unknown property '{what}' for tool {self}")

    #
    # data processing
    #

    def apply(self, data: Data, *args, result: Union[str, Tuple[str]] = None,
              **kwargs) -> None:
        """Apply the tool to the given data object. Results are stored
        as data attributes.

        Arguments
        ---------
        data:
            The :py:class:`Data` object to which this :py:class:`Tool`
            should be applied.  Results will be stored as attributes
            of that object.
        result:
            The name(s) of the attribute(s) under which the results
            should be stored in the `data` object.  These names
            are prefixed by the name of this :py:class:`Tool`.
        """
        # FIXME[todo]: batch data ...
        if result is None:
            result = self.external_result
        LOG.debug("Applying tool %r to data %r, result=%s", self, data, result)
        values = self(data, *args, result=result, **kwargs)

        # Store the result(s) in the data object
        if isinstance(result, str):
            result, values = (result,), (values, )
        elif len(result) == 1:
            values = (values, )

        if result is not None:
            for name, value in zip(result, values):
                # FIXME[hack]: batch handling
                batch = (data.is_batch and isinstance(value, list) and
                         len(value) == len(data))
                self.add_data_attribute(data, name, value, batch=batch)

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

    def add_data_attributes(self, data: Data, names: Tuple[str],
                            values: Any) -> None:
        """Add a tool specific attribute to a data object.
        """
        for name, value in zip(names, values):
            self.add_data_attribute(data, name, value)

    def duration(self, data: Data) -> float:
        """Provide the duration (in seconds) the tool needed for processing
        the given data. This property is only available after processing,
        and only if the timer was activated.
        """
        return self.get_data_attribute(data, 'duration')


class BatchTool(Tool):
    # FIXME[question/todo]: what is a batch tool supposed to be?
    """BatchTool
    """

    def _do_preprocess(self, *args, internal: bool = False,
                       batch: bool = False,
                       result: Union[str, Tuple[str]] = None,
                       **kwargs) -> Data:
        data = self._do_preprocess(*args, internal, batch, result, **kwargs)
        return data

    def _do_postprocess(self, data: Data) -> Any:
        return super()._do_postprocess(data)


class IterativeTool(Tool):
    # pylint: disable=abstract-method
    """An iterative tool performs its operation as an iterative process.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._stop_all = False

    #
    # public interface
    #

    def _process(self, *arguments, **kwargs) -> Datalike:
        """Apply the tool to the given data. The input data will
        not be modified. The tool specific result will be returned.
        """
        result = None
        for result in self._steps(*arguments, **kwargs):
            pass
        return result

    def step(self, *args, **kwargs) -> Datalike:
        """Perform a single step of the iterative process.
        """
        data = self._do_preprocess(*args, *kwargs)
        internal_arguments = \
            self._get_attributes(data, self.internal_arguments)
        internal_result = self._step(*internal_arguments)
        self._add_attributes(data, self.internal_result,
                             internal_result, simplify=True)
        return self._do_postprocess(data)

    def steps(self, *args, steps: int = None,
              stop: Event = None, **kwargs) -> Iterator[Datalike]:
        """Iterate the steps of the iterative process, providing
        an intermediate result at every step.
        """
        data = self._do_preprocess(*args, **kwargs)
        internal_arguments = \
            self._get_attributes(data, self.internal_arguments)
        for internal_result in self._steps(*internal_arguments,
                                           steps=steps, stop=stop):
            self._add_attributes(data, self.internal_result,
                                 internal_result, simplify=True)
            yield self._do_postprocess(data)

    #
    # private (internal) methods
    #

    def _step(self, *arguments) -> Any:
        # to be implemented by subclasses
        raise NotImplementedError()

    def _steps(self, *arguments, steps: int = None,
               stop: Event = None) -> Iterator[Any]:
        data, arguments = (arguments[0:len(self.internal_result)],
                           arguments[len(self.internal_result):])
        while True:
            data = self._step(*data, *arguments)
            yield data
            if steps is not None:
                steps -= 1
                if steps < 0:
                    break
            if stop is not None:
                if stop.isSet():
                    break
            if self._stop_all:
                break
            # FIXME[todo]: data specific stop criteria
            if len(self.internal_result) == 0:
                data = ()
            elif len(self.internal_result) == 1:
                data = (data, )

    #
    # data processing
    #

    def apply(self, data: Data, *args, result: Union[str, Tuple[str]] = None,
              stepwise: bool = False, **kwargs) -> None:
        # pylint: disable=arguments-differ
        """Perform iterative processing on the data object.
        After each step, the relevant attributes of the data object
        are updated (and interested observers are informed.

        """
        if result is None:
            result = self.external_result
        if result is None:
            result = ()
        elif isinstance(result, str):
            result = (result, )

        if stepwise:
            for values in self.steps(data, *args, result=result, **kwargs):
                if len(self.internal_result) == 0:
                    values = ()
                elif len(self.internal_result) == 1:
                    values = (values, )
                for name, value in zip(result, values):
                    self.add_data_attribute(data, name, value)
        else:
            super().apply(self, data, *args, result=result, **kwargs)
