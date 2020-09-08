"""A :py:class:`Processor` is intended for asynchronously using
a :py:class.`Tool`.
"""

# standard imports
import logging

# toolbox imports
from datasource import Data
from base import busy
from base.busy import BusyObservable
from .tool import Tool

# logging
LOG = logging.getLogger(__name__)


class Processor(BusyObservable, method='processor_changed',
                changes={'data_changed', 'process_finished'}):
    """A processor can be used to process data with a :py:class:`Tool`.
    The processor will hold a data object to which the tool is
    applied. The results are stored as new attributes of that data
    object.

    Processing can be done asynchronously. The processor is observable
    and will notify observers on the progress. All processors support
    the following notifications:

    data_changed:
        The data for processing was changed. The underlying tool
        has started processing the data, but it may not have finished
        yet. However, the :py:attr:`data` property will already
        provide the new :py:class:`Data` object.

    tool_changed:
        The :py:class:`Tool` to be used for processing was changed.

    process_finished:
        The processing has finished. The data object will now contain
        the results.

    """

    def __init__(self, tool: Tool = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = None
        self._next_data = None
        self._tool = tool
        LOG.info("New Processor created: %r", self)

    @property
    def data(self):
        """The :py:class:`Data` structure used by the processor.
        This data will contain results in specific attributes,
        depending on the tool used and its configuration.
        The data also includes the `duration` (in seconds).
        """
        return self._data

    @property
    def tool(self) -> Tool:
        """The :py:class:`Tool` applied by this :py:class:`Processor`."""
        return self._tool

    @tool.setter
    def tool(self, tool: Tool) -> None:
        """Change the :py:class:`Tool` to be applied by this
        :py:class:`Processor`.
        """
        LOG.info("Tool changed from %s to %s for processor %r",
                 self._tool, tool, self)
        if tool is not self._tool:
            self._tool = tool
            self.change('tool_changed')

    @property
    def ready(self) -> bool:
        """Check if this processor is ready for use.
        """
        return self._tool is not None and (self.processing or True)  # FIXME[todo/states]: self.tool.ready

    @property
    def processing(self) -> bool:
        """Check if this processor is currently processing data.
        """
        return self._next_data is not None

    def process(self, data: Data) -> None:
        """Run a data processing loop. This will set the detector
        into a busy state ("processing"), in which new input data
        are processed until no more new data are provided.
        If new data is given, before the previously provided data
        was processed, the previous data will be skipped.
        When processing one data item finishes, observers will
        receive a 'detection_finished' notification, and can obtain
        the data object including the detections via the
        :py:meth:`data` property. The detections can be accessed
        as a data attribute named by the :py:meth:`detector` property.

        The main motivation for this method is to process data from
        a data loop (like a webcam or a video) in real-time, always
        processing the most recent data available.
        """
        LOG.info("Processor for Tool '%s' processes data %r",
                 self.tool and self.tool.key, data)
        self._next_data = data
        if not self.busy:
            self._process()

    @busy("processing")  # FIXME[hack/bug]: if queueing is enabled, we are not really busy ...
    def _process(self):
        """The implementation of the process loop. This method
        is assumed to run in a background thread. It will
        check the property `_next_data` for fresh data and
        if present, it will hand (a copy) of this data to the
        detector and otherwise it will end the loop.

        The data object passed to the detector will also
        be stored as attribute (using the unique key of the detector
        as attribute name) in the original data passed to the
        detector. The results of the detector will be stored
        in detector data under the attribute `detections`.
        """
        while self._next_data is not None:

            data = self._next_data
            LOG.info("Processing next data (%r) with Tool %s.",
                     data, self._tool)
            self._data = data
            self.change(data_changed=True)
            with self.failure_manager(catch=True):
                self.tool.process(data)
                self.change(process_finished=True)
            if self._next_data is data:
                self._next_data = None
            LOG.info("Processing data (%r/%r) with Tool %s finished.",
                     data, self._data, self._tool)
    # FIXME[bug]: In case of an error, we may also cause some Qt error here:
    #   QObject::connect: Cannot queue arguments of type 'QTextBlock'
    #   (Make sure 'QTextBlock' is registered using qRegisterMetaType().)
    #   QObject::connect: Cannot queue arguments of type 'QTextCursor'
    #   (Make sure 'QTextCursor' is registered using qRegisterMetaType().)
    # This happens for example when the face panel is shown and a too
    # small image (244x244x3) is causing an error in the 'haar' detector.
    # The source of this messages is not clear to me yet (I have also
    # seen it at other locations ...)
