"""A :py:class:`Worker` is intended for (asynchronously) using
a :py:class.`Tool`.
"""

# standard imports
import logging

# toolbox imports
from base import busy
from base.busy import BusyObservable
from ..base.data import Data
from .tool import Tool

# logging
LOG = logging.getLogger(__name__)


class Worker(BusyObservable, method='worker_changed',
                changes={'tool_changed', 'data_changed', 'work_finished'}):
    """A worker can be used to work on data using a :py:class:`Tool`.
    The worker will hold a data object to which the tool is
    applied. The results are stored as new attributes of that data
    object.

    Working can be done asynchronously. The worker is observable
    and will notify observers on the progress. All workers support
    the following notifications:

    data_changed:
        The data for working was changed. The underlying tool
        has started working on the data, but it may not have finished
        yet. However, the :py:attr:`data` property will already
        provide the new :py:class:`Data` object.

    tool_changed:
        The :py:class:`Tool` to be used for working was changed.

    work_finished:
        The work was finished. The data object will now contain
        the results.

    """

    def __init__(self, tool: Tool = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = None
        self._next_data = None
        self._tool = tool
        LOG.info("New Worker created: %r", self)

    @property
    def data(self):
        """The :py:class:`Data` structure used by the worker.
        This data will contain results in specific attributes,
        depending on the tool used and its configuration.
        The data also includes the `duration` (in seconds).
        """
        return self._data

    @property
    def tool(self) -> Tool:
        """The :py:class:`Tool` applied by this :py:class:`Worker`."""
        return self._tool

    @tool.setter
    def tool(self, tool: Tool) -> None:
        """Change the :py:class:`Tool` to be applied by this
        :py:class:`Worker`.
        """
        LOG.info("Tool changed from %s to %s for worker %r",
                 self._tool, tool, self)
        if tool is not self._tool:
            self._tool = tool
            self.change('tool_changed')

    @property
    def ready(self) -> bool:
        """Check if this worker is ready for use.
        """
        # FIXME[todo/states]: self.tool.ready
        return (self.working or
                (self._tool is not None and
                 self._tool.prepared and
                 (not isinstance(self.tool, BusyObservable) or
                  not self.tool.busy)))

    @property
    def working(self) -> bool:
        """Check if this worker is currently working on data.
        """
        return self._next_data is not None

    def work(self, data: Data, **kwargs) -> None:
        """Run a data work loop. This will set the Worker
        into a busy state ("working"), in which new input data
        are worked on until no more new data are provided.
        If new data is given, before the previously provided data
        was done, the previous data will be skipped.
        When working one data item finishes, observers will
        receive a 'work_finished' notification, and can obtain
        the data object including the results via the
        :py:meth:`data` property. The results can be accessed
        as tool specific data attributes.

        The main motivation for this method is to work on data from
        a data loop (like a webcam or a video) in real-time, always
        working on the most recent data available.
        """
        LOG.info("Worker for Tool '%s' works on data %r",
                 self.tool and self.tool.key, data)
        self._next_data = data
        if not self.busy:
            self._work(**kwargs)

    @busy("working")  # FIXME[hack/bug]: if queueing is enabled, we are not really busy ...
    def _work(self, **kwargs):
        """The implementation of the work loop. This method
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
            LOG.info("Working on next data (%r) with Tool %s.",
                     data, self._tool)
            self._data = data
            self.change(data_changed=True)
            with self.failure_manager(catch=True):
                result = self.tool.result + ('duration', )
                self.tool.apply(data, result=result, **kwargs)
                self.change(work_finished=True)
            if self._next_data is data:
                self._next_data = None
            LOG.info("Working on data (%r/%r) with Tool %s finished.",
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
