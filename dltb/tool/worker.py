"""A :py:class:`Worker` is intended for (asynchronously) using
a :py:class.`Tool`.
"""

# standard imports
from typing import Optional
import logging

# toolbox imports
from ..base.busy import BusyObservable, busy
from ..base.data import Data
from .tool import Tool

# logging
LOG = logging.getLogger(__name__)


class Worker(BusyObservable, Tool.Observer, method='worker_changed',
             changes={'tool_changed', 'data_changed', 'worker_changed',
                      'work_step', 'work_finished'}):
    """A worker can be used to work on data using a :py:class:`Tool`.
    The worker will hold a data object to which the tool is
    applied. The results are stored as new attributes of that data
    object.

    Working can be done asynchronously. The worker is observable
    and will notify observers on the progress. All workers post
    the following notifications:

    data_changed:
        The data for working was changed. The underlying tool
        has started working on the data, but it may not have finished
        yet. However, the :py:attr:`data` property will already
        provide the new :py:class:`Data` object.

    tool_changed:
        The :py:class:`Tool` to be used for working was changed.

    worker_changed:
        The configuration of the worker changed.

    work_step:
        A work step was done. This will only happen, if the tool
        is an :py:class`IterativeTool` and that `stepwise=True`
        argument has been given. More fine grained tracking
        of the work process can be obtained by observing the
        :py:class:`Data` object itself.

    work_finished:
        The work was finished. The data object will now contain
        the results.

    """
    _tool: Optional[Tool] = None

    def __init__(self, tool: Tool = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = None
        self._next_data = None
        self._next_kwargs = None
        self._have_more_data = False

        self.tool = tool
        LOG.info("New Worker created: %r (tool=%s)", self, tool)

    @property
    def data(self):
        """The :py:class:`Data` structure used by the worker.
        This data will contain results in specific attributes,
        depending on the tool used and its configuration.
        The data also includes the `duration` (in seconds).
        """
        return self._data

    @property
    def tool(self) -> Optional[Tool]:
        """The :py:class:`Tool` applied by this :py:class:`Worker`."""
        return self._tool

    @tool.setter
    def tool(self, tool: Optional[Tool]) -> None:
        """Change the :py:class:`Tool` to be applied by this
        :py:class:`Worker`.
        """
        LOG.info("Tool changed from %s to %s for worker %r",
                 self._tool, tool, self)
        if tool is not self._tool:
            if self._tool is not None:
                self.unobserve(self._tool)
            self._set_tool(tool)
            if tool is not None:
                self.observe(tool, Tool.Change('tool_changed',
                                               'state_changed'))
                print("!!! tool.debug() BEGIN")
                tool.debug()
                print("!!! tool.debug() END")
            self.change('tool_changed')
            if self._data is not None:
                self.work(self._data)  # rework the current data with new tool

    def _set_tool(self, tool: Optional[Tool]) -> None:
        self._tool = tool

    @property
    def ready(self) -> bool:
        """Check if this worker is ready for use.
        """
        return self._ready()

    def _ready(self) -> bool:
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

        Arguments
        ---------
        data:
            The `Data` the tool should be applied to. May be `None`.
        """
        LOG.info("Worker for Tool '%s' (ready=%s, prepared=%s) "
                 "works on data: %r",
                 self.tool and self.tool.key, self.ready,
                 self.tool is not None and self.tool.prepared, data)
        self._have_more_data = True
        self._next_data = data
        self._next_kwargs = kwargs
        if self.ready and not self.busy:
            self._work_loop()

    @busy("working")
    # FIXME[hack/bug]: if queueing is enabled, we are not really busy ...
    # (that is we are busy, but nevertheless accepting more work)
    def _work_loop(self, stepwise: bool = False):
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
        tool = self.tool
        if tool is None:
            return
        while self._have_more_data:
            data = self._next_data
            kwargs = self._next_kwargs
            self._next_data = None
            self._have_more_data = False  # allow providing new data
            LOG.info("Working on next data (%r) with Tool %s (rework=%s).",
                     data, tool, data is self._data)
            if data is not self._data:
                self._data = data
                self.change(data_changed=True)
            if data is None:
                continue
            with self.failure_manager(catch=True):
                result = tool.external_result + ('duration', )
                LOG.debug("Worker %r applying tool %r on data %r "
                          "with result %s.", self, tool, data, result)
                if stepwise:
                    for values in tool.steps(data, result=result, **kwargs):
                        tool.add_data_attributes(data, result, values)
                        self.change(work_step=True)
                else:
                    self._work(data, result=result, **kwargs)
                self.change(work_finished=True)
            LOG.info("Working on data (%r/%r) with Tool %s finished.",
                     data, self._data, tool)
    # FIXME[bug]: In case of an error, we may also cause some Qt error here:
    #   QObject::connect: Cannot queue arguments of type 'QTextBlock'
    #   (Make sure 'QTextBlock' is registered using qRegisterMetaType().)
    #   QObject::connect: Cannot queue arguments of type 'QTextCursor'
    #   (Make sure 'QTextCursor' is registered using qRegisterMetaType().)
    # This happens for example when the face panel is shown and a too
    # small image (244x244x3) is causing an error in the 'haar' detector.
    # The source of this messages is not clear to me yet (I have also
    # seen it at other locations ...)

    def _work(self, data: Data, **kwargs) -> None:
        """This method does the actual work (applies the
        :py:class:`Tool` to the :py:class:`Data`).

        This method may be overwritten by subclasses, to perform
        worker specific argument processing.

        Arguments
        ---------
        data:
            The :py:class:`Data` to work on.  This will usually be
            the current `data` object of this `Worker`. The results
            of applying the :py:class:`Tool` are stored as attributes
            of that data object.
        kwargs:
            Additional keyword arguments, which may be general `Tool`
            arguments or `Worker` or `Tool` specific arguments.
        """
        tool = self.tool
        if tool is None:
            raise RuntimeError("Worker has lost her Tool.")
        tool.apply(data, **kwargs)

    #
    # Tool.Observer
    #

    def tool_changed(self, _tool: Tool, _info: Tool.Change) -> None:
        """React to a change of the :py:class:`Tool` by reworking
        the current :py:class:`Data`.
        """

        self.change('tool_changed')

        if self._data is None:
            return  # nothing to do ...

        # FIXME[todo]: what tool changes would require a recomputation
        #   - info.busy_change should not do this ...
        # if info.state_changed
        # if self._data is not None:
        #    self.work(self._data)
