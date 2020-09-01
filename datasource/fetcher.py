"""Code related to fetching data from a datasource.

"""
# standard imports
import time
import logging
import threading

# toolbox imports
from base import busy, BusyObservable
from .data import Data
from .datasource import Datasource, Loop

# logging
LOG = logging.getLogger(__name__)


class Datafetcher(BusyObservable, Datasource.Observer,
                  method='datafetcher_changed',
                  changes=['state_changed', 'datasource_changed',
                           'data_changed']):
    """A :py:class:`Datafetcher` asynchronously fetches a
    :py:class:`Data` object from a :py:class:`Datasource` and
    stores it as local attribute.

    Every fetch operation creates a new :py:class:`Data` object.
    The old data object will not be altered and not deleted (as long
    as other parties still hold a reference). The ratio is, that fetched
    data may be used at other places and hence should not be changed.
    If that party is interested in up to date data, it should become
    an observer of this :py:class:`Datafetcher`.

    The :py:class:`Datafetcher` class provides a loop logic.

    Attributes
    ----------
    _loop_stop_event: threading.Event
        An Event signaling that the loop should stop.
        If not set, this means that the loop is currently running,
        if set, this means that the loop is currently not running (or at
        least supposed to stop running soon).
    _loop_interval: float
        The time in (fractions of a second) between two successive
        fetches when running the loop.
    """

    def __init__(self, datasource: Datasource = None,
                 loop_interval: float = 0.2, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = None
        self._datasource = datasource

        #
        # Loop specific variables
        #
        self._loop_interval = loop_interval

        # An event manages a flag that can be set to true with the set()
        # method and reset to false with the clear() method.
        self._loop_stop_event = threading.Event()
        self._loop_stop_event.set()

    @property
    def datasource(self) -> Datasource:
        """The underlying :py:class:`Datasource` object.
        """
        return self._datasource

    @datasource.setter
    def datasource(self, datasource: Datasource) -> None:
        """Set the underlying :py:class:`Datasource` object.
        """
        if datasource is self._datasource:
            return  # nothing has changed

        if self.looping:
            self.looping = False

        if self._datasource is not None:
            self.unobserve(self._datasource)
        self._datasource = datasource
        if datasource is not None:
            interests = Datasource.Change('busy_changed', 'state_changed')
            self.observe(datasource, interests)
        self._data = None
        self.change('data_changed', 'datasource_changed')
        if self.ready:
            self.fetch()

    @property
    def fetched(self) -> bool:
        """Check if data have been fetched and are now available.
        If so, :py:meth:`data` should deliver this data.
        """
        return self._data is not None

    @property
    def data(self) -> Data:
        """Get the last data that was fetched from this :py:class:`Datasource`.
        """
        # if not self.fetched:
        #    raise RuntimeError("No data has been fetched on Datasource "
        #                       f"{self.__class__.__name__}")
        return self._data

    @busy("fetching")
    def fetch(self, **kwargs):
        """Fetch a new data point from the datasource. After fetching
        has finished, which may take some time and may be run
        asynchronously, this data point will be available via the
        :py:meth:`data` property. Observers will be notified by
        by 'data_changed'.

        Arguments
        ---------
        Subclasses may specify additional arguments to describe
        how data should be fetched (e.g. indices, preprocessing, etc.).

        Changes
        -------
        data_changed
            Observers will be notified on data_changed, once fetching
            is completed and data are available via the :py:meth:`data`
            property.
        """
        if not self.ready:
            raise RuntimeError("Datafetcher is not ready.")

        with self.failure_manager():
            LOG.info("Datasouce.fetch(%s)", kwargs)
            self._data = self._datasource.get_data(**kwargs)
            self.change('data_changed')

    def unfetch(self):
        """Unfetch the currently fetched data. This will set the
        internal :py:class:`Data` object to `None` and inform the
        observers that `data_changed`.
        """
        LOG.info("Datasource.unfetch - data_changed")
        self._data = None
        self.change('data_changed')

    def datasource_changed(self, datasource: Datasource,
                           change: Datasource.Change) -> None:
        """React to a change of the observed :py:class:`Datasource`.
        """
        # changing state and business of the datasource also changes
        # our readiness - we will inform our observers
        self.change('state_changed')

        # If the datasource became prepared and we have no data yet,
        # we will start fetching
        if change.state_changed:
            if self.data is None and self.ready:
                self.fetch()
            elif self.data is not None and not self._datasource.prepared:
                self.unfetch()

    @property
    def ready(self) -> bool:
        """A flag indicating if the datafetcher is ready for use.
        """
        return (self._datasource is not None and
                self._datasource.prepared and not self._datasource.busy)

    #
    # Loop
    #

    @property
    def loopable(self) -> bool:
        """Check if the datafetcher can run a loop. This is true,
        if it currently is running a loop or it is ready and the
        underlying datasource allows for loops.
        """
        return (self.looping or
                (self.ready and isinstance(self._datasource, Loop)))

    @property
    def looping(self) -> bool:
        """Check if this datasource is currently looping.
        """
        return not self._loop_stop_event.is_set()

    @looping.setter
    def looping(self, looping: bool) -> None:
        """Set the looping state, that is start or stop a loop.
        """
        self.loop(looping)

    def loop(self, looping: bool = None, interval: float = 0.5):
        """Start or stop looping through the Datasource.
        This will fetch one data point after another.
        This is mainly intended to display live input like
        movies or webcam, but it can also be used for other Datasources.
        """
        if looping is not None and (looping == self.looping):
            return

        if self.looping:
            LOG.info("Stopping datasource loop")
            self.stop_loop()
        else:
            LOG.info("Starting datasource loop")
            if interval is not None:
                self._loop_interval = interval
            self.start_loop()
            self.run_loop()

    def start_loop(self):
        """Start an asynchronous loop cycle. This method will return
        immediately, running the loop cycle in a background thread.
        """
        # FIXME[todo]: does it really?
        if self._loop_stop_event.is_set():
            self._loop_stop_event.clear()

    def stop_loop(self):
        """Stop a currently running loop.
        """
        if not self._loop_stop_event.is_set():
            self._loop_stop_event.set()

    @busy("looping")
    def run_loop(self):
        """
        This method is intended to be invoked in its own Thread.
        """
        LOG.info(f"Loop[{self}]: start loop")
        self.change('state_changed')
        self._run_loop()
        self.change('state_changed')
        LOG.info(f"Loop[{self}]: end loop")

    def _run_loop(self):
        """Actual implementation of the loop. This method schould
        be overwritten by subclasses for adaptation.
        """

        while not self._loop_stop_event.is_set():
            # Fetch a data item
            last_time = time.time()
            LOG.debug(f"Loop: {self._loop_stop_event.is_set()} "
                      f"at {last_time:.4f}")
            self.fetch()

            # Now wait before fetching the next input
            sleep_time = last_time + self._loop_interval - time.time()
            if sleep_time > 0:
                self._loop_stop_event.wait(timeout=sleep_time)
            else:
                LOG.debug(f"Loop: late for {-sleep_time:.4f}s")
