"""Code related to fetching data from a datasource.

"""
# standard imports
import time
import logging
import threading

# toolbox imports
from base import busy, BusyObservable
from .data import Data
from .datasource import Datasource, Loop, Snapshot, Random, Indexed

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
    _frames_per_second: float
        The number of frames to be fetched per second
        when running the loop.
    """

    def __init__(self, datasource: Datasource = None,
                 frames_per_seccond: float = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = None
        self._datasource = datasource

        #
        # Loop specific variables
        #
        self._frames_per_seccond = frames_per_seccond

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
            LOG.info("Datafetcher: fetch(%s) from %s",
                     kwargs, self._datasource)
            self._data = self._datasource.get_data(**kwargs)
            LOG.info("Datafetcher: fetched %s",
                     self._data and self._data.data is not None
                     and (self._data.is_batch or self._data.data.shape))
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
        """A flag indicating if the datafetcher is ready for use, meaning
        that a fetch operation should yield a result.
        """
        return (self._datasource is not None and
                self._datasource.prepared and
                not self._datasource.busy)

    @property
    def snapshotable(self) -> bool:
        """A flag indicating if the datafetcher is ready for taking
        a snapshot..
        """
        return isinstance(self._datasource, Snapshot)

    @property
    def randomable(self) -> bool:
        """A flag indicating if the datafetcher is ready for taking
        a snapshot..
        """
        return isinstance(self._datasource, Random)

    #
    # Loop
    #

    @property
    def loopable(self) -> bool:
        """Check if the datafetcher can run a loop. This is true,
        if it currently is running a loop or it is ready and the
        underlying datasource allows for loops.
        """
        return self.looping or isinstance(self._datasource, Loop)

    @property
    def looping(self) -> bool:
        """Check if this datafetcher is currently looping.
        """
        return isinstance(self._datasource, Loop) and self._datasource.looping

    @looping.setter
    def looping(self, looping: bool) -> None:
        """Set the looping state, that is start or stop a loop.
        """
        self.loop(looping)

    def loop(self, looping: bool = None, frames_per_second: float = None):
        """Start or stop looping through the Datasource.
        This will fetch one data point after another.
        This is mainly intended to display live input like
        movies or webcam, but it can also be used for other Datasources.
        """
        if looping is not None and (looping == self.looping):
            return

        if self.looping:
            LOG.info("Stopping datasource loop")
            self._datasource.stop_loop()
        else:
            LOG.info("Starting datasource loop")
            if frames_per_second is None:
                frames_per_second = self._frames_per_seccond
            if frames_per_second is None:
                frames_per_second = self._datasource.frames_per_second
            self._datasource.start_loop()
            self.run_loop(frames_per_second)

    @busy("looping")
    def run_loop(self, frames_per_second: float) -> None:
        """
        This method is intended to be invoked in its own Thread.
        """
        interval = 1. / frames_per_second
        start_time = time.time()
        LOG.info(f"Loop: start loop")
        self.change('state_changed')
        while self._datasource.looping:
            # Fetch a data item
            last_time = time.time()
            LOG.debug("Loop: %s at %.4f",
                      self._datasource.looping, last_time - start_time)
            self.fetch()

            # Now wait before fetching the next input
            sleep_time = last_time + interval - time.time()
            if sleep_time > 0:
                self._datasource.loop_stop_event.wait(timeout=sleep_time)
            else:
                LOG.debug(f"Loop: late for %.4fs", -sleep_time)
        self.change('state_changed')
        LOG.info(f"Loop: end loop")

    #
    # Public interface (convenience functions)
    #

    @property
    def indexable(self) -> bool:
        """Check if the datafetcher can be indexed.
        """
        return isinstance(self._datasource, Indexed)

    def _assert_indexable(self) -> None:
        if not self.indexable:
            raise TypeError("Datafetcher for datasource "
                            f"{type(self._datasource).__name__} "
                            "is not indexable.")

    @property
    def index(self):
        """The index of the currently fetched data item.
        """
        self._assert_indexable()
        if self._data is None or not hasattr(self._data, 'index'):
            raise RuntimeError("No index available.")
        return self._data.index

    def fetch_index(self, index: int, **kwargs) -> None:
        """This method should be implemented by subclasses that claim
        to be a py:meth:`Random` datasource.
        It should perform whatever is necessary to fetch a random
        element from the dataset.
        """
        self._assert_indexable()
        self.fetch(index=index, **kwargs)

    def fetch_next(self, cycle: bool = True, **kwargs) -> None:
        """Fetch the next entry. In a :py:class:`Indexed` datasource
        we can simply increase the index by one.
        """
        self._assert_indexable()
        current_index = self.index
        next_index = (current_index + 1) if current_index < len(self) else 0
        self.fetch(index=next_index, **kwargs)

    def fetch_prev(self, cycle: bool = True, **kwargs) -> None:
        """Fetch the previous entry. In a :py:class:`Indexed` datasource
        we can simply decrease the index by one.
        """
        self._assert_indexable()
        current_index = self.index
        next_index = (current_index - 1) if current_index > 0 else len(self)
        self.fetch(index=next_index, **kwargs)

    def fetch_first(self, **kwargs) -> None:
        """Fetch the first entry of this :py:class:`Indexed` datasource.
        This is equivalent to fetching index 0.
        """
        self._assert_indexable()
        self.fetch(index=0, **kwargs)

    def fetch_last(self, **kwargs) -> None:
        """Fetch the last entry of this :py:class:`Indexed` datasource.
        This is equivalent to fetching the element with index `len(self)-1`.
        """
        self._assert_indexable()
        self.fetch(index=len(self)-1, **kwargs)
