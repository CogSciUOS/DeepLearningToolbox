"""Code related to fetching data from a datasource.

"""
# standard imports
import logging

# toolbox imports
from base import busy, BusyObservable
from .data import Data
from .datasource import Datasource

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
    """

    def __init__(self, datasource: Datasource = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = None
        self._datasource = datasource

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
        if self._datasource is not None:
            self.unobserve(self._datasource)
        self._datasource = datasource
        if datasource is not None:
            interests = Datasource.Change('busy_changed', 'state_changed')
            self.observe(datasource, interests)
        self._data = None
        self.debug()
        self.change('data_changed', 'datasource_changed', debug=True)
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
            raise RuntimeError("No Datafetcher is not ready.")

        with self.failure_manager():
            LOG.info("Datasouce.fetch(%s)", kwargs)
            print(f"fetching: {kwargs}")
            self._data = self._datasource.get_data(**kwargs)
            print(f"fetched: {self._data}")
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
        # FIXME[hack]
        self.change('state_changed')

    @property
    def ready(self) -> bool:
        return (self._datasource is not None and
                self._datasource.prepared and not self._datasource.busy)
