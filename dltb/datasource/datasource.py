# FIXME[todo]:
# * finish refactoring (get & fetch)
#   - label
#   - batch
# * documentation
# * name and description
# * availability and download/install
# FIXME[warning]: doc/source/datasource.rst 6
# WARNING: duplicate object description of datasource.datasource,
# other instance in datasource, use :noindex: for one of them
#
# FIXME[old]: there a still a lot references to "fetch" here which should
# be removed (also from other files in the datasource directory)
#
#
# FIXME[todo]: several points are still to be done:
#  - description: we need a framework that allows to display description
#    to the user, in different formats (textual or in GUI) and at
#    a different level of detail.

# FIMXE[old]: some parts in the doc string seem to be somewhat outdated:
"""
.. moduleauthor:: Ulf Krumnack

.. module:: datasource.datasource


This module contains abstract definitions of :py:class:`Datasource`\\ s.
All basic functionality a datasource may have should be specified in
in this module.

A :py:class:`Datasource` realizes only synchronous accesse to the data.
Asynchronous access with notification is realized by the
:py:class:`Datafetcher`.

Other modules implementing access to :py:class:`Datasource`\\ s
should rely on the APIs defined here. Such modules include.
* qtgui.widgets.datasource


Implementation Guides
=====================

This file demonstrate how :py:class:`Datasource` should be subclassed
to provide more forms of data access. The new `get_x` method should be
introduced in an abstract class (here we add `Indexed`, `Snapshot`,
and `Random`). When implementing a Datasource that supports this
`get_x` method, it should be derived from that base class.

* Changing the order of base classes should not affect implementation
  (e.g. class MyData(Labeled, Timestamped) should not
  Hence use dict instead of tuple as return type for get_data
  (named.)

# FIMXE[old]: the following seems to be somewhat outdated:

Abstract classes 1
------------------
Provide a new data access mechanism x (like index, random, ...)
* may implement a preparation mechanism
* provide public API method get_x()
* overwrite private method _get(meta, done:bool, x=x):
  - should invoke self._get_x() or self._get_x_batch()
  - should call super()._get(done=True)
  - should provide dummy implementation of _get_x_batch()
  problem: _get_batch() should be more efficient than
     repeated _get()
     -> _get_x should return data instead of setting meta properties
        -> but how to deal with additional information like label?

Worker classes
--------------
* should implement _get_x() (and optionally _get_x_batch()) methods
  according to their type (usually it is sufficient to look at the
  direct superclass)
* the result of the _get_x() method depends on the type:
  - simple classes just return data
  - labeled classes would return pairs (data, label)
    (or better: dict {data: data, label: label})
* the result of the _get_batch_x() method also depends on the type:
  - it should return the same dict as _get_x(), but with values
    being lists or arrays of same length

"""

# standard imports
from typing import Union, Tuple, Iterator, Any, Dict, Callable, AbstractSet
from typing import Optional, Iterable, List
from abc import abstractmethod
import time
import random
import logging
import threading
import itertools
import collections.abc

# third party imports
import numpy as np

# toolbox imports
from ..base.data import Data
from ..base.image import Image
from ..base.sound import Sound
from ..base.implementation import Implementable
from ..base.register import RegisterClass
from ..base.fail import FailableObservable
from ..base import Preparable
from ..util.image import imread
from ..util.itertools import SizedGenerator

# logging
LOG = logging.getLogger(__name__)


Attributes = Union[str, Tuple[str, ...]]


class Datasource(Preparable, FailableObservable, Implementable, # ABC,
                 method='datasource_changed',
                 changes={'state_changed'},
                 metaclass=RegisterClass):
    # For some (unknown) reason, pylint currently complains when
    # not overriding an abstract method (load_datapoint_from_file)
    # pylint: disable=abstract-method
    """.. :py:class:: Datasource

    An abstract base class for different types of data sources.

    There are different APIs for navigating in a datasource, depending
    on what that datasource supports:

    array-like navigation: individual elements of a data source can be
    accessed using an array-like notation.

    random selection: select a random element from the dataset.

    Loaders
    -------
    It is possible to specify different (thirdparty) loaders for
    loading data from filenames or URLs.

    >>> datasouce.set_loader(loader=imread, kind='array', auto=True)
    >>> datasouce.set_loader(loader=PIL.Image.open, kind='pil', auto=True)
    >>> datasouce.set_loader(auto=False)

    Attributes
    ----------
    _description : str
        Short description of the dataset

    The data class can be overwritten by subclasses:

    _data_class: type

    Data loaders can be set globally, but they may be overwritten
    by subclasses or instances if desired.

    _loaders: Dict[str, Callable[[str], Any]] = {}
    _loader: Callable[[str], Any] = None
    _loader_kind: str = None
    _loader_auto: bool = True


    Changes
    -------
    state_changed:
        The state of the Datasource has changed (e.g. data were downloaded,
        loaded/unloaded, unprepared/prepared, etc.)

    """

    # FIXME[todo]: The datasource may also be in a failed state,
    # e.g. when trying to get images from a webcam which is disconnected.
    # maybe introduce a StatefulObservable with the following states:
    #  - prepared/unprepared
    #  - busy
    #  - failed
    # Adding a change: state_changed
    # Adding a property: state:
    # Adding a property: state_description: str
    # But: some states may be more permanent than others:
    #  - e.g. prepared may be interrupted by busy states, e.g. when
    #    getting data
    #  - failed may indicate that preparation failed (the resource
    #    will be in an unprepared state) or that getting failed
    #    (in which case it may still remain in the prepared state)

    _id: str = None
    _metadata = None

    _data_class: type = Data

    _loaders: Dict[str, Callable[[str], Any]] = {}
    _loader: Callable[[str], Any] = None
    _loader_kind: str = None
    _loader_auto: bool = True

    _postprocessors: List[Callable] = None
    
    def __init__(self, description: str = None, **kwargs) -> None:
        """Create a new Datasource.

        Parameters
        ----------
        description :   str
                        Description of the dataset
        """
        super().__init__(**kwargs)
        self._description = (self.__class__.__name__ if description is None
                             else description)
        self._data = None
        self._postprocessors = []

    def __call__(self, attributes: Optional[Attributes] = None,
                 batch_size: Optional[int] = None, **kwargs) -> Iterable:
        data_iterator = self.items(**kwargs) if batch_size is None else \
            self.batches(size=batch_size, **kwargs)
        for data in data_iterator:
            if attributes is None:
                yield data
            elif isinstance(attributes, str):
                yield getattr(data, attributes)
            else:
                yield tuple(getattr(data, attr) for attr in attributes)

    @property
    def name(self):
        """Get the name of this :py:class:`Datasource` to be
        presented to the user.
        """
        return f"{self.key}"

    #
    # Data
    #

    def get_data(self, batch: int = None, **kwargs) -> Data:
        """Get data from this :py:class:`Datasource`.

        Arguments
        ---------
        batch:
            Either `None` for getting a single (non-batch) data object,
            or a positive integer specifying the batch size.
        """
        data = self._data_class(datasource=self, batch=batch)
        LOG.debug("Datasource[%s].get_data(%s)", self, kwargs)
        data.add_attribute('datasource', self)
        data.add_attribute('datasource_argument')
        data.add_attribute('datasource_value')
        self._get_meta(data, **kwargs)
        if data.is_batch:
            self._get_batch(data, **kwargs)
            LOG.debug("Datasource[%s].get_batch(): %s", self, data)
        else:
            self._get_data(data, **kwargs)
            LOG.debug("Datasource[%s].get_data(): %s", self, data)
        self._postprocess(data)
        return data

    def _postprocess(self, data: Data) -> None:
        """Apply postprocessors to a given :py:class:`Data` object.
        """
        for postprocess in self._postprocessors:
            postprocess(data)

    def add_postprocessor(self, postprocessor: Callable[[Data], None]) -> None:
        """Add a postprocessor to this :py:class:`Datasource`.
        All :py:class:`Data` retrieved from this `Datasource`
        will be postprocessed by this function.
        """
        self._postprocessors.append(postprocessor)

    def _get_meta(self, data: Data, **_kwargs) -> None:
        """Enrich the :py:class:`Data` object with meta information.
        This method is called before :py:meth:`_get_data` or
        :py:meth:`_get_batch` is called.  It is intended to add
        data attributes to the data object, either directly providing
        a value, or preparing them so that values can be added by
        the `_get_data` or `_get_batch` methods.

        Sublasses that can provide meta information should overwrite
        this method and call `super()._get_meta()` in order to
        accumulate all meta information in the :py:class:`Data`
        object.
        """

    def _get_data(self, data: Data, **kwargs) -> None:
        """Get data from this :py:class:`Datasource`.
        """
        if not data:
            LOG.debug("Datasource[%r]._get_default(kwargs=%r): %r",
                      type(self), kwargs, data)
            self._get_default(data, **kwargs)
        LOG.debug("Datasource[%r]._get_data(kwargs=%r): %r",
                  type(self), kwargs, data)

    def _get_batch(self, data: Data, **kwargs) -> None:
        """Get a batch of data from this :py:class:`Datasource`.
        """
        if not data:
            # initialize all batch attributes
            data.initialize_attributes(batch=True)
            # get data for each item of the batch
            for item in data:
                self._get_data(item, **kwargs)

    @abstractmethod
    def _get_default(self, data: Data, **kwargs) -> None:
        """The default method for getting data from this
        :py:class:`Datasource`. This method should be implemented by
        all subclases of :py:class:`Datasource`.
        """

    #
    # Items (singe data points)
    #

    def items(self, loop: bool = False, **kwargs) -> Iterator[Data]:
        """Iterate over the data of this :py:class:`Datasoruce`.
        """
        if loop:
            # run an (infinite) loop, getting randomly selected data
            # from this datasource (with repetition)
            while True:
                yield self.get_data(**kwargs)
        else:
            # enumerate the data of this Datasource (without repetition)
            for index in range(0, len(self)):
                yield self.get_data(index=index, **kwargs)

    #
    # Batches (groups of data points)
    #

    def batches(self, size: int, loop: bool = False, random: bool = False,
                epochs: int = 1, **kwargs) -> Iterator[Data]:
        """Batchwise iterate the data of this :py:class:`Datasoruce`.

        Arguments
        ---------
        size:
            The batch size, that is the (maximal) number of datapoints
            per batch (the last batch may have a smaller batch size).
        loop:
            A flag indicating if an (infinite) loop providing
            (random) datapoints should be run.
        epochs:
            The special value -1 can be used to run an infinite
            loop.
        """

        #if random:
        #    # run an (infinite) loop, getting randomly selected data
        #    # from this datasource (with repetition)
        #    while True:
        #        yield self.get_data(batch=size, **kwargs)
        #else:
        return SizedGenerator(self._batches(size, loop, epochs, **kwargs),
                              (len(self) // size) + 1)

    def _batches(self, size: int, loop: bool,
                 epochs: int, **kwargs) -> Iterator[Data]:

        if loop or epochs == -1:
            # run an (infinite) loop, getting selected data
            # from this datasource (with repetition)
            epochs = itertools.count()
        else:
            # enumerate (batchwise) the data of this Datasource
            # (without repetition)
            epochs = range(epochs)

        for epoch in epochs:
            for index in range(0, len(self), size):
                # there may not be enough data to fill the last batch:
                # hence we will reduce the last batch size
                batch_size = min(size, len(self) - index)
                yield self.get_data(batch=batch_size, index=index,
                                    **kwargs)

    #
    # Description
    #

    @property
    def description(self) -> str:
        """Get a textual description of this
        :py:class:`Ddatasource`.
        """
        return self._get_description()

    def _get_description(self) -> str:
        """Provide a description of the Datasource or one of its
        elements.
        """
        return self._description

    #
    # Data loader
    #

    def set_loader(self, loader: Callable[[str], Any] = None,
                   kind: str = None, auto: bool = None) -> None:
        """Set the data loader to be used by this :py:class:`Datasource`.

        Arguments
        ---------
        loader: Callable[[str], Any]
            A function invoked to load data from a filename (or URL).
            The default (`None`) signals that the loader should be
            determined from the `kind` argument (if given). If no
            other parameter is given, data loading will be disabled
            completely.
        kind: str
            The kind of data delivered by the loader ('array' for
            numpy arrays, or 'pil' for PIL images). This is at the same
            time the attribute name by which the data is stored in the
            :py:class:`Data` object.
        auto: bool
            This flag indicates if data should be automatically loaded
            if possible (if a filename or url is available).
        """
        if loader is None:
            if kind is None and auto is None:
                self._loader = None
            elif kind is not None:
                self._loader = self._loader_for_kind(kind)
        else:
            if kind is None:
                raise ValueError("No 'kind' was specified for data loader.")
            self._loader = loader

        if kind is not None:
            self._loader_kind = kind

        if auto is not None:
            self._loader_auto = auto

    @classmethod
    def add_loader(cls, kind: str, loader: Callable[[str], Any]) -> None:
        """Add a new data loader to this Datasource.
        """
        cls._loaders[kind] = loader

    def _loader_for_kind(self, kind: str) -> Callable[[str], Any]:
        """Automatically determine a dataloader for a given kind of
        data. Additional dataloaders can be added by calling
        :py:meth:`add_loader`.
        """
        return self._loaders[kind]

    def load_datapoint_from_file(self, filename: str) -> np.ndarray:
        """Load a single datapoint from a file.
        This method should be implemented by subclasses.

        Arguments
        ---------
        filename: str
            The absolute filename from which the data should be loaded.

        Returns
        ------
        datapoint
            The datapoint loaded from file.
        """
        if self._loader is None:
            raise ValueError("No data loader was specified for Datasource "
                             f"{self} to load file '{filename}'.")
        try:
            return self._loader(filename)
        except ValueError:
            # FIXME[bug]: error reporting (on console) does not work
            print(f"{type(self)}: Error reading image file '{filename}'")
            LOG.error("Error reading image file '%s'", filename)
            raise


# Imagelike is intended to be everything that can be used as
# an image.
#
# Imagesourcelike:
#    An already initialized Imagesource object
# str:
#    The key of an registered Imagesource.
Imagesourcelike = Union['Imagesource', str]


class Imagesource(Datasource):
    # pylint: disable=abstract-method
    """A datasource providing images.

    FIXME[todo]:
    The image source can be set to a specific size. Images will then
    be provided in that size.  Parameters to control the resizing
    process correspond to those that can be passed to the image
    loading functions.
    """

    _data_class: type = Image

    #
    _loaders: Dict[str, Callable[[str], Any]] = {'array': staticmethod(imread)}
    _loader: Callable[[str], Any] = staticmethod(imread)
    _loader_kind: str = 'array'

    # FIXME[todo]:
    _size: Tuple[int, int] = None

    @classmethod
    def as_imagesource(cls, source: Imagesourcelike) -> 'Imagesource':
        """Get an :py:class:`Imagesource` from an
        :py:class:`Imagesourcelike` object.
        """
        if isinstance(source, cls):  # cls is supposed to be Imagesource
            return source
        if isinstance(source, str):
            return Datasource[source]
        raise TypeError("Imagesourcelike object has unexpected type: "
                        f"{type(source)}")

    def __init__(self, shape: Tuple = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.shape = shape

    def _get_meta(self, data: Data, **kwargs) -> None:
        """Set the image metdata for the :py:class:`data`.
        """
        super()._get_meta(data, **kwargs)
        shape = kwargs.get('shape', self.shape)
        data.add_attribute('shape', shape, batch=shape is None)

    def _get_data(self, data: Data, **kwargs) -> None:
        """Get data from this :py:class:`Datasource`.
        """
        super()._get_data(data, **kwargs)
        if data and not self.shape:
            data.shape = data.array.shape


Soundsourcelike = Union['Soundsource', str]


class Soundsource(Datasource):
    # pylint: disable=abstract-method
    """A datasource providing :py:class:`Sound` objects

    """

    _data_class: type = Sound

    #
    _loaders: Dict[str, Callable[[str], Any]] = {'array': staticmethod(imread)}
    _loader: Callable[[str], Any] = staticmethod(imread)
    _loader_kind: str = 'array'

    # FIXME[todo]:
    _samplerate: float = None

    @classmethod
    def as_soundsource(cls, source: Soundsourcelike) -> 'Soundsource':
        """Get an :py:class:`Soundsource` from an
        :py:class:`Soundsourcelike` object.
        """
        if isinstance(source, cls):  # cls is supposed to be Soundsource
            return source
        if isinstance(source, str):
            return Datasource[source]
        raise TypeError("Soundsourcelike object has unexpected type: "
                        f"{type(source)}")

    def __init__(self, samplerate: Union[int, float] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.samplerate = float(samplerate)

    def _get_meta(self, data: Data, **kwargs) -> None:
        """Set the sound metdata for the :py:class:`data`.
        """
        super()._get_meta(data, **kwargs)
        # FIXME[todo]:
        # samplerate = kwargs.get('samplerate', self.samplerate)
        # data.add_attribute('samplerate', samplerate, batch=samplerate is None)

    def _get_data(self, data: Data, **kwargs) -> None:
        """Get data from this :py:class:`Datasource`.
        """
        super()._get_data(data, **kwargs)
        # FIXME[todo]:
        # if data and not self.samplerate:
        #     data.shape = data.array.shape

    # FIXME[todo]: change the interface from `load_datapoint_from_file`
    # (which returns an array) to `load_data_from_file` (which)
    # actively fills a data object from a filename, provided as attribute
    # to he data object)
    def load_data_from_file(self, data: Sound) -> np.ndarray:
        """
        """


class Sectioned(Datasource):
    # pylint: disable=too-few-public-methods,abstract-method
    """A :py:class:`Datasource` with multiple sections (like `train`, `val`,
    `test`, etc.). A subclass of :py:class:`Sectioned` may provide a
    set of section labels via the `sections` class parameter.

    Class parameters
    ----------------
    sections: AbstractSet[str]
    """

    def __init_subclass__(cls, sections: AbstractSet[str] = None,
                          **kwargs) -> None:
        # pylint: disable=arguments-differ
        super().__init_subclass__(**kwargs)
        if sections is not None:
            cls.sections = sections

    def __init__(self, section: str = None, **kwargs) -> None:
        if section not in self.sections:
            raise ValueError(f"Invalid section '{section}'. "
                             f"Should be from {self.sections}.")
        super().__init__(**kwargs)
        self._section = section

    @property
    def section(self) -> str:
        """The section this :py:class:`Sectioned` object represents.
        """
        return self._section

    @section.setter
    def section(self, section: str) -> None:
        """Set the section this :py:class:`Sectioned` object represents.
        """
        self._section = section


class Livesource(Datasource):
    """A :py:class:`Livesource` can provide live data. Live data can
    be different on every retrieval. Typical live sources are
    video streams like webcams.

    A :py:class:`Livesource` can be put in online mode, meaning that
    it spawns a background thread that continously provides new data
    (corresponds to pressing play on a media player). In that mode, a
    call to :py:meth:`get_data `will simply return a :py:class:`Data`
    object for the current data.

    Attributes
    ----------
    loop_stop_event: threading.Event
        An Event signaling that the loop should stop.
        If not set, this means that the loop is currently running,
        if set, this means that the loop is currently not running (or at
        least supposed to stop running soon).
    """


    def __init__(self, loop_interval: float = 0.2, **kwargs) -> None:
        """

        Arguments
        ---------
        """
        super().__init__(**kwargs)
        self._loop_interval = loop_interval  # not really used

        # An event manages a flag that can be set to true with the set()
        # method and reset to false with the clear() method.
        self.loop_stop_event = threading.Event()
        # The flag is initially false, so we have to set it to True.
        self.loop_stop_event.set()

    def _unprepare(self):
        """Unpreparing a :py:class:`Loop` includes stopping the loop
        if running.
        """
        self.stop_loop()
        super()._unprepare()

    @property
    def frames_per_second(self) -> float:
        """Frames per second when looping.
        """
        return 10.

    @property
    def looping(self):
        """Check if this datasource is currently looping.
        """
        return not self.loop_stop_event.is_set()

    def start_loop(self):
        """Start an asynchronous loop cycle. This method will return
        immediately, running the loop cycle in a background thread.
        """
        if self.loop_stop_event.is_set():
            self.loop_stop_event.clear()

    def stop_loop(self):
        """Stop a currently running loop.
        """
        if not self.loop_stop_event.is_set():
            self.loop_stop_event.set()

    #
    # FIXME[old]: snapshot
    #

    def snapshot(self, **kwargs) -> Data:
        """Create a snapshot (synchronously).
        """
        return self.get_data(snapshot=True, **kwargs)

    #
    # private methods
    #

    def _get_meta(self, data: Data, snapshot: bool = None, **kwargs) -> None:
        # pylint: disable=arguments-differ
        """Getting data from :py:class:`Snapshot` datasource with
        the `snapshot` argument set to `True` will add a `timestamp`
        (batch) attribute to the :py:class:`Data` object.
        """
        super()._get_meta(data, **kwargs)
        if snapshot is not None and not data.datasource_argument:
            data.datasource_argument = 'snapshot'
            data.datasource_value = snapshot
        data.add_attribute('time', batch=True)

    def _get_data(self, data: Data, snapshot: bool = None, **kwargs) -> None:
        # pylint: disable=arguments-differ
        """A version of :py:meth:`_get_data` that allows for an
        taking a snapshot.

        Arguments
        ---------
        snapshot: bool
            A flag indicating that a snapshot should be taken.
        """
        if snapshot is not None and data.datasource_argument == 'snapshot':
            self._get_snapshot(data, snapshot, **kwargs)
        super()._get_data(data, **kwargs)

    @abstractmethod
    def _get_snapshot(self, data: Data, snapshot: bool = None,
                      **kwargs) -> None:
        """This method is intended to by overwritten by subclasses.
        Such an implementation should call
        `super()._get_shapshot(data, snapshot)` and than fill the
        actual data into the `Data` object.
        """
        LOG.debug("Datasource[%s]._get_snapshot(%s, %r): %s",
                  self, snapshot, kwargs, data)
        data.time = time.time()

    def _get_default(self, data: Data, **kwargs) -> None:
        """The default method of getting data with a :py:class:`Snapshot`
        object is to take a snapshot.
        """
        self._get_snapshot(data, **kwargs)


class Loop(Datasource):
    # pylint: disable=abstract-method
    # FIXME[todo]: make this a more general interface
    # can be derived from `Stateful`, adding a state `looping`
    """The :py:class:`Loop` class provides a loop logic.

    FIXME[design]: make clear what this is really supposed to do ...

    A :py:class:`Loop` :py:class:`Datasource` can be set in a loop
    state, where it continously reads data objects and places
    them in a in a local register (this sounds more like a fetcher ...)

    Examples:
    - Webcam
    - Video
    - other datasource (-> datafetcher)

    """


class Snapshot(Datasource):
    """Instances of this class are able to provide a snapshot.  Typical
    examples of :py:class:`Snapshot` datasources are sensors and
    cameras that obtain changing data from the environment.  Calling
    the :py:meth:`snapshot` method will provide data reflecting the
    current state of affairs.

    Notes
    -----

    Subclasses of :py:class:`Snapshot` may overwrite the
    py:class:`_get_snapshot` method to provide a snapshot.

    """


class Random(Loop):
    """An abstract base class for datasources that allow to get
    random datapoints and/or batches. Subclasses of this class should
    implement :py:meth:`_fetch_random()`.
    """

    def __init__(self, random_generator: str = 'random', **kwargs) -> None:
        super().__init__(**kwargs)
        self._random_generator = random_generator

    #
    # Public interface
    #

    def get_random(self, **kwargs) -> None:
        """
        This is equivalent to calling `get(random=True)`.
        """
        return self.get_data(random=True, **kwargs)

    #
    # Private methods
    #

    def _get_meta(self, data: Data, random: bool = False,
                  seed=None, **kwargs) -> None:
        # FIXME[todo]: redefined-outer-name
        # pylint: disable=arguments-differ,redefined-outer-name
        if random is not False and not data.datasource_argument:
            data.datasource_argument = 'random'
            data.datasource_value = seed or random
            if seed is not None:
                # we have a seed for the random number generator
                if self._random_generator == 'numpy':
                    np.random.seed(seed)
                else:
                    random.seed(seed)
        super()._get_meta(data, **kwargs)

    def _get_data(self, data: Data, random: bool = False, **kwargs) -> None:
        # pylint: disable=arguments-differ,redefined-outer-name
        """A version of :py:meth:`_get_data` that allows for an
        additional argument `random`.

        Arguments
        ---------
        random: bool
            If set, a random element is taken from this
            :py:class:`Datasource`.
        """
        if random is not False and data.datasource_argument == 'random':
            LOG.debug("Datasource[%s]._get_random(%s, %s)", self, data, kwargs)
            self._get_random(data, **kwargs)
        super()._get_data(data, **kwargs)

    @abstractmethod
    def _get_random(self, data: Data, **kwargs) -> None:
        """This method should be implemented by subclasses that claim
        to be a py:meth:`Random` datasource.
        It should perform whatever is necessary to get a random
        element from the dataset.
        """

    def _get_default(self, data: Data, **kwargs) -> None:
        """The default behaviour of an :py:class:`Random` datasource is to
        get a random element (if not changed by subclasses).
        """
        self._get_random(data, **kwargs)


class Indexed(Random, collections.abc.Sequence):
    """Instances of this class can be indexed.

    Indices can be of different kind (similar to numpy): integers will
    pick individual datapoints, while slices and arrays can be used
    to access a batch of data.

    Index access will usually return a :py:class:`Data` object
    (either single or batch).  By providing an optional second index
    argument, one may instead access individual attributes.

    .. code-block:: python

        datasource = Datasource(module='mnist')
        data = dataource[17]
        batch = datasource[23:87]
        np_image = datasource[23, 'array']
        np_images, np_labels = datasource[10:20, ('array', 'label')]

    :py:class:`Indexed` is an abstract base class. Subclasses should
    implement :py:meth:`__len__` and :py:meth:`_get_index` methods.
    """

    def __getitem__(self, index: Union[int, Tuple[int, Attributes]]) -> Data:
        if isinstance(index, Tuple):
            index, attributes = index
        else:
            attributes = None

        batch = None
        length = len(self)
        if isinstance(index, slice):
            start = index.start or 0
            stop = min(index.stop or length, length)
            step = index.step or 1
            index = np.arange(start, stop, step)
        if isinstance(index, np.ndarray):
            batch = len(index)

        data = self.get_data(index=index, batch=batch)

        if attributes is None:
            return data
        if isinstance(attributes, str):
            return getattr(data, attributes)
        return tuple(getattr(data, attr) for attr in attributes)

    @abstractmethod
    def __len__(self) -> int:
        """Number of data items in this :py:class:`Datasource`.
        """

    def _get_description(self, index=None, **kwargs) -> str:
        # pylint: disable=arguments-differ
        description = super()._get_description(**kwargs)
        if index is not None:
            description += f", index={index}"
        return description

    #
    # data access
    #

    def _get_meta(self, data: Data, index: int = None, **kwargs) -> None:
        """An :py:class:`Indexed` datasource will add an `index`
        attribute to the :py:class:`Data` object. This will be
        a batch attribute, holding the respective index for each
        element of the batch.
        """
        # pylint: disable=arguments-differ
        if index is not None and not data.datasource_argument:
            data.datasource_argument = 'index'
            data.datasource_value = index
        data.add_attribute('index', batch=True)
        super()._get_meta(data, **kwargs)

    def _get_data(self, data: Data, index: int = None, **kwargs):
        # pylint: disable=arguments-differ
        """A version of :py:meth:`_get_data` that allows for an
        additional argument `index`.

        Arguments
        ---------
        index: int
            The index of the element in this
            :py:class:`Indexed` datasource.
        """
        if data.index is not None:
            index = data.index
        if index is not None and data.datasource_argument == 'index':
            LOG.debug("Datasource[%r]._get_index(%s, %s): %s",
                      type(self), index, kwargs, data)
            self._get_index(data, index, **kwargs)
        super()._get_data(data, **kwargs)

    def _get_batch(self, data: Data, index: int = None, **kwargs) -> None:
        # pylint: disable=arguments-differ
        if index is not None:
            if isinstance(index, (int, np.integer)):
                data.index = np.arange(index, index+len(data), dtype=np.int)
            else:
                data.index = index
        super()._get_batch(data, **kwargs)

    #
    # Data
    #

    @abstractmethod
    def _get_index(self, data: Data, index: int, **kwargs) -> None:
        """This method should be implemented by subclasses that claim
        to be a py:meth:`Indexed` datasource.
        It should perform whatever is necessary to get a element with
        the given index from the dataset.
        """
        data.index = index

    def _get_random(self, data: Data, **kwargs) -> None:
        """Get a random element. In a :py:class:`Indexed` datasource
        we can simply choose a random index and get that index.
        """
        index = random.randrange(len(self))
        self._get_index(data, index=index, **kwargs)


class Labeled(Datasource):
    # pylint: disable=abstract-method
    """A :py:class:`Datasource` that provides labels for its data.
    The labels will be stored in a data (batch) attribute called
    `label`.

    Datasources that provide labeled data should derive from this class.
    The should, in their `get_data` and/or `get_batch` method fill
    in the label value.
    """

    def _get_meta(self, data: Data, **kwargs) -> None:
        data.add_attribute('label', batch=True)
        super()._get_meta(data, **kwargs)
