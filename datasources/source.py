from .meta import Metadata
from base import BusyObservable, change
from util.image import imread

import os
import random
from collections import namedtuple
from typing import Dict, List, Sequence, Union
import numpy as np


InputData = namedtuple('Data', ['data', 'name'])



class Datasource(BusyObservable, method='datasource_changed',
                 changes=['state_changed', 'metadata_changed',
                          'data_changed', 'batch_changed']):
    """.. :py:class:: Datasource

    An abstract base class for different types of data sources.

    There are different APIs for navigating in a datasource, depending
    on what that datasource supports:

    array-like navigation: individual elements of a data source can be
    accessed using an array-like notation.

    random selection: select a random element from the dataset.

    Attributes
    ----------
    _description : str
        Short description of the dataset
        
    Changes
    -------
    state_changed:
        The state of the DataSource has changed (e.g. data were downloaded,
        loaded/unloaded, unprepared/prepared, etc.)
    data_changed:
        The data have changed (e.g. by invoking some fetch ... method).
    batch_changed:
        The batch has changed (e.g. by invoking the fetch_batch method).
    metadata_changed:
    
    """

    _metadata = None

    def __init__(self, description: str=None, **kwargs):
        """Create a new Datasource.

        Parameters
        ----------
        description :   str
                        Description of the dataset
        """
        super().__init__(**kwargs)
        self._description = (self.__class__.__name__ if description is None
                             else description)

    @property
    def prepared(self) -> bool:
        """Report if this Datasource prepared for use.
        A Datasource has to be prepared before it can be used.
        """
        return True  # to be implemented by subclasses

    @change
    def prepare(self):
        """Prepare this Datasource for use.
        """
        if not self.prepared:
            self._prepare_data()
            self.fetch()
            self.change('state_changed')

    def _prepare_data(self):
        raise NotImplementedError("_prepare_data() should be implemented by "
                                  "subclasses of Datasource.")

    def unprepare(self):
        """Unprepare this Datasource for use.
        """
        if self.prepared:
            self._unprepare_data()
            self.change('state_changed', 'data_changed')

    def _unprepare_data(self):
        """Undo the :py:meth:`_prepare_data` operation. This will free
        resources used to store the data.
        """
        pass  # to be implemented by subclasses

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
        if self.prepared:
            self._metadata = Metadata(description=f"Metadata for {self}")
            self._fetch(**kwargs)
            self.change('data_changed')
            
    def _fetch(self, **kwargs):
        raise NotImplementedError("_fetch() should be implemented by "
                                  "subclasses of Datasource.")

    @property
    def fetched(self) -> bool:
        """Check if data have been fetched and are now available.
        If so, :py:meth:`data` should deliver this data.
        """
        raise NotImplementedError("fetched() should be implemented by "
                                  "subclasses of Datasource.")

    @property
    def data(self) -> np.ndarray:
        """Get the current data point.
        This presupposes, that a current data point has been selected
        before by calling :py:meth:`fetch`
        """
        if not self.fetched:
            raise RuntimeException("No data has been fetched on Datasource "
                                   f"{self.__class__.__name}")
        return self._get_data()

    def _get_data(self) -> np.ndarray:
        """The actual implementation of the :py:meth:`data` property
        to be overwritten by subclasses.

        It can be assumed that a data point has been fetched when this
        method is invoked.
        """
        raise NotImplementedError(f"{self.__class__.__name__} claims to be "
                                  "a Datasource, but it does not "
                                  "implement the '_get_data' method")

    @property
    def batch_data(self) -> np.ndarray:
        """Get the currently selected batch.
        """
        return self._get_batch_data()

    def _get_batch_data(self) -> np.ndarray:
        """The actual implementation of the :py:meth:`batch_data`
        to be overwritten by subclasses.

        It can be assumed that a batch has been fetched when this
        method is invoked.
        """
        raise NotImplementedError(f"{self.__class__.__name__} claims to be "
                                  "a Datasource, but it does not "
                                  "implement the '_get_batch_data' method")
    
    @property
    def metadata(self):
        """Metadata for the currently selected data.
        """
        if not self.fetched:
            raise RuntimeException("No (meta)data has been fetched "
                                   f"on Datasource {self.__class__.__name}")
        return self._metadata

    @property
    def description(self) -> str:
        return self.get_description()

    def get_description(self) -> str:
        """Provide a description of the Datasource or one of its
        elements.

        Attributes
        ----------
        index: int
            In case of an indexed Datasource, provide a description
            of the given element.
        target: bool
            If target values are available (labeled dataset),
            also report that fact, or in case of an indiviual element
            include its label in the description.
        """
        return self._description

    def load_datapoint_from_file(self, filename: str):
        """Load a single datapoint from a file.
        This method should be implemented by subclasses.

        Arguments
        ---------
        filename: str
            The absolute filename from which the data should be loaded.

        Result
        ------
        datapoint
            The datapoint loaded from file.
        """
        raise NotImplementedError(f"The datasource {self.__class__.__name__} "
                                  "does not implement a method to load a "
                                  f"datapoint from file '{filename}'")

class Imagesource(Datasource):

    def load_datapoint_from_file(self, filename) -> np.ndarray:
        """Load a single datapoint, that is an image, from a file.

        Arguments
        ---------
        filename: str
            The absolute image filename. The file may be in any format
            supported by :py:meth:`imread`.

        Result
        ------
        image
            The image loaded from file.
        """
        return imread(filename)

    @property
    def image(self):
        return self.data

class Labeled(Datasource):
    """A :py:class:`Datasource` that provides labels for its data.


    Datasources that provide labeled data should derive from this class.
    These subclasses should implement the following methods:

    :py:meth:`_prepare_labels` to prepare the use of labels, after which
    the :py:attr:`number_of_labels` property should provide the number
    of labels. If there exists additional label formats, like textual labels
    or alternate indexes, this can be added by calling
    :py:meth:`add_label_format`.
    
    :py:meth:`_get_label` and :py:meth:`_get_batch_labels` have to
    be implemented to provide labels for the current data point/batch.

    Attributes
    ----------
    _label_formats: Mapping[str, Union[np.ndarray,list]]
        A mapping of supported container formats. Each format identifier
        is mapped to some lookup table.
    _label_reverse: Mapping[str, Union[np.ndarray,list]]
        A mapping of supported container formats. A format identifier
        is mapped to some reverse lookup table.  Not all reverse
        lookup tables may initially exist, but they are created
        on the fly in :py:meth:`format_labels` the first time they
        are needed.
    """
    _label_formats: dict = None
    _label_reverse: dict = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._label_formats = {}
        self._label_reverse = {}

    @property
    def number_of_labels(self) -> int:
        """The number of different labels for this dataset
        (e.g. 10 for MNIST, 1000 for ImageNet, etc.). Has to be
        ovwritten by subclasses to provide the actual number.
        """
        return None

    @property
    def labels_prepared(self) -> bool:
        """Check if labels for this dataset have been prepared.
        Labels have to be prepared before
        :py:meth:`get_label`, or :py:meth:`batch_labels` can be called.
        """
        return self.number_of_labels is not None

    @change
    def prepare(self):
        """Prepare this Datasource for use.
        """
        change = False
        if not self.prepared:
            self._prepare_data()
            change = True
        if not self.labels_prepared:
            self.prepare_labels()
            change = True
        if change:
            self.fetch()
            self.change('state_changed')

    def prepare_labels(self):
        """Prepare the labels for use. Has to be called before
        the labels can be used.
        """
        if not self.labels_prepared:
            self._prepare_labels()
            self.change('labels_changed')

    def _prepare_labels(self):
        """Actual implementation of :py:meth:`prepare_labels` to be
        overwritten by subclasses.
        """
        pass  # to be implemented by subclasses

    @change
    def unprepare(self):
        """Unprepare: release resources Prepare this Datasource for use.
        """
        super().unprepare()
        self.unprepare_labels()

    def unprepare_labels(self):
        """Free resources allocated by labels. Should be called
        when labels are no longer needed.
        """
        if self.labels_prepared:
            self._formats = {}
            self.change('labels_changed')

    @property
    def label(self) -> int:
        """Get the (numeric) for the current data point.
        This presupposes, that a current data point has been selected
        before by calling :py:meth:`fetch()`
        """
        return self.get_label()

    def get_label(self, format: str=None):
        """Get the (numeric) label for the current data point.
        This presupposes, that a current data point has been selected
        before by calling :py:meth:`fetch()`
        """
        if not self.labels_prepared:
            raise RuntimeException("Labels have not been prepared yet.")
        if not self.fetched:
            raise RuntimeException("No data has been fetched.")
        label = self._get_label()
        return label if format is None else self.format_label(label, format)

    def _get_label(self):
        """The actual implementation of the :py:meth:`label` property
        to be overwritten by subclasses.

        It can be assumed that a data point has been fetched when this
        method is invoked.
        """
        raise NotImplementedError(f"{self.__class__.__name__} claims to be "
                                  "a 'Labeled' datasource, but it does not "
                                  "implement the '_get_label' method")

    @property
    def batch_labels(self) -> np.ndarray:
        """Get the (numeric) labels for the currently selected batch.
        """
        return self._get_batch_labels()

    def _get_batch_labels(self) -> np.ndarray:
        """The actual implementation of the :py:meth:`batch_labels`
        to be overwritten by subclasses.

        It can be assumed that a batch has been fetched when this
        method is invoked.
        """
        raise NotImplementedError(f"{self.__class__.__name__} claims to be "
                                  "a 'Labeled' datasource, but it does not "
                                  "implement the '_get_batch_labels' method")

    @property
    def label_formats(self):
        """Get a list of label formats supported by this
        :py:class:`Datasource`. Formats from this list can be passed
        as format argument to the :py:meth:`get_label` method.

        Result
        ------
        formats: List[str]
            A list of str naming the supported label formats.
        """
        return list(self._label_formats.keys())
    
    def format_labels(self, labels, format: str=None, origin: str=None):
        """Translate labels in a given format.

        Arguments
        ---------
        labels:
            The labels to transfer in another format. Either a single
            (numerical) label or some iterable.
        format: str
            The label format, we want the labels in. If None, the
            internal format will be used.
        origin: str
            The format in which the labels are provided. In None, the
            internal format will be assumed.

        Result
        ------
        formated_labels:
            The formated labels. Either a single label or some iterable,
            depending how labels were provided.
        """
        if labels is None:
            return None

        # Step 1: convert labels into internal format
        if origin is not None:
            # labels are in some custom (not the internal) format
            # -> we will transform them into the internal format first

            # Step 1a: check if we know the label format
            if not origin in self._label_formats:
                raise ValueError(f"Format {origin} is not supported by "
                                 f"{self.__class__.__name__}. Known formats "
                                 f"are: {self._label_formats.keys()}")

            # Step 1b: check if we have a reverse lookup table
            if not origin in self._label_reverse:
                # if no such table exists yet, create a new one
                table = self._label_formats[origin]
                if isinstance(table, np.ndarray):
                    # FIXME[hack]: assume 0-based labels with no "gaps"
                    reverse_table = np.argsort(table)
                else:
                    reverse_table = { v: i for i, v in enumerate(table) }
                self._label_reverse[origin] = reverse_table
            else:
                reverse_table = self._label_reverse[origin]

            # Step 1c: do the conversion
            if isinstance(labels, int) or isinstance(labels, str):
                # an individual label
                labels = reverse_table[labels]
            elif isinstance(reverse_table, np.ndarray):
                # multiple labels with numpy reverse loopup table
                labels = reverse_table[labels]
            else:
                # multiple labels with another form of reverse loopup table
                labels = [reverse_table[label] for label in labels]

        # Step 2: convert labels into target format
        if format is None:
            return labels  # nothing to do ...

        if format in self._label_formats:
            table = self._label_formats[format]
            if isinstance(labels, int) or isinstance(labels, str):
                # an individual label
                return table[labels]
            elif isinstance(table, np.ndarray):
                # multiple labels with a numpy lookup table
                return table[labels]
            else:
                # multiple labels with another form of lookup table
                return [table[label] for label in labels]
        else:
            raise ValueError(f"Format {format} is not supported by "
                             f"{self.__class__.__name__}. Known formats "
                             f"are: {self._label_formats}")

    def get_formated_labels(self, format: str=None):
        """Get the formated labels of this :py:class:`Labeled`
        :py:class:`Datasource`.

        Result
        ------
        formated_labels:
            An iterable providing the formated labels.
        """
        return self.format_labels(range(self.number_of_labels), format)

    def add_label_format(self, format: str, formated_labels) -> None:
        """Provide some texts for the labels of this Labeled Datasource.

        Arguments
        ---------
        format: str
            The name of the format to add.
        formated_labels: list
            A list of formated labels.

        Raises
        ------
        ValueError:
            If the provided texts to not fit the labels.
        """
        if self.number_of_labels != len(formated_labels):
            raise ValueError("Provided wrong number of formated labels: "
                             f"expected {self.number_of_labels}, "
                             f"got {len(formated_labels)}")
        self._label_formats[format] = formated_labels

    def one_hot_for_labels(self, labels, format: str=None,
                           dtype=np.float32) -> np.ndarray:
        """Get the one-hot representation for a given label.

        Arguments
        ---------
        labels: Union[int, Sequence[int]]

        Result
        ------
        one_hot: np.ndarray
            A one hot vector (1D) in case a single label was provided,
            or 2D matrix containing one hot vectors as rows.
        """
        if format is not None:
            labels = self.format_labels(labels, format)
        if isinstance(labels, int):
            output = np.zeros(self.number_of_labels, dtype=dtype)
            output[label] = 1
        else:
            output = np.zeros((len(labels), self.number_of_labels),
                              dtype=dtype)
            for label in labels:
                output[label] = 1
        return output

    # FIXME[old]: 

    def has_text_for_labels(self):
        """Check if this :py:class:`Labeled` datasource provides text
        versions of its labels.

        This method can be overwritten by subclasses.
        """
        return 'text' in self._label_formats

    def text_for_label(self, label: int, format: str='text') -> str:
        """Get a text representation for a given label.

        Arguments
        ---------
        label: int
            The label to get a text version for.

        Result
        text: str
            The text version of the label.
        """
        return self.format_labels(label, format=format)

    def text_for_labels(self, labels, format: str='text') -> list:
        """Get text representation for set of labels.

        Arguments
        ---------
        labels: Iterable
            Some iterable providing the (numeric) labels.

        Result
        ------
        texts: list
            A list containing the corresponding text representations
            for the labels.
        """
        return self.format_labels(labels, format=format)

    def get_description(self, with_label: bool=False, **kwargs) -> str:
        """Provide a description of the Datasource or one of its
        elements.

        Attributes
        ----------
        with_label: bool
            Provide information if labels are available.
        """
        description = super().get_description(**kwargs)
        if with_label:
            if not self.labels_prepared:
                description += " (without labels)"
            else:  # FIXME[hack]: description2 is not used ...
                description2 = f" (with {self.number_of_labels})"
                if self.fetched:
                    label = self._get_label()
                    description += f": {label}"
                    if label is not None: # FIXME[bug]: it should never happen that label is None - repair and than remove the if statement ...
                        for format in self._label_formats.keys():
                            formated_label = self.format_labels(label, format)
                            description2 += f", {format}: {formated_label}"
        return description


class Random(Datasource):
    """An abstract base class for datasources that allow to fetch
    random datapoints and/or batches. Subclasses of this class should
    implement :py:meth:`_fetch_random()`.
    """

    def _fetch(self, random: bool=False, **kwargs) -> None:
        """A version of :py:meth:`fetch` that allows for an
        additional argument `random`.

        Arguments
        ---------
        random: bool
            If set, a random element is fetched from this
            :py:class:`Datasource`.
        """
        if random:
            self._fetch_random(**kwargs)
            self.change('data_changed')
        else:
            super().fetch(**kwargs)

    def _fetch_random(self, **kwargs) -> None:
        """This method should be implemented by subclasses that claim
        to be a py:meth:`Random` datasource.
        It should perform whatever is necessary to fetch a random
        element from the dataset.
        """
        raise NotImplementedError(f"{self.__class__.__name__} claims to be "
                                  "a 'Random' datasource, but it does not "
                                  "implement the '_fetch_random' method.")


class Indexed(Random):
    """Instances of this class can be indexed.
    """

    def __getitem__(self, index):
        self.fetch_index(index=index)
        return self.get_data()

    def _fetch(self, index=None, **kwargs):
        """A version of :py:meth:`fetch` that allows for an
        additional argument `random`.

        Arguments
        ---------
        random: bool
            If set, a random element is fetched from this
            :py:class:`Datasource`.
        """
        if index is not None:
            self._fetch_index(index, **kwargs)
        else:
            self._fetch_random(**kwargs)

    def _fetch_index(self, index, **kwargs) -> None:
        """This method should be implemented by subclasses that claim
        to be a py:meth:`Random` datasource.
        It should perform whatever is necessary to fetch a random
        element from the dataset.
        """
        raise NotImplementedError(f"{self.__class__.__name__} claims to be "
                                  "a 'Random' datasource, but it does not "
                                  "implement the '_fetch_random' method.")

    def _fetch_random(self, **kwargs) -> None:
        """Fetch a random image. In a :py:class:`Indexed` datasource
        we can simply choose a random index and fetch that index.
        """
        self._fetch_index(index=random.randrange(len(self)), **kwargs)


    @property
    def index(self) -> int:
        raise NotImplementedError(f"Subclasses of {Indexed.__name__} "
                                  "should implement property 'index', but "
                                  f"{self.__class__.__name__} doesn't do that.")


    def __len__(self) -> int:
        raise NotImplementedError(f"Subclasses of {Indexed.__name__} "
                                  "should implement 'len', but "
                                  f"{self.__class__.__name__} doesn't do that.")

    def get_description(self, index=None, **kwargs) -> str:
        description = super().get_description(**kwargs)
        if index is not None:
            description += f", index={index}"
        return description


from threading import Event


class Loop(Datasource):
    """The :py:class:`Loop` class provides a loop logic.

    Notice: the methods of this class are not intended to be
    called directly, but they are engaged via the
    :py:meth:`loop` method of the :py:class:`datasources.Controller`
    class.
    """

    # An event manages a flag that can be set to true with the set()
    # method and reset to false with the clear() method. The wait()
    # method blocks until the flag is true.
    _loop_event: Event = None
    _loop_running: bool = False
    _loop_interval: float = None
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._loop_running = False
        self._loop_event = None
        self._loop_interval = 0.2

    @property
    def looping(self):
        """Check if this datasource is currently looping.
        """
        return self._loop_running   

    def start_loop(self):
        if not self._loop_running:
            self._loop_running = True
            self._loop_event = Event()
            self.change('state_changed')

    def stop_loop(self):
        if self._loop_running:
            self._loop_running = False
            self.change('state_changed')

    def run_loop(self):
        """
        This method is intended to be invoked in its own Thread.
        """
        # self._logger.info("Running datasource loop")
        while self._loop_running:
            self.fetch(random=True)
            # Now wait before fetching the next input
            self._loop_event.clear()
            self._loop_event.wait(timeout=self._loop_interval)
        self._loopEvent = None


class Snapshot(Datasource):
    """Instances of this class are able to provide a snapshot.
    """

    def __init__(self, **kwargs) -> None:
        """Instantiate a new :py:class:`Snapshot` object.
        """
        super().__init__(**kwargs)

    def snapshot(self) -> None:
        """Create a snapshot.
        """
        self.fetch()
