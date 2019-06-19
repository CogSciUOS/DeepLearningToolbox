from base import BusyObservable, change

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
            raise RuntimeException("No data has been fetched.")
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
    """
    _label_formats: dict = {}
    _label_reverse: dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

        if origin is not None:
            if not origin in self._label_formats:
                raise ValueError(f"Format {origin} is not supported by "
                                 f"{self.__class__.__name__}. Known formats "
                                 f"are: {self._label_formats.keys()}")
            if not origin in self._label_reverse:
                table = self._label_formats[origin]
                if isinstance(table, np.ndarray):
                    # FIXME[hack]: assume 0-based labels with no "gaps"
                    reverse_table = np.argsort(table)
                else:
                    reverse_table = { v: i for i, v in enumerate(table) }
                self._label_reverse[origin] = reverse_table
            else:
                reverse_table = self._label_reverse[origin]

            if isinstance(labels, int) or isinstance(labels, str):
                labels = reverse_table[labels]
            elif  isinstance(reverse_table, np.ndarray):
                labels = reverse_table[labels]
            else:
                labels = [reverse_table[label] for label in labels]
        
        if format is None:
            return labels

        if format in self._label_formats:
            table = self._label_formats[format]
            if isinstance(labels, int) or isinstance(table, np.ndarray):
                return table[labels]
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
                    if label is not None: # FIXME[bug] should never be the case
                        for format in self._label_formats.keys():
                            formated_label = self.format_labels(label, format)
                            description2 += f", {format}: {formated_label}"
        return description


class Random(Datasource):
    """An abstract base class for datasources that allow to fetch
    random datapoints and/or batches. Subclasses of this class should
    implement :py:meth:`_fetch_random()`.
    """

    def fetch(self, random: bool=False, **kwargs) -> None:
        if random:
            self._fetch_random(**kwargs)
            self.change('data_changed')
        else:
            super().fetch(**kwargs)

    def _fetch_random(self, **kwargs) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} claims to be "
                                  "a 'Random' datasource, but it does not "
                                  "implement the '_fetch_random' method.")


class Predefined(Datasource):
    """An abstract base class for predefined data sources.
    """
    #
    # Static data and methods
    #

    datasources = {}

    @staticmethod
    def get_data_source_ids():
        return list(Predefined.datasources.keys())

    @staticmethod
    def get_data_source(id):
        return Predefined.datasources[id]

    _id: str = None

    def __init__(self, id: str=None, **kwargs):
        if id is None:
            raise ValueError("You have to provde an id for "
                             "a Predefined datasoure")
        super().__init__(**kwargs)
        self._id = id
        Predefined.datasources[id] = self

    @property
    def id(self):
        """Get the "public" ID that is used to identify this datasource.  Only
        predefined Datasource should have such an ID, other
        datasources should provide None.
        """
        return self._id

    def get_public_id(self):
        """Get the "public" ID that is used to identify this datasource.  Only
        predefined Datasource should have such an ID, other
        datasources should provide None.
        """
        return self._id

    def check_availability(self):
        """Check if this Datasource is available.

        Returns
        -------
        True if the Datasource can be instantiated, False otherwise.
        """
        return False

    def download(self):
        raise NotImplementedError("Downloading this datasource is "
                                  "not implemented yet.")

from threading import Event


class Loop:

    # An event manages a flag that can be set to true with the set()
    # method and reset to false with the clear() method. The wait()
    # method blocks until the flag is true.
    _loop_event: Event = None
    _loop_running: bool = False
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._loop_running = False
        self._loop_event = None

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
        # self._logger.info("Running datasource loop")
        while self._loop_running:
            self.fetch(random=True)
            self._loop_event.clear()
            self._loop_event.wait(timeout=.2)
        self._loopEvent = None
