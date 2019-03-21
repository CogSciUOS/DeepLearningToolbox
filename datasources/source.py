from base import BusyObservable, change

from collections import namedtuple
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
        loaded/unloaded, etc.)
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
            self.change('state_changed')

    def _unprepare_data(self):
        pass  # to be implemented by subclasses

    def fetch(self, **kwargs):
        if self.prepared:
            self._fetch(**kwargs)
            self.change('data_changed')
            
    def _fetch(self):
        raise NotImplementedError("_fetch() should be implemented by "
                                  "subclasses of Datasource.")

    @property
    def data(self) -> np.ndarray:
        """Get the current data point.
        This presupposes, that a current data point has been selected
        before by calling :py:meth:`fetch()`
        """
        if self.fetched:
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

    :py:meth:`_prepare_labels` to prepare the use of labels, which should
    assign the :py:attr:`_number_of_labels` attribute to provide the number
    of labels. If there exists some text representation for the label,
    these should be assigned to the list :py:attr:`_text_for_labels`.
    
    :py:meth:`_get_label` and :py:meth:`_get_batch_labels` have to
    be implemented to provide labels for the current data point/batch.

    Attributes
    ----------
    _text_for_labels: list[str] = None
        Optional texts for the labels. None if no such texts are provided.
    """

    _number_of_labels: int = None
    _text_for_labels: list = None  # list[str]

    def __init__(self, label_texts: list=None, **kwargs):
        super().__init__(**kwargs)
        self.set_label_texts(label_texts)

    @property
    def labels_prepared(self) -> bool:
        return self._number_of_labels is not None

    @change
    def prepare(self):
        """Prepare this Datasource for use.
        """
        super().prepare()
        self.prepare_labels()

    def prepare_labels(self):
        """Prepare the labels for used. Has to be called before
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
        """Prepare the labels for used. Has to be called before
        the labels can be used.
        """
        if self.labels_prepared:
            self._number_of_labels = None
            self._text_for_labels = None
            self.change('labels_changed')

    @property
    def number_of_labels(self) -> int:
        """The number of different labels for this dataset
        (e.g. 10 for MNIST, 1000 for ImageNet, etc.). Has to be
        ovwritten by subclasses to provide the actual number.
        """
        if not self.labels_prepared:
            raise RuntimeException("Labels have not been prepared yet.")
        return self._number_of_labels

    @property
    def label(self) -> int:
        """Get the (numeric) for the current data point.
        This presupposes, that a current data point has been selected
        before by calling :py:meth:`fetch()`
        """
        if not self.labels_prepared:
            raise RuntimeException("Labels have not been prepared yet.")
        return self._get_label()

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

    def has_text_for_labels(self):
        """Check if this :py:class:`Labeled` datasource provides text
        versions of its labels.

        This method can be overwritten by subclasses.
        """
        return self._text_for_labels is not None

    def text_for_label(self, label: int) -> str:
        """Get a text representation for a given label.

        Arguments
        ---------
        label: int
            The label to get a text version for.

        Result
        text: str
            The text version of the label.
        """
        if not self.labels_prepared:
            raise RuntimeException("No text available for labels.")
        return self._get_text_for_label(label)

    def text_for_labels(self, labels, **kwargs) -> list:
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
        if self._number_of_labels is None:
            raise RuntimeException("No text available for labels.")
        result = []
        for label in labels:
            result.append(self.text_for_label(label, **kwargs))

    def _get_text_for_label(self, label: int, short: bool=False) -> str:
        text = self._text_for_labels[label]
        return text

    def get_label_texts(self) -> list:
        return [text for text in self._text_for_labels]

    def set_label_texts(self, text_for_labels: list) -> None:
        """Provide some texts for the labels of this Labeled Datasource.

        Arguments
        ---------
        text_for_labels: list
            A list of strings providing the texts for the labels.
            The actual (numerical) label is used as (0-based) index
            in this list. The length of the list should coincide
            with the number of labels.

        Raises
        ------
        ValueException:
            If the provided texts to not fit the labels.
        """
        if (text_for_labels is not None and self.labels_prepared and
            self.number_of_labels != len(text_for_labels)):
            # mismatch
            raise ValueException("Provided wrong number of texts for labels: "
                                 f"expected {self.number_of_labels}, "
                                 f"got {len(text_for_labels)}")
        self._text_for_labels = text_for_labels

    def one_hot_for_label(self, label: int, dtype=np.float32) -> np.ndarray:
        """Get the one-hot representation for a given label.
        """
        output = np.zeros(self.number_of_labels, dtype=dtype)
        output[label] = 1
        return output
        
    def one_hot_for_labels(self, labels) -> np.ndarray:
        """Get the one-hot representation for a set of labels.
        """
        output = np.zeros((len(labels), self.number_of_labels), dtype=dtype)
        for label in labels:
            output[label] = 1
        return output        

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
            elif not self.has_text_for_labels():
                description += (f" (with {self.number_of_labels()} "
                                "labels but withoug text)")
            else:
                description += (f" (with {self.number_of_labels()}"
                                "labels and texts)")
        return description

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
