"""Abstract base class for tools.
"""

# standard imports
from typing import Any, Union
import time
import logging

# third party imports
import numpy as np

# toolbox imports
from dltb.base.resource import Resource
from base import MetaRegister
from datasource.data import Data, BatchWrapper, BatchDataItem

# logging
LOG = logging.getLogger(__name__)


Datalike = Union[Data, np.ndarray]


class Tool(Resource, metaclass=MetaRegister, method='tool_changed'):
    # pylint: disable=too-many-ancestors
    """:py:class:`Tool` is an abstract base class for tools.
    A Tool can be applied to process data. This abstract base class
    does not define any specific method for processing data,
    this should be done by subclasses (e.g., a Detector provides a
    detect method).

    In addition, :py:class:`Tool` provides some support for processing
    data with a Processor.

    This is an abstract base class and subclasses can (or have to)
    implement at least some of the following methods:

    :py:meth:`_preprocess_data`:
        Preprocess the given data. This method will be invoked
        before processing to bring the data into an appropriate
        format. This method should be overwritten if the Tool expects
        data to be in a specific format.
    :py:meth:`_process_data` and :py:meth:`_process_batch`:
        Do the actal processing and store results as tool-specific
        attributes in the given :py:class:`Data` object. Either
        `_process_data` or `_process_batch` have to be overwritten
        (but it is also allowed to owverwrite both).

    Attributes
    ----------
    timer: bool
        A flag indicating if timing information should be added
        to the :py:class:`Data` object during processing.
    """

    def __init__(self, timer: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timer = timer
        LOG.info("New tool: %s (%s)", self.key, type(self).__name__)

    def __str__(self) -> str:
        return f"{type(self).__name__}[{self.key}]"

    def _prepare(self, **kwargs):
        # pylint: disable=arguments-differ
        """Prepare this tool.
        """
        super()._prepare(**kwargs)

    def preprocess(self, data: Datalike, **kwargs) -> np.ndarray:
        """Perform preprocessing necessary to obtain data suitable for
        for processing with this tool.
        """
        if isinstance(data, np.ndarray):
            return self._preprocess(data, **kwargs)

        if not isinstance(data, Data):
            ValueError(f"Argment data is of invalid type {type(data)}.")

        array = self.preprocessed(data)
        if array is not None:
            return array

        return self._preprocess(data.array, **kwargs)

    def _preprocess(self, array: np.ndarray, **_kwargs) -> np.ndarray:
        # pylint: disable=no-self-use
        """Preprocess the given data. The default implementation does nothing,
        it just returns the input data. The method may be implemented
        by subclasses to do the actual preprocessing.

        Arguments
        ---------
        data: np.ndarray
            The data to be preprocessed.

        Result
        ------
        array: np.ndarray
            The preprocessed data.
        """
        return array

    def add_data_attribute(self, data: Data, name: str, value: Any = None,
                           batch: bool = True) -> None:
        """Add a tool specific attribute to a data object.
        """
        data.add_attribute(self.key + '_' + name, value, batch=batch)

    def set_data_attribute(self, data: Data, name: str, value: Any,
                           index: int = None) -> None:
        """Set a tool specific attribute in a data object.
        """
        setattr(data if index is None else data[index],
                self.key + '_' + name, value)

    def get_data_attribute(self, data: Data, name: str, default: Any = None,
                           index: int = None) -> Any:
        """Get a tool specific attribute from a data object.
        """
        value = getattr(data if index is None else data[index],
                        self.key + '_' + name, None)
        return default if value is None else value

    #
    # data Preprocessing
    #

    def preprocess_data(self, data: Data, **kwargs) -> None:
        """Preprocess the data given data. The preprocessed version of
        the data is stored in the tool-specific 'preprocessed' attribute
        of the :py:class:`Data` object.

        Arguments
        ---------
        data: Data
            A :py:class:`Data` object that can be a batch or a single
            data item.

        """
        self._preprocess_data(data, **kwargs)

    def _preprocess_data(self, data: Data, **kwargs) -> None:
        """This method does the actual preprocessing.

        This method may be overwritten by subclasses to add attributes
        to the data object, without assigning values yet (this can be
        done in :py:meth:`_process_data` or
        :py:meth:`_process_batch`). This method may set the data
        object and notify observers, allowing them to observe how the
        data object gets filled during processing.
        """
        self.add_data_attribute(data, 'preprocessed')
        self.set_data_attribute(data, 'preprocessed',
                                self.preprocess(data, **kwargs))
        if self.timer:
            self.add_data_attribute(data, 'duration')

    def preprocessed(self, data: Data) -> np.ndarray:
        """Get the tool-specific preprocessed data from a
        :py:class:`Data` object.  This may be `None` if the data object
        has not yet undergone preprocessing.
        """
        return self.get_data_attribute(data, 'preprocessed', data.array)

    #
    # Processing
    #

    def process(self, data: Data, **kwargs) -> None:
        """Process the data and adapt the :py:class:`Data` object to
        hold the results.
        This method is to be implemented by subclasses.
        Subclasses may also augment the given data object with
        intermediate results if requested.

        Arguments
        ---------
        data: Data
            A :py:class:`Data` object that is guaranteed not to be a batch.
        """
        self.preprocess_data(data, **kwargs)
        if self.timer:
            start = time.time()
        if data.is_batch:
            self._process_batch(data, **kwargs)
        else:
            self._process_data(data, **kwargs)
        if self.timer:
            end = time.time()
            self.set_data_attribute(data, 'duration', end - start)
        self._postprocess_data(data, **kwargs)

    def _process_data(self, data: Data, **kwargs) -> None:
        """Adapt the data.
        To be implemented by subclasses. Subclasses may augment
        the given data object with the result of their processing.

        Arguments
        ---------
        data: Data
            A :py:class:`Data` object that is guaranteed not to be a batch
            and to provide preprocessed data in its tool-specific
            'preprocessed' attribute.
        """
        if isinstance(data, BatchDataItem):
            raise ValueError("Use _process_batch to process batch data.")
        self._process_batch(BatchWrapper(data))

    def _process_batch(self, batch: Data, **kwargs) -> None:
        """Process batch data. The base implementation just
        processes each batch element individually. Subclasses may
        overwrite this to do real batch processing.

        Arguments
        ---------
        batch: Data
            A :py:class:`Data` object that is guaranteed to be a batch
            and to provide preprocessed data in its tool-specific
            'preprocessed' attribute.
        """
        if isinstance(batch, BatchWrapper):
            raise ValueError("Use _process_data to process non-batch data.")
        for data in batch:
            self._process_data(data)

    def duration(self, data) -> float:
        """Provide the duration (in seconds) the tool needed for processing
        the given data. This property is only available after processing,
        and only if the timer was activated.
        """
        return self.get_data_attribute(data, 'duration')

    def _postprocess_data(self, data: Data, **kwargs) -> None:
        """Postprocess the data. This may result in adding tool specific
        attributes to the data object.  It is assumed that the data
        object was successfully processed by :py:meth:`_process_data`
        before this method is called.
        
        Arguments
        ---------
        data: Data
            The data object to be postprocessd.
        """
        # to be extended by subclasses


class ImageTool(Tool):
    """Abstract base class for tools that operate on images.
    Several convenience methods are provided by this class.
    """

    def __init__(self, size: Tuple[int, int] = None,
                 max_size: Tuple[int, int] = None,
                 min_size: Tuple[int, int] = None,
                 resize_policy: str = 'error', **kwargs) -> None:
        """
        Arguments
        ---------
        size:
            The exact size an image is expected to have for processing
            with this :py:class:`ImageTool`.
        min_size:
            The minimal image size processable by this :py:class:`ImageTool`.
        min_size:
            The maximal image size processable by this :py:class:`ImageTool`.
        resize_policy:
            The resize policy to adapt, if the aspect ratio of the image
            does not fit to this tool. Options are: 'pad' = pad the
            borders on the shorter side (with 0s), 'crop' = cut away
            some parts from the longer side, 'distort' = use different
            scaling along horizontal and vertical axis, resulting
            in a change of aspect ratio, 'error' = raise a ValueError.
        """
        super().__init__(**kwargs)
        if size is not None:
            if max_size is not None:
                LOG.warning("Argument 'max_size' (=%s) is ignored "
                            "when 'size' (=%s) is given.", max_size, size)
            if min_size is not None:
                LOG.warning("Argument 'min_size' (=%s) is ignored "
                            "when 'size' (=%s) is given.", min_size, size)
            self._max_size = size
            self._min_size = size
        else:
            if (min_size is not None and max_size is not None and
                    (min_size[0] > max_size[0] or min_size[1] > max_size[1])):
                raise ValueError(f"Minimal size (={min_size}) is larger than"
                                 f"maximal size (={max_size}).")
            self._max_size = max_size
            self._min_size = min_size

        self._resize_policy = resize_policy

    def fit_image(self, image: Imagelike, policy=) -> np.ndarray:
        """Resize the image to be suitable for processing with
        this :py:class:`ImageTool`.
        """
        size = image.shape[:1:-1]

        if self._min_size is not None:
            min_scale_x = max(self._min_size[0] / size[0], 1.)
            min_scale_y = max(self._min_size[1] / size[1], 1.)
            min_scale = max(min_scale_x, min_scale_y)
        else:
            max_scale = 0.0

        if self._max_size is not None:
            max_scale_x = min(self._max_size[0] / size[0], 1.)
            max_scale_y = min(self._max_size[1] / size[1], 1.)
            max_scale = min(max_scale_x, max_scale_y)
        else:
            max_scale = float('inf')

        if min_scale <= 1. <= max_scale:
            # The image fits into the desired size
            return image
        
        if min_scale <= max_scale:
            # Choose the scale closest to 1.
            scale = max_scale if max_scale < 1. else min_scale
            return imscale(image, scale=scale)

        # Here max_scale < min_scale, that is there is no way to fit
        # the image to a valid size and maintain the aspect ratio.
        # Different strategies can be applied here.
        if self._resize_policy == 'error':
            ValueError(f"Cannot fit image of size {size} into the "
                       f"acceptable size for procssing with {self} "
                       f"(min_size={self._min_size}, "
                       f"max_size={self._max_size}).")

        if self._resize_policy == 'pad':
            scale_x = min_scale
            scale_y = min_scale
        elif self._resize_policy == 'crop':
            scale_x = max_scale
            scale_y = max_scale
        else:  # if self._resize_policy == 'distort':
            # FIXME[todo]: find optimal scale (minimal distortion)
            # (this should be a compromise between minimal and
            # maximal scaling)
            # FIXME[hack]
            scale_x = min_scale_x if min_scale_x < 1. else max_scale_x
            scale_y = min_scale_y if min_scale_y < 1. else max_scale_y

        scaled_image = imscale(image, scale_x=scale_x, scale_y=scale_y)

        if self._resize_policy == 'pad':
            # FIXME[todo]: do padding
            padded_image = scaled_image
            return padded_image
        
        if self._resize_policy == 'pad':
            # FIXME[todo]: do cropping
            cropped_image = scaled_image
            return padded_image

        return scaled_image
