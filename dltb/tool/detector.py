"""Abstract base class for detectors.
"""
# standard imports
from typing import Union, Tuple
import logging

# third party imports
import numpy as np
import imutils

# toolbox imports
from datasource import Data, Metadata
from .tool import Tool

# logging
LOG = logging.getLogger(__name__)


# A type for possible detections
Detections = Union[Metadata]


class Detector(Tool):
    # pylint: disable=too-many-ancestors
    """A general detector. A detector is intended to detect something
    in some given data.

    The basic detector interface (:py:meth:`detect`) simply maps given
    data to detections.  What detections are and how they are represented
    will differ for specific subclasses (for example an ImageDetector
    typically returns a list of bounding boxes).
    """

    #
    # Detector
    #

    # FIXME[todo]: working on batches (data.is_batch). Here arises the
    #   question what the result type should be for the functional API
    #   (A): a list/tuple or some iterator, or even another structure
    #   (a batch version of Metadata)
    def detect(self, data: Data, **kwargs) -> Detections:
        """Preprocess the given data and apply the detector.

        This method is intended for synchronous use - it dose neither
        alter the `data` object, nor the detector itself. Depending
        on the detector, it may be possible to run the method multiple
        times in parallel.

        Arguments
        ---------
        data: Data
            The data to be fed to the detector. This may be
            a :py:class:`Data` object or simple data array.

        Result
        ------
        detection: Detections
            The dections.
        """
        if not self.prepared:  # FIXME[todo]: decorator @assert_prepared...
            raise RuntimeError("Running unprepared detector.")

        # FIXME[todo/hack]: the following will data batches
        # currently we simply flatten the batch, taking the first item.
        # The correct approach would be to really do detection on
        # the whole batch
        if data.is_batch:
            raise ValueError("Detector currently does not support "
                             "batch detection.")

        LOG.info("Running detector '%s' on data %r", self.key, data)

        if not data:
            return None

        # obtain the preprocessed input data
        preprocessed_data = self.preprocess(data)

        # do the actual processing
        detections = self._detect(preprocessed_data, **kwargs)

        LOG.info("Detector '%s' with %s detections",
                 self.key, detections)

        return detections

    def _detect(self, data: np.ndarray, **kwargs) -> Detections:
        """Do the actual detection.

        The detector will return a Metadata structure containing the
        detections as a list of :py:class:`Location`s (usually of type
        :py:class:`BoundingBox`) in the 'regions' property.
        """
        raise NotImplementedError("Detector class '" +
                                  type(self).__name__ +
                                  "' is not implemented (yet).")

    def _detect_batch(self, data: np.ndarray, **kwargs) -> Detections:
        # FIXME[todo]: batch processing
        raise NotImplementedError("Detector class '" +
                                  type(self).__name__ +
                                  "' is not implemented (yet).")

    #
    # Processor
    #

    def _preprocess_data(self, data: Data, **kwargs) -> None:
        """This method does the actual preprocessing.

        This method may be overwritten by subclasses to add attributes
        to the data object, without assigning values yet (this can be
        done in :py:meth:`_process_data` or
        :py:meth:`_process_batch`). This method may set the data
        object and notify observers, allowing them to observe how the
        data object gets filled during processing.
        """
        super()._preprocess_data(data)
        self.add_data_attribute(data, 'detections')

    def _process_data(self, data) -> None:
        """Process the given data. This will run the detector on
        the data and add the detection results as new attribute
        'detections'.
        """
        LOG.debug("Processing data %r with detector %s", data, self)
        detections = self.detect(data)
        self.set_data_attribute(data, 'detections', detections)
        LOG.debug("Detections found 2: %s, %s", self.detections(data), data)

    def detections(self, data) -> Metadata:
        """Provide the detections from a data object that was processed
        by this :py:class:`Detector`.
        """
        return self.get_data_attribute(data, 'detections')


class ImageDetector(Detector):
    # pylint: disable=too-many-ancestors
    """A detector to be applied to image data.
    """

    def __init__(self, size: Tuple[int, int] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._size = size

    def _preprocess(self, array: np.ndarray, **kwargs) -> np.ndarray:
        """Preprocess the image. This will resize the image to the
        target size of this tool, if such a size is set.
        """
        if array.ndim != 2 and array.ndim != 3:
            raise ValueError("The image provided has an illegal format: "
                             f"shape={array.shape}, dtype={array.dtype}")

        if self._size is not None:
            # resize_ratio = array.shape[1]/400.0
            array = imutils.resize(array, width=400)  # FIXME[hack]

        return super()._preprocess(array, **kwargs)

    def detect(self, data: Data, **kwargs) -> Detections:
        """Preprocess the given data and apply the detector.
        """
        detections = super().detect(data, **kwargs)
        self.rescale(detections, data.data.shape)
        return detections

    def rescale(self, detections: Detections, size: Tuple[int, int]) -> None:
        """Rescale detection made by this detector to some other size.

        """
        if detections is None or self._size is None or self._size == size:
            return  # nothing to do

        resize_ratio = max(self._size[0]/size[0], self._size[1]/size[1])
        detections.scale(resize_ratio)
