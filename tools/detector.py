"""Abstract base class for detectors.
"""
# standard imports
import time
import logging

# third party imports
import numpy as np
import imutils

# toolbox imports
from base import busy
from datasource import Data, Metadata
from .tool import Processor

# logging
LOG = logging.getLogger(__name__)


class Detector(Processor, method='detector_changed',
               changes=['data_changed', 'detection_finished']):
    # pylint: disable=too-many-ancestors
    """A general detector.

    detector_changed:
        The input data given to the detector have changed.

    detection_finished:
        The detection has finished. This is reported when the
        detector has finished its work.

    Attributes
    ----------
    _data:
    _next_data:
    _detections:
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._data = None
        self._next_data = None
        self._detections = None

    def __str__(self) -> str:
        return f"{type(self).__name__}[{self.key}]"

    #
    # Detector
    #

    @busy("detecting")
    def detect(self, data: Data, **kwargs):
        """Apply the detector to the given data.

        Detection will be run asynchronously and set the detector
        in a busy state ('detecting').
        """
        if not self.prepared:
            raise RuntimeError("Running unprepared detector.")

        LOG.info("Running detector '%s' on data %r", self.key, data)

        if not data:
            return None

        # obtain the (preprocessed) input data
        detector_in = self.get_data_attribute(data, 'in', default=data.data)

        try:
            start = time.time()
            detections = self._detect(detector_in, **kwargs)  # Metadata
            end = time.time()
            detections.duration = end - start
        except Exception:
            LOG.error("MTCNN: error during detection!")
            raise

        LOG.info("Detector '%s' finished after %.4fs",
                 self.key, detections.duration)

        return detections

    def _detect(self, image: np.ndarray, **kwargs) -> Metadata:
        """Do the actual detection.

        The detector will return a Metadata structure containing the
        detections as a list of :py:class:`Location`s (usually of type
        :py:class:`BoundingBox`) in the 'regions' property.
        """
        raise NotImplementedError("Detector class '" +
                                  type(self).__name__ +
                                  "' is not implemented (yet).")

    #
    # Processor
    #

    def _process_data(self, data) -> None:
        """Process the given data. This will run the detector on
        the data and add the detection results as new attribute
        'detections'.
        """
        LOG.info("processing loop <%s> passes <%s> to detector",
                 self, data)
        self.add_data_attribute(data, 'detections')
        self.set_data_attribute(data, 'detections', self.detect(data))

    @property
    def detections(self) -> Metadata:
        """A metadata holding the results of the last invocation
        of the detector.
        """
        return self.get_data_attribute(self._data, 'detections')


class ImageDetector(Detector):
    # pylint: disable=too-many-ancestors
    """A detector to be applied to image data.
    """

    def detect(self, data: Data, **kwargs) -> Metadata:
        """Apply the detector to a given image.
        """
        if not data:
            return None

        # FIXME[todo/hack]: the following will data batches
        # currently we simply flatten the batch, taking the first item.
        # The correct approach would be to really do detection on
        # the whole batch
        data = data[0] if data.is_batch else data
        
        image = data.data

        resize_ratio = image.shape[1]/400.0
        image = imutils.resize(image, width=400)

        if image.ndim != 2 and image.ndim != 3:
            raise ValueError("The image provided has an illegal format: "
                             f"shape={image.shape}, dtype={image.dtype}")

        data.add_attribute(self.key + '_in', image)

        detections = super().detect(data, **kwargs)

        if detections is not None and resize_ratio != 1.0:
            detections.scale(resize_ratio)

        return detections

    @property
    def image(self):
        """The image processed by this :py:class:`ImageDetector`.
        """
        return self.data
