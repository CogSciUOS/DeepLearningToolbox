"""Abstract base class for detectors.

FIXME[todo]: the detector interface seems to be somewhat inconsistent.
Clean it!

Example:

from dltb.tool import Tool
Tool['haar'].detect_and_show('examples/reservoir-dogs.jpg')

detect_and_show

from dltb.thirdparty.opencv.face import DetectorHaar
detector = DetectorHaar()

filename = '/space/data/lfw/lfw/Arnold_Schwarzenegger/Arnold_Schwarzenegger_0002.jpg'


The following currently works:

from dltb.base.image import Image
image = Image(filename)
detections = detector.detect(image) # Metadata

# and 

data = detector.process_image(filename)
detections = detector.detections(data)  # Metadata



The following does not work:

d = detector.detect(filename)
# AttributeError: 'str' object has no attribute 'is_batch'

boxes = list(detector.detect_boxes(image))
# boxes=[]
"""
# standard imports
from typing import Union, Tuple, List, Any, Iterable
import logging

# third party imports
import numpy as np

# toolbox imports
from ..base.data import Data
from ..base.meta import Metadata
from ..base.image import Image, Imagelike, Region, BoundingBox
from ..util.image import imshow
from .tool import Tool
from .image import ImageTool

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

    def _process(self, data, **kwargs) -> Any:
        """Processing data with a :py:class:`Detector` means detecting.
        """
        return self._detect(data, **kwargs)

    # FIXME[todo]: working on batches (data.is_batch). Here arises the
    #   question what the result type should be for the functional API
    #   (A): a list/tuple or some iterator, or even another structure
    #   (a batch version of Metadata)
    def detect(self, data: Data, **kwargs) -> Detections:
        """Preprocess the given data and apply the detector.

        This method is intended for synchronous use - it does neither
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
        # FIXME[old]: preprocessed_data = self.preprocess(data)
        preprocessed_data = self._preprocess_data(data)
        if preprocessed_data is None:
            preprocessed_data = data.array
        print("Detector.detect: data =", type(data),
              "; preprocessed_data =", type(preprocessed_data))

        # do the actual processing
        detections = self._detect(preprocessed_data, **kwargs)
        LOG.info("Detector '%s' with %s detections",
                 self.key, detections)

        detections = self._adapt_detections(detections, data)

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

    def _adapt_detections(self, detections: Detections,
                          data: Data) -> Detections:
        raise NotImplementedError("Detector class '" +
                                  type(self).__name__ +
                                  "' is not implemented (yet).")

    #
    # Processor
    #

    def _preprocess_data(self, data: Data, **kwargs) -> None:
        """This will add a detector specific `'detections'` attribute
        to the  :py:class:`Data` object.  This attribute is intended
        to hold the detections.  It is going to be filled by
        :py:class:`process_data`.
        """
        super()._preprocess_data(data, **kwargs)
        self.add_data_attribute(data, 'detections')

    def _process_data(self, data: Data, **kwargs) -> None:
        """Process the given data. This will run the detector on
        the data and add the detection results as new attribute
        `'detections'` to  `data`.
        """
        LOG.debug("Processing data %r with detector %s", data, self)
        # self.detect() includes preprocessing and postprocessing
        detections = self.detect(data)
        self.set_data_attribute(data, 'detections', detections)
        LOG.debug("Detections found 2: %s, %s", self.detections(data), data)

    def detections(self, data) -> Metadata:
        """Provide the detections from a data object that was processed
        by this :py:class:`Detector`.
        """
        return self.get_data_attribute(data, 'detections')


class ImageDetector(Detector, ImageTool):
    # pylint: disable=too-many-ancestors
    """A detector to be applied to image data.

    In case of an :py:class:`ImageDetector` the :py:class:`Metadata`
    essentially is a set of :py:class:`Region`s, that is a
    :py:class:`Location`, optionally annotated with additional
    information like confidence values, or a class lable.
    The :py:class:`Location` is typically a :py:class:`BoundingBox`,
    but it may also be a more sophisticated contour are a bit mask.

    Notice that the resulting :py:class:`Metadata` can contain
    multiple regions (in case that the detector found multiple
    patterns in the input image), and also no region at all (in case
    that no pattern has been detected).

    """

    def __init__(self, size: Tuple[int, int] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._size = size

    #
    # Implementation of the private API
    #

    external_result: Tuple[str] = ('detections', )   # Metadata
    internal_result: Tuple[str] = ('_detections', )  # internal

    def _postprocess(self, context: Data, name: str) -> None:
        # FIXME[todo]: batch processing
        if name == 'detections':
            if hasattr(context, '_detections'):
                detections = context._detections
                if self._size is not None and hasattr(context, 'image'):
                    size = context.image.shape
                    resize_ratio = max(self._size[0]/size[0],
                                       self._size[1]/size[1])
                    detections.scale(resize_ratio)
            else:
                detections = None
            context.add_attribute('detections', detections)

        elif name == 'mark':
            if not hasattr(context, 'detections'):
                self._postprocess(context, 'detections')
            context.add_attribute(name, self.mark_image(context.input_image,
                                                        context.detections))

        elif name == 'extract':
            if not hasattr(context, 'detections'):
                self._postprocess(context, 'detections')
            context.add_attribute(name,
                                  self.extract_from_image(context.image,
                                                          context.detections))

        else:
            super()._postprocess(context, name)

    @staticmethod
    def intersection_over_union(box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate the intersection over union of two boxes (rectangles).
        The intersection of union is computed as the ratio of the overlap
        of the two boxes and their union.
        """
        intersection_area = (box1 * box2).area()
        union_area = box1.area() + box2.area() - intersection_area
        return intersection_area/union_area

    # @staticmethod
    def evaluate_single_image(self, gt_boxes, pred_boxes, iou_thr):
        """Calculates number of true_pos, false_pos, false_neg
        from single batch of boxes.

        Arguments
        ---------
        gt_boxes (list of list of floats):
            list of locations of ground truth objects as
            [xmin, ymin, xmax, ymax]
        pred_boxes (dict):
            dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float):
            value of IoU to consider as threshold for a true prediction.
        Returns:
            dict: true positives (int), false positives (int),
            false negatives (int)
        """
        # FIXME[todo]:
        # This implementation has been adapted from [x] and is probably
        # not the best approach (and maybe even not correct)
        # [x] https://towardsdatascience.com/
        #     evaluating-performance-of-an-object-detection-model-137a349c517b
        detections = []  # list of detections
        gt_idx_thr = []  # indices of ground truth objects that were detected
        pred_idx_thr = []  # indices of correct predictions (detections)
        ious = []  # the ious values for each detection.

        # find detections by comparing each ground truth object with
        # each prediction (cartesian product)
        for ipb, pred_box in enumerate(pred_boxes):
            for igb, gt_box in enumerate(gt_boxes):
                iou = self.intersection_over_union(gt_box, pred_box)

                if iou > iou_thr:
                    detections.append((igb, ipb, iou))
                    gt_idx_thr.append(igb)
                    pred_idx_thr.append(ipb)
                    ious.append(iou)

        if len(detections) == 0:
            # true_positive, false_positive, false_negative
            return (0, len(pred_boxes), len(gt_boxes))

        # Now clean the detections by removing double detections
        gt_match_idx = []
        pred_match_idx = []
        iou_sort = np.argsort(ious)[::1]
        for idx in iou_sort:
            gt_idx, pr_idx, _iou = detections[idx]

            # If the ground truth and predictions are both unmatched,
            # mark them (this ensures 1:1 mapping, but it may not be
            # the optimal mapping, isn't it?)
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                # ground truth object has not been marked as detected yet and
                # the detection has not yet been marked -> mark both of them
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)

        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)
        return tp, fp, fn  # true_positive, false_positive, false_negative

    def calc_precision_recall(image_results):
        """Calculates precision and recall from the set of images

        Arguments
        ---------
        img_results (dict):
            Iterable[Tuple[image_id,
                           true_pos: int, false_pos: int, false_neg: int]]
        Result
        -------
        tuple: of floats of (precision, recall)
        """
        true_positive = 0
        false_positive = 0
        false_negative = 0
        for _img_id, tp, fp, fn in image_results.items():
            true_positive += tp
            false_positive += fp
            false_negative += fn

        precision, recall = ((0.0, 0.0) if not true_positive else
                             (true_positive/(true_positive + false_positive),
                              true_positive/(true_positive + false_negative)))
        return (precision, recall)

    #
    # FIXME[old]:
    #

    def _preprocess_old(self, array: np.ndarray, **kwargs) -> np.ndarray:
        """Preprocess the image. This will resize the image to the
        target size of this tool, if such a size is set.
        """
        if array.ndim != 2 and array.ndim != 3:
            raise ValueError("The image provided has an illegal format: "
                             f"shape={array.shape}, dtype={array.dtype}")

        # if self._size is not None:
            # resize_ratio = array.shape[1]/400.0
            # array = imutils.resize(array, width=400)  # FIXME[hack]

        return super()._preprocess(array, **kwargs)

    def _adapt_detections(self, detections: Detections,
                          data: Data) -> Detections:

        if detections is None:
            return None

        # if we have scaled the input data, then we have to apply reverse
        # scaling to the detections.
        if self._size is not None:
            size = data.array.shape
            resize_ratio = max(self._size[0]/size[0], self._size[1]/size[1])
            detections.scale(resize_ratio)

        return detections

    def _postprocess_data(self, data: Data, mark: bool = False,
                          extract: bool = False, **_kwargs) -> None:
        """Apply different forms of postprocessing to the data object,
        extending it by additional tool specific attributes.

        Arguments
        ---------
        mark: bool
            Visually mark the detections in a copy of the image and
            store the result in the data object as the tool
            specific attribute `marked`.
        extract: bool
            Extract a list of image patches corresponding to the
            detections from the image and store the result in the
            data object as the tool specific attribute `extractions`.
        """
        if mark:
            self.mark_data(data)

        if extract:
            self.extract_data(data)

    def detect_regions(self, image: Imagelike) -> Iterable[Region]:
        """Iterate the :py:class:`Region`s detected by applying this
        :py:class:`ImageDetector` to an `image`.
        """
        metadata = self.detect(image)
        for region in metadata.regions:
            yield region

    #
    # Image specific methods
    #

    def detect_image(self, image: Imagelike, **_kwargs) -> Detections:
        """Apply the detector to the given image.

        Arguments
        ---------
        image:
            The image to be processed by this :py:class:`ImageDetector`.

        Result
        ------
        detectons:
            The detections obtained from the detector.
        """
        return self.detect(Image(image))

    def detect_and_show(self, image: Imagelike, **_kwargs) -> None:
        """Apply the detector to the given `image` and show the result.

        Arguments
        ---------
        image:
            The image to be processed by this :py:class:`ImageDetector`.
        """
        imshow(self.mark_image(image))

    def process_image(self, image: Imagelike, **kwargs) -> Image:
        """Create an :py:class:`Image` data object and process it with this
        :py:class:`ImageDetector`.

        Arguments
        ---------
        image:
            The image to be processed by this :py:class:`ImageDetector`.

        Result
        ------
        image:
            The processed image object. This object may have additional
            properties depending on the optional arguments passed to
            this function.
        """
        data = Image(image)
        self.apply(data, **kwargs)
        return data

    #
    # Marking detections
    #

    def mark_image(self, image: Imagelike, detections: Detections = None,
                   copy: bool = True, best: int = None,
                   group_size: int = 1) -> np.ndarray:
        """Mark the given detections in an image.

        Arguments
        ---------
        image: Imagelike
            The image into which the detections are to be drawn.
        detections: Detections
            The detections to draw.
        copy: bool
            A flag indicating if detections should be marked in
            a copy of the image (`True`) or into the original
            image object (`False`).
        best:
            If given, it will be the index of the "best" detection, that
            should then be marked in a different way.

        Result
        ------
        marked_image: np.ndarray
            An image in which the given detections are visually marked.
        """
        array = Image.as_array(image, copy=copy)
        # array.setflags(write=True)
        array = array.copy()  # FIXME[why]: was alread copied above
        if detections is None:
            detections = self.detect(Image(array))
        if detections:
            for idx, region in enumerate(detections.regions):
                if best is None or idx // group_size == best:
                    region.mark_image(array, color=(0, 255, 0))
                else:
                    region.mark_image(array, color=(255, 0, 0))
        return array
    
    def mark_data(self, data: Data, detections: Detections = None,
                  **kwargs) -> None:
        """Extend the given `Data` image object by a tool specific attribute,
        called `marked`, holding a copy of the original image in which
        the detections are marked. This function assumes that the
        detect has already be applied to the given data object and the
        detections are stored in a tool specific attribute called
        `detections`.

        Arguments
        ---------
        data: Data
            The data object do be marked.
        detections: Detections
            The detections to mark in the image. If None are provided
            the detections from the tools specific attribute `detections`
            is used.
        **kwargs:
            Additional keyword arguments to be passed to
            :py:meth:`mark_image`.
        """
        if detections is None:
            detections = self.detections(data)
        marked_image = self.mark_image(data.array, detections, copy=True,
                                       **kwargs)
        self.add_data_attribute(data, 'mark', marked_image)

    def marked_image(self, data) -> np.ndarray:
        """Get a version of the image with visually marked detections.
        This method assumes that this image has already be stored as
        an attribute to the data object, e.g., by calling the method
        :py:meth:`mark_data`, or by provding the argument `mark=True`
        when calling :py:meth:`process`.
        """
        return self.get_data_attribute(data, 'mark')

    #
    # Extracting detections
    #

    def extract_from_image(self, image: Imagelike, detections: Detections,
                           copy: bool = True) -> List[np.ndarray]:
        """Extract detections as a list of image patches from a given
        image.

        Arguments
        ---------
        image: Imagelike
            The image into which the detections are to be drawn.
        detections: Detections
            The detections to draw.
        copy: bool
            A flag indicating if extracted images should be realized
            as views using the same memory as the original image (False),
            or if real copies should be created (True). In some situations
            (e.g., if the detection includes invalid regions outside
            the image), only copy is valid and will be done
            no matter the value of this argument.

        Result
        ------
        extractions: List[np.ndarray]
            An list of extracted image regions.
        """
        array = Image.as_array(image)
        extractions = []
        if detections:
            for region in detections.regions:
                extractions.append(region.location.extract(array, copy=copy))
        return extractions

    def extract_data(self, data: Data,
                     detections: Detections = None) -> None:
        """Extend the given `Data` image object by a tool specific attribute,
        called `extractions`, holding a list of extracted image
        patches based on the detections done by this
        :py:class:`ImageDetector`. This function assumes that the
        detector has already be applied to the given data object and
        the detections are stored in a tool specific attribute called
        `detections`.

        Arguments
        ---------
        data: Data
            The data object do be marked.
        detections: Detections
            The detections to be extracted from the image. If None are
            provided the detections from the tools specific
            data attribute `detections` is used.

        """
        if detections is None:
            detections = self.detections(data)
        extractions = self.extract_from_image(data, detections)
        self.add_data_attribute(data, 'extract', extractions)

    def extractions(self, data) -> List[np.ndarray]:
        """Get a list of image patches extracted from the original image
        based on detections of this :py:class:ImageDetector`.
        This method assumes that this list has already been stored as
        an tool specific attribute `extractions` in the data object,
        e.g., by calling the method :py:meth:`extract_data`, or by provding
        the argument `extract=True` when calling :py:meth:`process`.
        """
        return self.get_data_attribute(data, 'extract')

    @staticmethod
    def select_best(detections, image: Imagelike) -> int:
        """Select the main face from a set of alternative detections.

        The criteria for selecting the main detection may vary
        depending on the data.  For example, if data is known to have
        the object at central prosition and/or with a specific size,
        these information can be used to select the best candidate.
        Also the confidence score provided by the detector may be
        used.

        Arguments
        ---------
        bounding_boxes:
            The bouding boxes returned by the detector.
        image:
            The image to which the bounding boxes refer.

        Result
        ------
        best:
            Index of the best detection.

        """
        # only consider the bounding boxes (some detectors may provide
        # other detections, like landmarks)
        bounding_boxes = [region.location for region in detections.regions
                          if isinstance(region.location, BoundingBox)]

        if not bounding_boxes:
            raise ValueError("No detection")

        if len(bounding_boxes) == 1:
            return 0  # there is only one detection -> assume that is correct

        # select bounding box with center closest to the image center
        shape = Image.as_shape(image)
        center = (shape[1]/2, shape[0]/2)
        best, distance = -1, float('inf')
        for index, bounding_box in enumerate(bounding_boxes):
            loc_center = bounding_box.center
            dist2 = ((loc_center[0]-center[0]) ** 2 +
                     (loc_center[1]-center[1]) ** 2)
            if dist2 < distance:
                best, distance = index, dist2

        return best


class BoundingBoxDetector:
    """Convenience methods for Detectors that report their detections
    as :py:class:`BoundingBox`.
    """

    def detect_boxes(self, image: Imagelike) -> Iterable[BoundingBox]:
        """Detect :py:class:`BoundingBox`s in an image.
        """
        metadata = self.detect(image)
        for region in metadata.regions:
            location = region.location
            if isinstance(location, BoundingBox):
                yield location
