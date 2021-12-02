"""Torch implementation of MTCNN.  Taken from [1]


Test:
-----
from dltb.thirdparty.mtcnn2 import Detector
d = Detector()

from dltb.base.image import Image
#image = Image('/space/home/ulf/github/arcface-tf2/data/BruceLee.jpg')
image = Image('examples/reservoir-dogs.jpg')

metadata = d.detect(image)
type(metadata.regions[1].location)

for region in metadata.regions:
    region.mark_image(image)

from dltb.util.image import imshow
imshow(image)
# ---

from dltb.tool.face.mtcnn import Detector
d = Detector(module='mtcnn2')

# error: No implementation for class dltb.tool.face.detector.Detector
# could be loaded with module constraint
from dltb.tool.face.detector import Detector
list(Detector.implementations())
d = Detector(implementation='dltb.thirdparty.opencv.face.DetectorHaar',
             model_file='/space/home/ulf/virtenvs/cv-teach/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')


[1] https://github.com/ZhaoJ9014/face.evoLVe

"""
# standard imports
from typing import Tuple
import logging
import os

# third party imports
import numpy as np
import PIL

# toolbox imports
from dltb.config import config
from dltb.util.importer import Importer
from dltb.base.meta import Metadata
from dltb.base.data import Data
from dltb.base.image import Imagelike, BoundingBox, Size
from dltb.tool.face.mtcnn import Landmarks, Detector as DetectorBase

# logging
LOG = logging.getLogger(__name__)


class Detector(DetectorBase):
    """Torch-based implementation of the MTCNN detector from the
    face.evoLVe repository [1].

    [1] https://github.com/ZhaoJ9014/face.evoLVe
    """
    internal_arguments: Tuple[str, ...] = ('pil', )

    # Default reference facial points for crop_size = (96, 112); (this
    # is taken from the the face_evoLVe repository,
    # applications/align/align_trans.py. Note: According to the
    # comments in that file, these landmarks are for a "facial points
    # crop_size = (112, 112)", however, the crop size is then
    # specified as (96, 112), and this actually seems to be more
    # appropriate).  The coordinates are of the form (x, y), with the
    # origin located at the bottom left corner of the image.
    _reference_landmarks = Landmarks(points=np.asarray([
        [30.29459953,  51.69630051],   # left mouth
        [65.53179932,  51.50139999],   # right mouth
        [48.02519989,  71.73660278],   # nose
        [33.54930115,  92.3655014],    # left eye
        [62.72990036,  92.20410156]    # right eye
    ]))

    _reference_size = Size(96, 112)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._face_evolve_repository = \
            'https://github.com/ZhaoJ9014/face.evoLVe'
        self._face_evolve_directory = config.github_directory / 'face.evoLVe'
        self._module_detector = None

    def _prepared(self) -> bool:
        return (self._module_detector is not None) and super()._prepared()

    def _preparable(self) -> bool:
        return self._face_evolve_directory.is_dir() and super()._preparable()

    def _prepare(self) -> None:
        super()._prepare()

        # (1) load the model
        align_code_directory = \
            self._face_evolve_directory / 'applications' / 'align'
        self._module_detector = \
            Importer.import_module_from('detector',
                                        directory=align_code_directory)

    def _preprocess_data(self, data: Data, **kwargs):
        print("Preprocessing data")  # FIXME: is not called
        return PIL.Image.fromarray(data.array)

    def _preprocess(self, *args, **kwargs) -> Data:
        context = super()._preprocess(*args, **kwargs)
        context.add_attribute('pil', PIL.Image.fromarray(context.input_image))
        return context

    def _detect(self, image: PIL.Image, **kwargs) -> Metadata:
        """Apply the MTCNN detector to detect faces in the given image.

        Arguments
        ---------
        image:
            The image to detect faces in. Expected is a RGB image
            with np.uint8 data.

        Returns
        ------
        metadata: Metadata
            A Metadata structure in which BoundingBoxes and
            FacialLandmarks are provided, annotated with a numeric 'id'
            and a 'confidence' value.
        """
        # FIXME[hack]:
        if isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(image)
        #
        # (1) Run the MTCNN detector
        #
        try:
            # FIXME[question]: what is going on here? can this be done
            # in prepare?
            align_code_directory = \
                self._face_evolve_directory / 'applications' / 'align'
            prev_cwd = os.getcwd()
            os.chdir(align_code_directory)
            LOG.info("MTCNN: detecting facess ...")
            bounding_boxes, landmarks = \
                self._module_detector.detect_faces(image)
            LOG.info("MTCNN: ... found %d faces.", len(bounding_boxes))
        finally:
            os.chdir(prev_cwd)

        #
        # (2) Create Metadata
        #
        self.detect_boxes = True
        self.detect_landmarks = True

        detections = Metadata(
            description='Detections by the Torch MTCNN detector')
        for face_id, (bbox, mark) in enumerate(zip(bounding_boxes, landmarks)):
            confidence = bbox[4]

            if self.detect_boxes:
                # The bounding boxes are reported as a 5-tuple:
                # (x1, y1, x2, y2, confidence)
                detections.add_region(BoundingBox(x1=bbox[0], y1=bbox[1],
                                                  x2=bbox[2], y2=bbox[3]),
                                      confidence=confidence, id=face_id)

            if self.detect_landmarks:
                # landmarks are reported as array of length 10, consisting
                # of 5 consecutive x-coordinates followed by 5 corresponding
                # y-coordinates.  The order of these pairs
                # is: left_mouth, right_mouth, nose, left_eye, right_eye
                # That is the array holds
                #    [left_mouth_x, right_mouth_x, nose_x,
                #     left_eye_x, right_eye_x,
                #     left_mouth_y, right_mouth_y, nose_y, 
                #     left_eye_y, right_eye_y]
                points = mark.reshape(2, 5).T.astype(int)
                detections.add_region(Landmarks(points=points),
                                      confidence=confidence, id=face_id)

        return detections
