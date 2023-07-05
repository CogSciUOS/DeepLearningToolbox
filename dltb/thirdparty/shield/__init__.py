""" @package shield.pipeline.models.Model
This module contains the abstract Model superclass


Essentially:

model = ModelClass()


preprocessed_img = self.preprocessing(image)
labels = self.model_instance(preprocessed_img)
processed_labels = self.process_labels(labels)


* add extra **kwargs and super.__init__(..., **kwargs) call
* rename __call__ to shield_cal

"""
# standard imports
from abc import abstractmethod

# thirdparty imports
import numpy as np

# toolbox imports
from ...base.implementation import Implementable
from ...base.image import BoundingBox as DltbBoundingBox
from ...tool.detector import ImageDetector
from ...tool.segmenter import ImageSegmenter, Segmentation
from ...base.metadata import Metadata


#
# constants
#

VEHICLE_DETECTION = "VehicleDetection"
FACE_DETECTION = "FaceDetection"
PERSON_DETECTION = "PersonDetection"
TEXT_DETECTION = "TextDetection"
LICENSE_PLATE_DETECTION = "LicensePlateDetection"
SCREEN_DETECTION = "ScreenDetection"
DEEPFACE_DETECTION = "DeepFaceDetection"
MODELS = [
    VEHICLE_DETECTION,
    FACE_DETECTION,
    PERSON_DETECTION,
    TEXT_DETECTION,
    LICENSE_PLATE_DETECTION,
    SCREEN_DETECTION,
]

URL_PIPELINE = "http://pipeline:8080/shield/pipeline/"

URL_MODELS = {
    VEHICLE_DETECTION: "http://vehicle-detection:8081/shield/model/vehicle-detection/",
    FACE_DETECTION: "http://face-detection:8082/shield/model/face-detection/",
    LICENSE_PLATE_DETECTION: "http://licenseplate-detection:8083/shield/model/license-plate-detection/",
    SCREEN_DETECTION: "http://screen-detection:8084/shield/model/screen-detection/",
    TEXT_DETECTION: "http://text-detection:8085/shield/model/text-detection/",
    PERSON_DETECTION: "http://person-detection:8086/shield/model/person-detection/",
}

URL_HEALTH_CHECKS = {
    VEHICLE_DETECTION: "http://vehicle-detection:8081/shield/service/vehicle-detection/",
    FACE_DETECTION: "http://face-detection:8082/shield/service/face-detection/",
    LICENSE_PLATE_DETECTION: "http://licenseplate-detection:8083/shield/service/license-plate-detection/",
    SCREEN_DETECTION: "http://screen-detection:8084/shield/service/screen-detection/",
    TEXT_DETECTION: "http://text-detection:8085/shield/service/text-detection/",
    PERSON_DETECTION: "http://person-detection:8086/shield/service/person-detection/",
}

# ----- Anonymisations -----
BLURRING = "Blurring"
MASKING = "Masking"
TATTOO = "TattooAnonymization"
FACE = "FaceAnonymization"

MASKBITMASK = "MaskImageWithBitMask"
MASKBBOX = "MaskImageWithEdgePoints"
BLURBBOX = "BlurImageWithEdgePoints"

ANON_METHODS = [MASKING, BLURRING, TATTOO, FACE]

ANON_METHODS_GENERAL = [
    (None, None),
    (BLURRING, BLURRING),
    (MASKING, MASKING),
]
ANON_METHODS_FACE = [
    (None, None),
    (BLURRING, BLURRING),
    (MASKING, MASKING),
    (TATTOO, TATTOO),
    (FACE, FACE),
]
ANON_METHODS_PERSON = [
    (None, None),
    (BLURRING, BLURRING),
    (MASKING, MASKING),
    (TATTOO, TATTOO),
]

#
# Boundingbox
#

class BoundingBox(dict):
    """The Bounding Box class for the coordinate based Bounding Boxes of
    the labels.

    This class is used for a consistent bounding box
    structure. Creates a bounding box with two points. So the upper
    left and lower right corner. First it checks if the first x or y
    has the lower value, or changes them accordingly

    Attributes:
        x_1 (int): First x value
        y_1 (int): First y value
        x_2 (int): Second x value
        y_2 (int): Second y value

    """

    def __init__(self, x_1: int, y_1: int, x_2: int, y_2: int):
        x_1, y_1, x_2, y_2 = self.check_coordinates(x_1, y_1, x_2, y_2)
        # seems redundant, but needed for jsonification
        dict.__init__(
            self,
            x_1=int(x_1),
            y_1=int(y_1),
            x_2=int(x_2),
            y_2=int(y_2),
        )
        self.x_1 = int(x_1)
        self.y_1 = int(y_1)
        self.x_2 = int(x_2)
        self.y_2 = int(y_2)

    def check_coordinates(self, x_1, y_1, x_2, y_2) -> tuple:
        """Checks if the coordinates in the right order

        Checks if the coordinates are in the corrected order, and the
        boundingbox is created properly, so always with the lower value
        first and then the higher. If that is not the case, rearrange
        the coordinates in a correct order

        Args:
            x_1 (int): First x value
            y_1 (int): First y value
            x_2 (int): Second x value
            y_2 (int): Second y value

        Returns:
            tuple: The correct placed coordinates
        """
        min_x = min(x_1, x_2)
        max_x = max(x_1, x_2)
        min_y = min(y_1, y_2)
        max_y = max(y_1, y_2)
        return (min_x, min_y, max_x, max_y)

    def get_coordinates(self) -> dict:
        """Returns all coordinates as dict as upper_left and lower_right
        Returns:
            dict: As {"upper_left": (x_1, y_1), "lower_right": (x_2, y_2)}
        """
        return {"upper_left": (self.x_1, self.y_1), "lower_right": (self.x_2, self.y_2)}

    def get_coordinates_as_tuple(self) -> tuple:
        """Returns the coordinates as tuple as upper left x y and lower right x y
        Returns:
            tuple: As (x_1, y_1, x_2, y_2)
        """
        return (self.x_1, self.y_1, self.x_2, self.y_2)

    def get_upper_x(self) -> int:
        """Returns the upper left x coordinate"""
        return self.x_1

    def get_upper_y(self) -> int:
        """Returns the upper left y coordinate"""
        return self.y_1

    def get_lower_x(self) -> int:
        """Returns the lower right x coordinate"""
        return self.x_2

    def get_lower_x(self) -> int:
        """Returns the lower right y coordinate"""
        return self.y_2

    def __str__(self) -> str:
        """Function for printing methods"""
        return f"BoundingBox: upper_x-{self.x_1}, upper_y-{self.y_1}, lower_x-{self.x_2}, lower_y-{self.y_2}"

    def __repr__(self) -> str:
        """Function for printing methods"""
        return f"BoundingBox: upper_x-{self.x_1}, upper_y-{self.y_1}, lower_x-{self.x_2}, lower_y-{self.y_2}"

#
# Bitmask
#
from pickle import FALSE

class BitMask(dict):
    """The Bit Mask class for a consistent bit mask representation

    Needs a bit Mask as an input with just bool values, otherwise returns
    an TypeError.

    Attributes:
        bitmask: The bitmask the BitMask object contains

    Attributes:
        bitmask: The bitmask as list
    """

    def __init__(self, bitmask):
        self.bitmask = self.check_bit_mask(bitmask)
        dict.__init__(self, bitmask=self.bitmask)

    def check_bit_mask(self, bitmask):
        """Check if bitmask is in the right format(bool)

        Args:
            bitmask: An array that should be checked

        Returns:
            The controlled array

        Raise:
            TypeError: if the bitmask was not bool
            ValueError: if the bitmask was empty
        """
        if not isinstance(bitmask, np.ndarray):
            bitmask = np.asarray(bitmask)
        else:
            pass

        try:
            if bitmask.dtype == bool:
                return bitmask.tolist()
            else:
                raise TypeError(
                    f"The bit mask has the wrong dtype. Is {bitmask.dtype} not bool"
                )
        except:
            raise ValueError(f"The bit mask is empty.")

    def get_bitmask(self) -> np.ndarray:
        """Returns the bit mask as np.ndarry

        Returns:
            np.ndarray: the actual bitmask of this BitMask object
        """
        return np.asarray(self.bitmask)

    def __str__(self) -> str:
        """Function for printing methods"""
        npmask = np.array(self.bitmask)
        return f"BitMask with {np.count_nonzero(npmask == True)} TRUE and {np.count_nonzero(npmask==False)} FALSE"

    def __repr__(self) -> str:
        """Function for printing methods"""
        npmask = np.array(self.bitmask)
        return f"BitMask with {np.count_nonzero(npmask == True)} TRUE and {np.count_nonzero(npmask==False)} FALSE"


class Model:
    """This is the abstract superclass for all models

    Every submodel class inherits from this class. All subclasses have
    to overwrite the load_model(), preprocessing() and process_label()
    functions.  The call method uses these functions to process the
    image and return the corresponding BoundingBoxes and BitMasks as a
    dictionary.  First the model gets loaded. When the model is then
    called, the image runs through the preprocessing, then through the
    model and then the results will be processed.

    Args:
        name(str): Name to instantiate the model with
        image(np.ndarray): The image to be processed when model is called

    Attributes:
        name(str): Name of the model
        model_instance: The loaded model

    Returns:
            dict{"BoundingBoxes":[BoundingBox], "BitMasks":[BitMask]}

    """

    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.model_instance = self.load_model()

    def shield_call(self, image: np.ndarray) -> dict:
        preprocessed_img = self.preprocessing(image)
        labels = self.model_instance(preprocessed_img)
        processed_labels = self.process_labels(labels)
        return processed_labels

    @abstractmethod
    def load_model(self):
        """This method loads the model and weights.

        Returns:
            function: a callable version of the model that takes just an image
                      as input.
        """

    @abstractmethod
    def preprocessing(self, image: np.ndarray):
        """Preprocesses the image to prepare it for the model.

        Args:
            image(np.ndarray): The image to be preprocessed
 
        Returns:
            image(np.ndarray): The preprocessed image
        """

    @abstractmethod
    def process_labels(self) -> dict:
        """Transforms the output of the model into BoundingBox and BitMask objects.

        Processes and filters the detection results from the models.
        Returns a dictionary with all the BoundingBox and BitMask objects,
        in a list, or if there is none as Empty list.

        Returns:
            dict{"BoundingBoxes":[BoundingBox], "BitMasks":[BitMask]}
        """

    def get_name(self):
        """Getter function for the name of the model class"""
        return self.name
   

class ShieldDetector(Model, ImageDetector, Implementable):
    """
    """
    
    def _detect(self, image: np.ndarray, **kwargs) -> Metadata:
        """This is the post request entry into the vehicle_detection model

        This function processes the received post request from the
        pipeline. Therefore it reads the image from the files["file"]
        field and decode it to numpy array and send it into the
        vehicle detection model. The results of boundingboxes and
        bitmasks the model returns, are send as an json dict back to
        the pipeline.

        Returns
        -------
        json: The boundingboxes and bitmasks as dict inside a json
        """

        results = self.shield_call(image)
        bboxes, _bitmasks = results['BoundingBoxes'], results['BitMasks']
        # {'BoundingBoxes': bboxes, 'BitMasks': bitmasks}
        # bboxes: List[shield.utils.BoundingBox.BoundingBox]
        # bitmasks: List[shield.utils.BitMask.BitMask]

        detections = Metadata(
            description=("Detections by the Shield detector "
                         f"{self.__class__.__name__}"))
        for bbox in bboxes:
            detections.add_region(DltbBoundingBox(x1=bbox.x_1, y1=bbox.y_1,
                                                  x2=bbox.x_2, y2=bbox.y_2))
        return detections


class ShieldSegmenter(Model, ImageSegmenter, Implementable):

    def _segment(self, image: np.ndarray, **kwargs) -> Segmentation:
        """This is the post request entry into the vehicle_detection model

        This function processes the received post request from the
        pipeline. Therefore it reads the image from the files["file"]
        field and decode it to numpy array and send it into the
        vehicle detection model. The results of boundingboxes and
        bitmasks the model returns, are send as an json dict back to
        the pipeline.

        Returns
        -------
        json: The boundingboxes and bitmasks as dict inside a json
        """

        results = self.shield_call(image)
        _bboxes, bitmasks = results['BoundingBoxes'], results['BitMasks']
        # {'BoundingBoxes': bboxes, 'BitMasks': bitmasks}
        # bboxes: List[shield.utils.BoundingBox.BoundingBox]
        # bitmasks: List[shield.utils.BitMask.BitMask]

        return Segmentation(bitmasks=bitmasks)
