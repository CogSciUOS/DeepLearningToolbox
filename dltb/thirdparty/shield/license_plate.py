"""
https://web.inf.ufpr.br/vri/publications/layout-independent-alpr/

wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/vehicle-detection.cfg
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/vehicle-detection.data
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/vehicle-detection.weights
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/vehicle-detection.names
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/lp-detection-layout-classification.cfg
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/lp-detection-layout-classification.data
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/lp-detection-layout-classification.weights
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/lp-detection-layout-classification.names
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/lp-recognition.cfg
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/lp-recognition.data
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/lp-recognition.weights
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/lp-recognition.names
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/sample-image.jpg
wget http://www.inf.ufpr.br/vri/databases/layout-independent-alpr/data/README.txt


"""# standard imports
import os

# thirdparty imports
import numpy as np
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector
from mmdet.models import build_detector

# toolbox imports
# from ...base.implementation import Implementation
from . import LICENSE_PLATE_DETECTION, BoundingBox, BitMask
from . import ShieldDetector, ShieldSegmenter

from pathlib import Path

import cv2
import numpy as np
import torch

MODULE_PATH = Path('/home/ulf/uni/teaching/2022-sp-privacy/code/shield/pipeline/models/license_plate_detection')

MODEL_PATH = Path('/home/ulf/tmp/data')

import sys
sys.path.insert(0, str(MODULE_PATH))

from pytorch_yolo2.cropping import crop_img, last_coordinates
from pytorch_yolo2.model.darknet import Darknet
from pytorch_yolo2.utils import detect_, load_class_names



class Detector(ShieldDetector):
    """LicensePlateDetection class for detecting license plates.

    Uses the created load_model function to load the weights,
    configuration needed for the use of the vehicle detection
    and license plate detection models.
    The call function is called by the pipeline and takes
    an image and puts it through the vehicle and license
    plate detectionmodels. With the preprocessing, the
    actual coordinates creation and the transformation
    of these coordinates into the correct shape,
    it returns a dictionary, containing the coordinates in
    BoundingBoxes and in BitMasks.

    Args:
        image(np.ndarray): The Image to put into the model

    Attributes:
        name(str): Name of the model
        conf_thresh_vehicle(int): minimum of confidence level on vehicle detection.
        conf_thresh_lp(int): minimum of confidence level on license plate detection.
        cuda(bool): default False.

    Returns:
        processed_labels(dict): dictionary with all the BoundingBox and
            BitMask objects, in a list, or if there is none as Empty list.
    """

    def __init__(
        self,
        cuda: bool = False,
        conf_thresh_vehicle: float = 0.5,
        conf_thresh_lp: float = 0.5,
        **kwargs
    ):
        super().__init__( LICENSE_PLATE_DETECTION, **kwargs)
        self.name = LICENSE_PLATE_DETECTION

        # model constants
        self.vehicle_detection_cfg = (
            MODEL_PATH / "vehicle-detection.cfg"
        )
        self.vehicle_detection_weights = (
            MODEL_PATH / "vehicle-detection.weights"
        )
        self.lp_detection_cfg = (
            MODEL_PATH / "lp-detection-layout-classification.cfg"
        )
        self.lp_detection_weights = (
            MODEL_PATH / "lp-detection-layout-classification.weights"
        )

        # class/annotation names for the models
        self.path_lp_detection = (
            MODEL_PATH / "lp-detection-layout-classification.names"
        )
        self.path_vehicle = \
            MODEL_PATH / "vehicle-detection.names"

        # model parameters
        self.conf_thresh_vehicle = conf_thresh_vehicle
        self.conf_thresh_lp = conf_thresh_lp
        self.cuda = cuda

        # models
        self.model_instance_vehicle_detection = self.load_model_vehicle_detection()
        self.model_instance_license_plate_detection = (
            self.load_model_license_plate_detection()
        )

    def shield_call(self, image: np.ndarray) -> dict:
        # pass the image through the first model
        preprocessed_img = self.preprocessing_vehicle_input_image(image)
        labels_vehicle_detection = self.model_instance_vehicle_detection(
            preprocessed_img
        )

        # pass the processed image through the second model
        processed_patched_images = self.preprocessing_license_plate_input_image(
            labels_vehicle_detection, preprocessed_img
        )
        labels_license_plate_detection = [
            self.model_instance_license_plate_detection(img)
            for img in processed_patched_images
        ]

        # change output of the model into pipeline data classes
        processed_labels = self.process_labels(
            labels_vehicle_detection, labels_license_plate_detection, image
        )

        return processed_labels

    def load_model_vehicle_detection(self):
        """This method loads the vehicle detection model.

        Returns:
            labels_vehicle_detection(list): list with class and vehicle
            detection bounding boxes coordinates
        """
        # loading the vehicle detection model
        self.vehicle_detection = Darknet(self.vehicle_detection_cfg)
        self.vehicle_detection.load_weights(self.vehicle_detection_weights)
        self.vehicle_detection.eval()
        device = torch.device("cuda" if self.cuda else "cpu")
        self.vehicle_detection.to(device)

        # wrapper function to use the detect function on images (np.arrays)
        class_names = load_class_names(self.path_vehicle)
        model = lambda img: detect_(
            self.vehicle_detection,
            img,
            self.conf_thresh_vehicle,
            class_names,
            self.cuda,
        )

        return model

    def load_model_license_plate_detection(self):
        """This method loads the licence plate detection model.

        Returns:
            labels_license_plate_detection(list): list with class and
            license plate detection bounding boxes coordinates
        """
        # loading the vehicle detection model
        self.lp_detection = Darknet(self.lp_detection_cfg)
        self.lp_detection.load_weights(self.lp_detection_weights)
        self.lp_detection.eval()

        device = torch.device("cuda" if self.cuda else "cpu")
        self.lp_detection.to(device)

        # detecting the license plate in the image with a confidence of 0.5
        class_names = load_class_names(self.path_lp_detection)

        return lambda img: detect_(
            self.lp_detection, img, self.conf_thresh_lp, class_names, self.cuda
        )

    def preprocessing_vehicle_input_image(self, image: np.ndarray):
        """Preprocessing steps for the image for the model to function

        Resized the input image following the width and height of the
        vehicle detection model.

        Args:
            image(np.ndarray): input image fed to the model.

        Returns:
            image(np.ndarray): resized image with shape of the vehicle
                detection model.

        """
        image = cv2.resize(
            image, (self.vehicle_detection.width, self.vehicle_detection.height)
        )

        return image

    def preprocessing_license_plate_input_image(self, list_: list, image: np.ndarray):
        """Preprocessing steps for the images for the model to function

        Crops each image given the image patches of the detected bonding box and save
        this cropped image in a list.

        Args:
            list(list): list with class and vehicle detection bounding
                boxes coordinates
            img(np.ndarray): image loaded with cv2

        Returns:

            images(np.ndarray: a cropped image given the coordinates of the
                detected bounding box.

        """
        images = crop_img(image, list_)

        return images

    def process_labels(
        self, labels_vehicle_detection: list, labels_license_plate_detection: list, img
    ) -> dict:
        """Processing the output of the model to later fit in
         the anonymization process.

        This method takes the lists with coordinates of the
        bounding boxes of both vehicle detection and license plate
        and returns a dictionary with all the BoundingBox and BitMask
        objects, in a list, or if there is none as Empty list.

        Args:
            labels_vehicle_detection(list): list with class and
                vehicle detection bounding boxes coordinates
            labels_license_plate_detection(list): list with class and license
                plate bounding boxes coordinates

        Returns:
            labels(dict):  dictionary with all the BoundingBox and
                BitMask objects, in a list, or if there is none as Empty list
        """
        # create a list with the license plate bounding boxes
        # coordinates with respect to the original input image.
        coordinates = last_coordinates(
            labels_vehicle_detection,
            labels_license_plate_detection,
            img,
            self.vehicle_detection.width,
            self.vehicle_detection.height,
        )

        labels = {"BoundingBoxes": [], "BitMasks": []}

        for list_ in coordinates:
            x_1, y_1, x_2, y_2 = list_
            x_1, y_1, x_2, y_2 = (
                int(float(x_1)),
                int(float(y_1)),
                int(float(x_2)),
                int(float(y_2)),
            )
            labels["BoundingBoxes"].append(BoundingBox(x_1, y_1, x_2, y_2))

        return labels

    def get_name(self):
        """Getter function for the name of the model class"""

        return self.names
