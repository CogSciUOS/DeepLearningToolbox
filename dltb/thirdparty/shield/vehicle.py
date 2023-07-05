"""

name: vehicle-detection
channels:
  - defaults
  - conda-forge
  - anaconda
dependencies:
  - python=3.9
  - pytorch
  - torchvision=0.8.2
  - pillow=9.0.1
  - pip
  - pip:
    - opencv-python-headless
      # this headless version is for use with docker
      # if you are not using docker install a normal version of opencv
    - mmdet
    - openmim
    - Flask==2.0.1
    - flask_restful==0.3.9
    - python-dateutil==2.8.2
    - python-keycloak==0.25.0
    - waitress==2.0.0
    - PyYAML==6.0


mim install mmcv-full

The checkpoints can be downloaded from `Detector.CHECKPOINTS_URL`

testing:

img = imageio.imread("examples/vehicle.jpg")
img2 = model.preprocessing(img)


The file checkpoints.dvc""
------------------
outs:
- md5: 180c242d2e543a5fd2e053744c8b8839.dir
  size: 177867208
  nfiles: 2
  path: checkpoints
-----------------
        177867103
        d81764edc8fa416ccd2b620909960bb9

------

@package shield.pipeline.models.vehicle_detection.VehicleDetection
This module contains the VehicleDetection model, a subclass of the Model
Attributes:
    THRESHOLD: detection threshold where only results above the threshold
               will be used

------

@package shield.pipeline.models.vehicle_detection.vehicle_detection_api
This module contains the api for the vehicle detection model.

When this module is called as main, it will initialize the model and start
the api server.

Attributes:
    model: initialize the vehicle detection when the API gets started

"""
# standard imports
import os

# thirdparty imports
import numpy as np
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector
from mmdet.models import build_detector

# toolbox imports
# from ...base.implementation import Implementation
from . import VEHICLE_DETECTION, BoundingBox, BitMask
from . import ShieldDetector, ShieldSegmenter


class Detector(ShieldDetector, ShieldSegmenter):  # , Implementation
    # FIXME[bug]: adding Implementation to the base classes results in
    # an error
    """The submodel of the Model class for the VehicleDetection model
    Is able to return BoundingBoxes and Bitmasks.

    Attributes:
        name(str): Name of the model
        model_instance: The loaded Model

    Returns:
        labels(dict): dict{
            "BoundingBoxes": [BoundingBox],
            "BitMasks": [BitMask]}
    """

    CONFIG_PATH = 'experiments/shield/vehicle_detection/configs'
    CONFIG_NAME = 'mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco'
    CHECKPOINTS_PATH = 'experiments/shield'
    CHECKPOINTS_NAME = CONFIG_NAME + \
        '_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
    CHECKPOINTS_URL = (
        'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/' +
        CONFIG_NAME + '/' + CHECKPOINTS_NAME
    )

    THRESHOLD = 0.85  # Threshold for the vehicle detection

    def __init__(self, **kwargs):
        super().__init__(VEHICLE_DETECTION, **kwargs)

    def load_model(self):
        config = os.path.join(self.CONFIG_PATH, 'mask_rcnn',
                              self.CONFIG_NAME + '.py')
        checkpoint = os.path.join(self.CHECKPOINTS_PATH,
                                  self.CHECKPOINTS_NAME)

        device = "cpu"
        # Load the config
        config = mmcv.Config.fromfile(config)
        # Set pretrained to be None since we do not need pretrained model here
        config.model.pretrained = None
        # Initialize the detector
        model = build_detector(config.model)
        checkpoint = load_checkpoint(model, checkpoint, map_location=device)
        model.CLASSES = checkpoint["meta"]["CLASSES"]
        model.cfg = config
        model.to(device)
        model.eval()
        # lambda function so load_model only needs an image
        return lambda img: inference_detector(model, img)

    def preprocessing(self, image: np.ndarray) -> np.ndarray:
        return image

    def process_labels(self, results = None):
        labels = {"BoundingBoxes": [], "BitMasks": []}
        bbox_result, segm_result = results
        bboxes = bbox_result[2]
        bmasks = segm_result[2]

        for bbox, bmask in zip(bboxes, bmasks):
            if bbox[-1] > self.THRESHOLD:
                labels["BoundingBoxes"].append(
                    BoundingBox(bbox[0], bbox[1], bbox[2], bbox[3])
                )
                labels["BitMasks"].append(BitMask(bmask))

        return labels
