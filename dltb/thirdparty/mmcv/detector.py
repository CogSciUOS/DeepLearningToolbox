"""
See
* https://github.com/open-mmlab/mmcv


Installation:

 - mmdet
 - openmim

mim install mmcv-full


conda install -c esri mmcv-full

The checkpoints can be downloaded from `Detector.CHECKPOINTS_URL`
"""
# No CUDA runtime is found, using CUDA_HOME='/usr'
# -> check: torch.cuda.is_available()

# UserWarning: On January 1, 2023, MMCV will release v2.0.0, in which
# it will remove components related to the training process and add a
# data transformation module. In addition, it will rename the package
# names mmcv to mmcv-lite and mmcv-full to mmcv. See
# https://github.com/open-mmlab/mmcv/blob/master/docs/en/compatibility.md
# for more details.
# => Hack: ignore warning

# UserWarning: "ImageToTensor" pipeline is replaced by
# "DefaultFormatBundle" for batch inference. It is recommended to
# manually replace it in the test data pipeline in your config file.
#
# => Hack: ignore warning
# => Solution: adapt the files in Detector.CONFIG_PATH

# libGL error: MESA-LOADER: failed to open swrast:
# /usr/lib/dri/swrast_dri.so: Kann die Shared-Object-Datei nicht
# öffnen: Datei oder Verzeichnis nicht gefunden (search paths
# /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix
# _dri)
# libGL error: failed to load driver: swrast
#
# => Solution 1: conda install -c conda-forge libstdcxx-ng

# standard imports
import os

# FIXME[hack]
import warnings
warnings.filterwarnings(action='ignore', module=r'mmcv')
warnings.filterwarnings(action='ignore', module=r'mmdet.datasets.utils')

# thirdparty imports
import numpy as np
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector
from mmdet.models import build_detector

# toolbox imports
# from ...base.implementation import Implementation
from ...base.image import BoundingBox
from ...base.metadata import Metadata
from ...tool.detector import ImageDetector


class Detector(ImageDetector):  # , Implementation
    """An MMVC based image detector.
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
        super().__init__(**kwargs)
        self._model = None
        self._prepare()

    def _prepare(self) -> None:
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

        self._model = model

    def _detect(self, data: np.ndarray, **kwargs) -> Metadata:

        bbox_result, segm_result = inference_detector(self._model, data)

        bboxes = bbox_result[2]
        bmasks = segm_result[2]

        detections = Metadata(
            description=("Detections by the MMVC detector "
                         f"{self.__class__.__name__}"))

        for bbox, _bmask in zip(bboxes, bmasks):
            if bbox[-1] > self.THRESHOLD:
                detections.add_region(BoundingBox(x1=bbox[0], y1=bbox[1],
                                                  x2=bbox[2], y2=bbox[3]))
                # labels["BitMasks"].append(BitMask(bmask))

        return detections
