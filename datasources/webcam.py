from . import Predefined, InputData, Loop

import importlib
import numpy as np

class DataWebcam(Predefined, Loop):
    """A data source fetching images from the webcam.

    Attributes
    ----------
    _capture:
        A capture object
    """
    _device: int = 0
    _capture = None # cv2.Capture
    _frame: np.ndarray = None

    @staticmethod
    def check_availability():
        """Check if this Datasource is available.

        Returns
        -------
        True if the OpenCV library is available, False otherwise.
        """
        return importlib.util.find_spec('cv2')

    def __init__(self, id: str="Webcam", description: str="<Webcam>",
                 device: int=0, **kwargs):
        """Create a new DataWebcam

        Raises
        ------
        ImportError:
            The OpenCV module is not available.
        """
        super().__init__(id=id, description=description, **kwargs)

    @property
    def prepared(self) -> bool:
        """Report if this Datasource prepared for use.
        A Datasource has to be prepared before it can be used.
        """
        return self._capture is not None

    def _prepare_data(self):
        """Prepare this Datasource for use.
        """
        from cv2 import VideoCapture
        self._capture = VideoCapture(self._device)
        if not self._capture:
            raise RuntimeError("Acquiring video capture failed!")

    def _unprepare_data(self):
        """Unprepare this Datasource for use.
        """
        self._capture.release()
        self._capture = None
        self._frame = None

    @property
    def fetched(self):
        return self._frame is not None

    def _fetch(self, **kwargs):
        ret, frame = self._capture.read()
        if not ret:
            raise RuntimeError("Reading an image from video capture failed!")
        self._frame = frame[:,:,::-1]

    def _get_data(self):
        return self._frame

    def __str__(self):
        return "Webcam"
