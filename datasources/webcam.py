from . import Datasource, Predefined, InputData
import importlib

class DataWebcam(Datasource, Predefined):
    """A data source fetching images from the webcam.

    Attributes
    ----------
    _capture:
        A capture object
    """
    _device: int = 0
    _capture = None # cv2.Capture

    @staticmethod
    def check_availability():
        """Check if this Datasource is available.

        Returns
        -------
        True if the OpenCV library is available, False otherwise.
        """
        return importlib.util.find_spec('cv2')

    def __init__(self, device: int=0):
        """Create a new DataWebcam

        Raises
        ------
        ImportError:
            The OpenCV module is not available.
        """
        super().__init__("<Webcam>")
        Predefined.__init__(self, "Noise")

    def __getitem__(self, index):
        if not self.prepared:
            return InputData(None, None)

        ret, frame = self._capture.read()
        if not ret:
            raise RuntimeError("Reading an image from video capture failed!")
        return InputData(frame[:,:,::-1], "Webcam")

    @property
    def prepared(self) -> bool:
        """Report if this Datasource prepared for use.
        A Datasource has to be prepared before it can be used.
        """
        return self._capture is not None

    def prepare(self):
        """Prepare this Datasource for use.
        """
        if self.prepared:
            return  # nothing to do

        from cv2 import VideoCapture
        self._capture = VideoCapture(self._device)
        if not self._capture:
            raise RuntimeError("Acquiring video capture failed!")
            
        ret, frame = self._capture.read()
        if not ret:
            self._capture = None
            raise RuntimeError("Reading an image from video capture failed!")
        self.change('state_changed')

    def unprepare(self):
        """Unprepare this Datasource for use.
        """
        if not self.prepared:
            return
        self._capture.release()
        self._capture = None
        self.change('state_changed')


