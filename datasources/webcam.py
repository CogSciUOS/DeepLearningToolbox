from datasources import DataSource, InputData


class DataWebcam(DataSource):
    """A data source fetching images from the webcam.

    Attributes
    ----------
    _capture:
        A capture object
    """
    _capture = None

    @staticmethod
    def check_availability():
        """Check if this Datasource is available.

        Returns
        -------
        True if the OpenCV library is available, False otherwise.
        """
        return importlib.util.find_spec('cv2')

    def __init__(self):
        """Create a new DataWebcam

        Raises
        ------
        ImportError:
            The OpenCV module is not available.
        """
        super().__init__()
        global VideoCapture
        from cv2 import VideoCapture
        # self._capture = VideoCapture(0)

    def __getitem__(self, index):
        capture = VideoCapture(0)
        ret, frame = capture.read()
        capture.release()
        if not ret:
            raise RuntimeError("Video Capture failed!")
        return InputData(frame, "Webcam")

    def __len__(self):
        # FIXME[hack]
        return 100
