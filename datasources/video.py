import importlib.util

from datasources import DataSource, InputData


class DataVideo(DataSource):
    """A data source fetching frames from a video.

    Attributes
    ----------
    _capture: cv2.VideoCapture
        A capture object
    """
    _capture = None
    _filename: str = None

    @staticmethod
    def check_availability() -> bool:
        """Check if this Datasource is available.

        Returns
        -------
        True if the OpenCV library is available, False otherwise.
        """
        return importlib.util.find_spec('cv2')

    def __init__(self, filename: str):
        """Create a new DataWebcam
        """
        super().__init__()
        self._filename = filename

        from cv2 import VideoCapture
        self._capture = VideoCapture(filename)
        if not self._capture.isOpened():
            print("could not open :", filename)
        print(type(self._capture))
        print(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(self._capture.get(cv2.CAP_PROP_FPS))
        self._description = "Frames from the video \"{}\"".format(filename)

    def __getitem__(self, index):
        self._capture.set(1, index)
        ret, frame = self._capture.read()
        return InputData(frame, "Video-Frame:" + str(index))

    def __len__(self):
        frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # FIXME[hack]: sometime CAP_PROP_FRAME_COUNT does only return 0
        return frames if frames > 0 else 1000
