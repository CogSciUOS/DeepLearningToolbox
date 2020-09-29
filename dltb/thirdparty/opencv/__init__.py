"""Interface to the OpenCV library. This module provides different
classes mapping the toolbox API onto the opencv library.
"""

# standard imports
from typing import Union, Iterator, Tuple, List
import logging
import threading

# third party imports
import cv2
import numpy as np

# toolbox imports
from ...base import image, video

# logging
LOG = logging.getLogger(__name__)


class OpenCV:
    """
    """

    def info(self) -> str:
        """
        """
        return cv2.getBuildInformation()

    def check_url_https(self) -> bool:
        """According to the OpenCV documentation, starting with OpenCV version
        3.2.0 the filename provided to the VideoCapture constructor
        can be an URL. However, I experienced problems with the
        HTTPS-protocol using OpenCV version 3.4.2 (wheres the HTTP
        protocol works).  With OpenCV version 4.2.0 (either from pypi
        or from conda-forge).

            url = 'http://example.com/test.mp4'
            url = 'https://example.com/test.mp4'
            cam = cv2.VideoCapture(url)
            cam.isOpened()
            cam.release()

        Observation:

          http   https                                    OpenCV
         -------------------------------------------------------
          True   False                  opencv=3.4.2      3.4.2
          True   False    anaconda      opencv=3.4.2      3.4.2
          True   False    conda-forge   opencv=3.4.2      3.4.2
          True   True     conda-forge   opencv=4.2.0      4.2.0
          True   True     pypi   opencv-python=4.2.0.34   4.2.0

        """
        pass


class ImageIO(image.ImageReader, image.ImageWriter, image.ImageDisplay):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._window = None

    def __del__(self) -> None:
        cv2.destroyAllWindows()
        super().__del__()

    def read(self, filename: str, **kwargs) -> np.ndarray:
        # cv2.imread(path, flag=cv2.IMREAD_COLOR)
        #
        # cv2.IMREAD_COLOR:
        #    It specifies to load a color image. Any transparency of
        #    image will be neglected. It is the default
        #    flag. Alternatively, we can pass integer value 1 for this
        #    flag.
        #
        # cv2.IMREAD_GRAYSCALE:
        #    It specifies to load an image in grayscale
        #    mode. Alternatively, we can pass integer value 0 for this
        #    flag.
        #
        # cv2.IMREAD_UNCHANGED:
        #    It specifies to load an image as such including alpha
        #    channel. Alternatively, we can pass integer value -1 for
        #    this flag.
        return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)

    def write(self, image: np.ndarray, filename: str, **kwargs) -> None:
        cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def show(self, image: np.ndarray, **kwargs) -> None:
        cv2.imshow('Image', image)


class ImageUtils(image.ImageResizer):

    def resize(self, image: np.ndarray, size=(640, 360)) -> np.ndarray:
        """Resize the frame to a smaller resolution to save computation cost.
        """
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


class VideoReader(video.Reader):
    """A :py:class:`VideoReader` realized by an OpenCV
    :py:class:`VideoCapture` object. This is a superclass of
    :py:class:`VideoFileReader` and :py:class:`Webcam`, providing
    common functionality.

    Attributes
    ----------
    _capture: cv2.VideoCapture
        A capture object
    _lock: threading.Lock
        A lock to avoid race conditions due to simultanous access to
        the video capture object.
    """

    def __init__(self, init: str, **kwargs) -> None:
        """Constructor for this :py:class:`VideoBackend`.
        The underlying :py:class:`cv2.VideoCapture` object will be
        created.
        """
        super().__init__(**kwargs)
        self._capture = cv2.VideoCapture(init)
        if not self._capture:
            raise RuntimeError("Creating video capture object for "
                               f"'{init}' failed.")
        if not self._capture.isOpened():
            raise RuntimeError("Opening the video capture object for "
                               f"'{init}' failed.")
        self._lock = threading.Lock()
        LOG.debug("Capture object: %r", self._capture)
        LOG.info("Frame size: %d x %d",
                 self._capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                 self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __del__(self) -> None:
        """Destructor for this :py:class:`VideoReader`.
        The underlying :py:class:`cv2.VideoCapture` object will be
        released and deleted.
        """
        if self._capture is not None:
            LOG.info("Releasing OpenCV capture")
            self._capture.release()
            del self._capture
            self._capture = None
        # super().__del__()

    #
    # Iterator
    #

    def __next__(self) -> np.ndarray:
        """Get the next frame from the OpenCV Video Capture.
        """
        if not self.prepared:
            raise RuntimeError("VideoHelper object was not prepared.")
        ok, frame = self._capture.read()
        # ok: signals success of the operation
        # frame: is the image (in BGR!)
        if not ok:
            raise RuntimeError("Error reading frame fom video.")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #
    # FIXME[question]: when is this needed?
    #
    

    def prepared(self) -> bool:
        return self._capture is not None

    def prepare(self) -> None:
        self._capture = cv2.VideoCapture(self._video_file)
        if not self._capture.isOpened():
            raise RuntimeError("Video capture for '{self._video_file}' "
                               "could not be opened.")
        print(f"Prepared video '{self._video_file}' with {len(self)} frames.")

    def unprepare(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class VideoFileReader(VideoReader, video.FileReader):

    def __init__(self, filename: str, **kwargs) -> None:
        super().__init__(init=filename, filename=filename, **kwargs)
        LOG.info("Movie file: '%s'", filename)
        LOG.info("Frame count: %d",
                 self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        LOG.info("Frames per second: %f",
                 self._capture.get(cv2.CAP_PROP_FPS))
        seconds = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) //
                      self._capture.get(cv2.CAP_PROP_FPS))
        LOG.info("Duration: %02d:%02d:%02d",
                 seconds // 3600, (seconds // 60) % 60, seconds % 60)

    def __len__(self):
        """The number of frames in this video.
        """
        if not self.prepared:
            raise RuntimeError("VideoFileReader object was not prepared.")
        # FIXME[problem]: sometimes CAP_PROP_FRAME_COUNT does only return 0
        # -> describe this in more detail - find examples and analyze!
        return int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __getitem__(self, frame: int = None) -> np.ndarray:
        """Get the frame for a given frame number.

        Note: after getting a frame, the current frame number (obtained
        by :py:meth:`frame`) will be frame+1, that is the number of
        the next frame to read.

        Arguments
        ---------
        frame: int
            The number of the frame to be read. If no frame is given,
            the next frame available will be read from the capture
            object.
        """
        with self._lock:
            if frame is not None and frame != self.frame:
                LOG.debug("OpencvVideoBackend: actively setting frame "
                          "from %d to %d", self.frame, frame)
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            # note: read() will advance the frame position to the next
            # frame to be read.
            ret, image = self._capture.read()
            if not ret:
                raise RuntimeError("Reading an image from video capture "
                                   f"at frame {frame} failed!")
        return image[:, :, ::-1]  # convert OpenCV BGR representation to RGB

    def get_frame_at(self, time: Union[int, float, str]):
        """Get the frame at the given timepoint.

        Arguments
        ---------
        time:
            The time in any format understood by
            :py:meth:`time_in_seconds()`.
        """
        time_in_seconds = self.time_in_seconds(time)
        with self._lock:
            self._capture.set(cv2.CAP_PROP_POS_MSEC, time_in_seconds * 1000)
            ret, frame = self._capture.read()
            if not ret:
                raise RuntimeError("Reading an image from video capture "
                                   f"at {time_in_seconds:.4f}s "
                                   f"({time}) failed!")
        return frame[:, :, ::-1]  # convert OpenCV BGR representation to RGB

    @property
    def frame(self) -> int:
        """The number of the current frame.

        Result
        ------
        frame: int
            The index of the current frame. If currently no frame is
            selected, this will be -1.
        """
        # Note that cv2.CAP_PROP_POS_FRAMES is one more than the last
        # frame read from this VideoBackend. So if we want to provide
        # the frame number of the last frame read, we have to return
        # this number minus 1.
        frame = self._capture.get(cv2.CAP_PROP_POS_FRAMES)
        return int(frame) - 1

    @property
    def time(self) -> float:
        """Current position in seconds.
        """
        milliseconds = self._capture.get(cv2.CAP_PROP_POS_MSEC)
        return milliseconds * 1000.

    @property
    def frames_per_second(self) -> float:
        """Frames per second in this video.
        """
        return self._capture.get(cv2.CAP_PROP_FPS)

    def frame_at(self, time: Union[int, float, str]) -> int:
        """Get the number of the frame to be displayed at the given time.

        time:
            The time in any format understood by
            :py:meth:`time_in_seconds()`.
        """
        return int(self.frames_per_second * self.time_in_seconds(time))


class VideoFileWriter(video.Writer):

    def prepare(self):
        # Open video writer
        output_path = self._video_file  # 'fireworks_stylized.webm'
        fourcc = cv2.VideoWriter_fourcc(*'vp80')
        self._output = cv2.VideoWriter(output_path, fourcc,
                                       self._fps, self._size)

    def unprepare(self):
        self._output.release()

    @staticmethod
    def copy(video: video.Reader, transform=None, progress=None):
        # Set parameters of results
        with VideoFileWriter(fps=30.0, size=(512, 288)) as writer:

            # Go through frames and add them to video
            if progress is not None:
                video = progress(video)
            for frame in video:
                if transform is not None:
                    frame = transform(frame)
                writer(frame)


class Webcam(video.Webcam, VideoReader):
    """A :py:class:`WebcamBackend` realized by an OpenCV
    :py:class:`VideoCapture` object.


    Attributes
    ----------
    _capture: cv2.VideoCapture
        A capture object
    """

    _devices: Tuple = None
    
    @classmethod
    def devices(cls, update: bool = False) -> List[int]:
        """Obtain a list of available devices.
        """
        if cls._devices is not None and not update:
            return list(cls._devices)

        max_device = 10
        devices = []
        for index in range(max_device):
            try:
                print("A")
                cap = cv2.VideoCapture(index)
                print("B", cap)
                if not cap.read()[0]:
                    break
                print("C", cap)
                devices.append(index)
            finally:
                cap.release()
        cls._devices = tuple(devices)
        return devices

    def __init__(self, device: int = 0, **kwargs) -> None:
        """Constructor for this :py:class:`WebcamBackend`.
        The underlying :py:class:`cv2.VideoCapture` object will be
        created.
        """
        super().__init__(init=device, device=device, **kwargs)
        LOG.info("Camera device: %d", device)
        LOG.info("Frames per second: %f",
                 self._capture.get(cv2.CAP_PROP_FPS))
