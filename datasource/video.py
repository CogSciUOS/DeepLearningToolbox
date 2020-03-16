import importlib.util

from datasource import Indexed, Imagesource, Loop, InputData


class DataVideo(Imagesource, Indexed, Loop):
    """A data source fetching frames from a video.

    Attributes
    ----------
    _backend:
        The VideoBackend use for accessing the video file.
    _filename:
        The filename frome which the video is read.
    _frame:
        The currently fetched frame.
        If the frame changes, Observers of this :py:class:`DataVideo`
        will receive a `data_changed` notification.
    """

    @staticmethod
    def check_availability() -> bool:
        """Check if this Datasource is available.

        Returns
        -------
        True if the OpenCV library is available, False otherwise.
        """
        return importlib.util.find_spec('cv2')

    def __init__(self, filename: str, **kwargs):
        """Create a new DataWebcam
        """
        super().__init__(**kwargs)
        self._filename = filename
        self._backend = None
        self._frame = None
        self._description = "Frames from the video \"{}\"".format(filename)

    def __str__(self):
        return 'Movie'

    @property
    def prepared(self) -> bool:
        """Report if this Datasource is prepared for use.
        A Datasource has to be prepared before it can be used.
        """
        return self._backend is not None

    def _prepare_data(self):
        """Prepare this Datasource for use.
        """
        self._backend = OpencvVideoBackend(self._filename)
        self._loop_interval = 1. / self._backend.frames_per_second

    def _unprepare_data(self):
        """Unprepare this Datasource. This will free resources but
        the webcam can no longer be used.  Call :py:meth:`prepare`
        to prepare the webcam for another use.
        """
        del self._backend
        self._backend = None
        self._frame = None

    @property
    def fetched(self):
        """Check if data was fetched and is available now from this
        :py:class:`Datasource`.
        """
        return self._frame is not None

    def _fetch(self, frame: int=None, time: float=None, **kwargs):
        """Fetch a video frame from this :py:class:`DataVideo`.

        Arguments
        ---------
        frame: int            
            The frame number of the frame to fetch from the video.
        time: int
            The temporal position (point in time) of the frame
            to fetch in seconds (fractions may be given).
        """
        if frame is not None:
            self._fetch_frame(frame)
        elif time is not None:
            self._fetch_frame_at(time)
        else:
            super()._fetch(**kwargs)

    def _fetch_frame(self, frame: int=None):
        """Fetch a video frame from this :py:class:`DataVideo`.

        Arguments
        ---------
        frame: int
            The frame number of the frame to fetch from the video.
            If no frame is given, the next frame of the video will
            be fetched (advancing the frame number by 1).
        """
        if frame is not None and frame == self.frame:
            return  # we already have the desired frame
        self._frame = self._backend.get_frame(frame)

    def _fetch_frame_at(self, time: float):
        """Fetch a video frame at a given timepoint from this
        :py:class:`DataVideo`.

        Arguments
        ---------
        time: int
            The temporal position (point in time) of the frame
            to fetch in seconds (fractions may be given).
        """
        self._frame = self._backend.get_frame_at(time)
        
    def _get_data(self):
        return self._frame

    #
    # Public Video API
    #

    def fetch_frame(self, frame: int, **kwargs) -> None:
        self.fetch(frame=frame, **kwargs)

    @property
    def frame(self) -> int:
        return self._backend.frame - 1

    @property
    def time(self) -> float:
        return self._backend.time

    @property
    def frames_per_second(self) -> float:
        return self._backend.frames_per_second

    #
    # Implementation of the 'Indexed' API
    #
        
    def _get_index(self) -> int:
        return self.frame

    def __len__(self):
        return self._backend.frame_count if self._backend else 0

    def _fetch_index(self, index: int):
        """Implementation of the :py:class:`Indexed` datasource interface.
        The index will be interpreted as frame number.

        # FIXME[todo]: we may allow for other index formats: float (seconds),
        # or time (str)
        """
        self._fetch_frame(frame=index)

class VideoBackend:
    
    @property
    def frame_count(self):
        """The number of Frames in this video.
        """
        raise NotImplementedError(f"{self.__class__.__name__} claims to "
                                  "be a VideoBackend, but provides no "
                                  "'frame_count' property.")

    def get_frame(self, frame: int):
        raise NotImplementedError(f"{self.__class__.__name__} claims to "
                                  "be a VideoBackend, but provides no "
                                  "'get_frame' method.")
        

import cv2
import threading

class OpencvVideoBackend:
    """A :py:class:`VideoBackend` realized by an OpenCV
    :py:class:`VideoCapture` object.

    
    Attributes
    ----------
    _capture: cv2.VideoCapture
        A capture object
    _lock: threading.Lock
        A lock to avoid race conditions due to simultanous access to
        the video capture object.
    """

    def __init__(self, filename: str):
        """Constructor for this :py:class:`VideoBackend`.
        The underlying :py:class:`cv2.VideoCapture` object will be
        created.
        """
        self._capture = cv2.VideoCapture(filename)
        if not self._capture:
            raise RuntimeError("Creating video capture object for "
                               f"'{filename}' failed.")
        if not self._capture.isOpened():
            self.__del__()
            raise RuntimeError("Opening the video capture object for "
                               f"'{filename}' failed.")
        self._lock = threading.Lock()
        print("Capture object:", type(self._capture))
        print(f"Movie file: '{filename}'")
        print("Frame count:", self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Frame width:", self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("Frame height:", self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Frame fps:", self._capture.get(cv2.CAP_PROP_FPS))
        seconds = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) //
                      self._capture.get(cv2.CAP_PROP_FPS))
        print("Duration: "
              f"{seconds//3600:02d}:{(seconds//60)%60:02d}:{seconds%60:02d}")
    
    def __del__(self):
        """Destructor for this :py:class:`VideoBackend`.
        The underlying :py:class:`cv2.VideoCapture` object will be
        released and deleted.
        """
        self._capture.release()
        del self._capture
        self._capture = None

    @property
    def frame_count(self):
        # FIXME[problem]: sometimes CAP_PROP_FRAME_COUNT does only return 0
        # -> describe this in more detail - find examples and analyze!
        return int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def frame(self):
        """The number of the next frame to be read.

        Note that this is one more than the last frame read from this
        VideoBackend. So if you want to display the frame number of
        the last frame read from this :py:class:`VideoBackend`, you
        have to take backend.frame - 1.
        """
        frame = self._capture.get(cv2.CAP_PROP_POS_FRAMES)
        return int(frame)

    @property
    def time(self) -> float:
        """Current position in seconds.
        """
        milliseconds = self._capture.get(cv2.CAP_PROP_POS_MSEC)
        return milliseconds * 1000.

    @property
    def frames_per_second(self) -> float:
        """Current position in seconds.
        """
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        return fps

    def get_frame(self, frame: int=None):
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
                print(f"OpencvVideoBackend: actively setting frame from {self.frame} to {frame}")
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            # note: read() will advance the frame position to the next
            # frame to be read.
            ret, image = self._capture.read()
            if not ret:
                raise RuntimeError("Reading an image from video capture "
                                   f"at frame {frame} failed!")
        return image[:,:,::-1]  # convert OpenCV BGR representation to RGB

    def get_frame_at(self, time: float):
        """Get the frame at the given timepoint.

        Arguments
        ---------
        time: float
            The timepoint in seconds in the video to read the frame from.
            Fractions are possible.
        """
        with self._lock:
            self._capture.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
            ret, frame = self._capture.read()
            if not ret:
                raise RuntimeError("Reading an image from video capture "
                                   f"at {time:.4f}s failed!")
        return frame[:,:,::-1]  # convert OpenCV BGR representation to RGB
