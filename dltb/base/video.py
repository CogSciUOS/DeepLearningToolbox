# standard imports
from typing import Union, List, Iterator
import os
import sys
import logging
import datetime
import threading

# third party imports
import numpy as np

# toolbox imports
from .image import ImageOperator
from .. import thirdparty
from ..util.time import time_str

# logging
LOG = logging.getLogger(__name__)


class Reader:
    """An abstract interface to read videos. A :py:class:`Reader`
    allows to read a video source frame by frame.
    """

    def __new__(cls, filename: str = None, device: int = None,
                module: Union[str, List[str]] = None) -> 'Reader':
        """Create a new Reader.
        """

        if filename is not None:
            classname = 'VideoReader'
        elif device is not None:
            classname = 'Webcam'
        else:
            raise ValueError("You have to specify either filename or device "
                             "to create a video Reader object.")

        video_class = thirdparty.import_class(classname, module=module)
        return super(Reader, video_class).__new__(video_class)

    @staticmethod
    def time_in_seconds(time: Union[float, str]) -> float:
        """Convert a time argument into seconds.

        time:
            The time argument. Either a number (int or float) specifying
            the time in seconds or a string of the format "H:M:S".
        """
        if isinstance(time, str):
            hours, minutes, seconds = map(int, time.split(':'))
            duration = datetime.timedelta(hours=hours, minutes=minutes,
                                          seconds=seconds)
            time = duration.total_seconds()
        elif isinstance(time, str):
            time = float(time)
        return time

    @property
    def time(self) -> float:
        """Current position in seconds.
        """
        # FIXME[hack]: just an approximation - will not work for
        # variable frame rates ...
        return self.frame / self.frames_per_second

    @property
    def time_str(self) -> str:
        """A string representing the current position in time
        of this :py:class:`VideoBackend`.
        """
        return time_str(self.time)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Implementation of the :py:class:`Iterable` interface.
        """
        return self

    def __next__(self) -> np.ndarray:
        """Implementation of the :py:class:`Iterator` interface.
        """
        raise NotImplementedError(f"{self.__class__.__name__} claims to "
                                  "be a Reader, but does not implement "
                                  "the __next__ method.")


class VideoReader(Reader):
    """A :py:class:`VideoReader` is a :py:class:`Reader` that reads
    a video. This class adds length and index access.
    """

    def __len__(self) -> int:
        """Implementation of the :py:class:`Sized` interface.
        The length of the video in frames.
        """
        raise NotImplementedError(f"{self.__class__.__name__} claims to "
                                  "be a VideoReader, but does not implement "
                                  "the __len__ method.")

    def __getitem__(self, frame: int) -> np.ndarray:
        """Implementation of the :py:class:`Mapping` interface.
        Get the frame at the given position from this video.
        """
        raise NotImplementedError(f"{self.__class__.__name__} claims to "
                                  "be a VideoReader, but provides no "
                                  "'__getitem__' method.")


class FileBase:
    """
    """

    def __init__(self, filename: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._filename = filename

    @property
    def filename(self) -> str:
        return self._filename


class FileReader(VideoReader, FileBase):
    """An abstract interface to read videos. A :py:class:`FileReader`
    extends the base :py:class:`Reader` by adding the idea of length:
    a video file has an beginning and an end.  Within this interval,
    the reader points to a current position from which a frame is
    read.

    """


class URLReader(VideoReader):
    """Reading videos from a URL.

    Attributes
    ----------

    _url: str
        The URL for the video.
    """

    def __init__(self, url: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._url = url

    @property
    def url(self) -> str:
        return self._url


class Webcam(Reader):
    """An abstract webcam backend.

    Attributes
    ----------
    _device: int
        Device number of the webcam to use.

    _lock: threading.Lock
        A lock to avoid race conditions due to simultanous access to
        the webcam.
    """

    def __init__(self, device: int, **kwargs) -> None:
        LOG.info("Acqiring Webcam (%d) for %s.", device, type(self))
        super().__init__(**kwargs)
        self._device = device
        self._lock = threading.Lock()

    def __del__(self) -> None:
        if self._lock is not None:
            del self._lock
            self._lock = None

    def get_frame(self, clear_buffer: bool = True) -> np.ndarray:
        """The default (and only) mode of getting data from the webcam
        is reading the next frame.

        Arguments
        ---------
        clear_buffer: bool
            On some systems, the video driver buffers some frames, so that
            just reading the frame may result in outdated data. Clearing
            the buffer may avoid this problem, but may delay the whole
            process.
        """
        LOG.debug("%s.get_data(clear_buffer=%s)", type(self), clear_buffer)
        with self._lock:
            if clear_buffer:
                self._clear_buffer()
            return next(self)

    def _clear_buffer(self) -> None:
        """Clear the webcam buffer.
        """
        # Hack: under linux, the av-based linux capture code is using
        # an internal fifo buffer (5 frames, iirc),  you cannot clean (or
        # say, flush) it.
        #
        # Hence we will skip some frames to really get the current image.
        if sys.platform == 'linux':
            ignore = 4
            for _ in range(ignore):
                next(self)

class Writer:
    """
    """

    def __init__(self, fps: float, size) -> None:
        self._fps = fps
        self._size = size

    def __del__(self) -> None:
        if self.opened:
            self.close()

    def __call__(self, video: VideoReader, **kwargs) -> None:
        self._assert_opened()
        for frame in video:
            self.write_frame(self._prepare_frame(frame), **kwargs)

    def __iadd__(self, frame: np.ndarray) -> 'VideoWriter':
        self._assert_opened()
        self.write_frame(frame)
        return self

    def _assert_opened(self) -> None:
        if not self.opened:
            raise RuntimeError("VideoWriter is not open.")

    def write_frame(self, frame: np.ndarray, **kwargs) -> None:
        """Check if this :py:class:`VideoWriter` is still open, meaning
        that it accepts more frames to be written.
        """
        self._assert_opened()
        self._write_frame(self._prepare_frame(frame, **kwargs))

    @property
    def opened(self) -> bool:
        """Check if this :py:class:`VideoWriter` is still open, meaning
        that it accepts more frames to be written.
        """
        return self._is_opened()

    def close(self) -> None:
        """Close this :py:class:`VideoWriter`. This will free resources
        occupied by this writer. After closing the writer, no more
        frames should be written.

        The actual implementation depends on the nature of the writer.
        Depending on the implementation, the :py:class:`VideoWriter`
        may still report :py:meth:`opened` to be True.

        """
        self._close()

    #
    # private API (to be overwritten by subclasses)
    #

    def _write_frame(self, frame: np.ndarray) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} claims to "
                                  "be a VideoWriter, but provides no "
                                  "'_write_frame' method.")

    def _is_opened(self) -> bool:
        """Check if this :py:class:`VideoWriter` is still open, meaning
        that it accepts more frames to be written.
        """
        return True  # may be overwritten by subclasses

    def _close(self) -> None:
        """Actual implementation of the closing operation.
        This should release all system resources acquired by this
        :py:class:`VideoWriter`, like file handles, etc.
        """
        pass  # may be overwritten by subclasses

    def _prepare_frame(self, frame: np.ndarray) -> None:
        """Prepare a frame to be written. This includes operations
        like resizing it to fit into the video.
        """
        # FIXME[todo]: check frame (size, colors, etc)
        return frame  # may be overwrittern by subclasses


class VideoWriter(Writer):
    pass  # FIXME[hack]


class VideoUtils:
    """A helper class for videos, that allows to
    (1) download videos from URL and
    (2) transform video into sequence of images.
    """

    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)

    def download(self, url: str, filename: str) -> None:
        pass

    def download_youtube(self, url: str, filename: str) -> None:
        pass


class VideoOperator:
    """A :py:class:`VideoProcessor` processes a video frame by frame.
    It thereby transforms an input video into an output video.
    """

    def __init__(self, operator: ImageOperator, **kwargs) -> None:
        super().__init__(**kwargs)
        self._operator = operator

    def __call__(self, source: VideoReader) -> VideoReader:
        for frame in source:
            yield self._operator(frame)

    def transform(self, source: VideoReader, target: VideoWriter):
        for frame in self(source):
            target += frame


class DirectoryBase:

    def __init__(self, directory: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._directory = directory

    def filename_for_frame(self, index: int) -> str:
        return os.path.join(self._frames_directory, f'frame_{index}.jpg')


class DirectoryReader(Reader, DirectoryBase):
    """A :py:class:`VideoReader` for videos that are stored as individual
    images in a directory.
    """

    def __len__(self) -> int:
        if self._len is None:
            if not os.path.isdir(self._frames_directory):
                raise RuntimeError("Frames directory "
                                   f"'{self._frames_directory}' "
                                   "does not exist.")
            else:
                self._len = len(os.listdir(self._frames_directory))
        return self._len

    def __getitem__(self, index: int) -> np.ndarray:
        if not os.path.isdir(self._frames_directory):
            raise RuntimeError(f"Frame directory '{self._frames_directory}' "
                               "does not exist.")
        if not 0 <= index < len(self):
            raise IndexError(f"Invalid frame index {index}"
                             f"is not in [0, {len(self)}]")
        return self._loader(self.filename_for_frame(index))


class DirectoryWriter(VideoWriter, DirectoryBase):

    def __init__(self, image_writer, **kwargs) -> None:
        super().__init__(**kwargs)
        self._image_writer = image_writer

    def prepare(self) -> None:
        super().prepare()
        os.makedirs(self._directory, exist_ok=True)

    def copy(self, video, transform=None, progress=None):
        if progress is not None:
            video = progress(video)
        for index, frame in enumerate(video):
            if transform is not None:
                frame = transform(frame)
            self._image_writer(self.filename_for_frame(index), frame)
