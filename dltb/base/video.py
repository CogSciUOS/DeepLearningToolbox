# standard imports
from typing import Union, List, Iterator, Any
from abc import abstractmethod
from pathlib import Path
import os
import sys
import logging
import datetime
import threading

# third party imports
import numpy as np

# toolbox imports
from .image import Image, Imagelike, ImageDisplay, ImageOperator
from .image import Format as ImageFormat, Size as ImageSize
from .prepare import Preparable
from .. import thirdparty
from ..util.time import time_str

# logging
LOG = logging.getLogger(__name__)


class Video:
    pass


Videolike = Union[Video]


class Reader:
    """An abstract interface to read videos. A :py:class:`Reader`
    allows to read a video source frame by frame.
    """

    def __new__(cls, filename: str = None, device: int = None,
                module: Union[str, List[str]] = None) -> 'Reader':
        """Create a new Reader.
        """

        video_class = None
        if cls.__name__ in ('VideoReader', 'Webcam'):
            classname = cls.__name__
        elif cls.__name__ in ('RandomReader', 'Thumbcinema'):
            video_class = cls
        elif filename is not None:
            classname = 'VideoReader'
        elif device is not None:
            classname = 'Webcam'
        else:
            raise ValueError("You have to specify either filename or device "
                             "to create a video Reader object.")

        if video_class is None:
            video_class = thirdparty.import_class(classname, module=module)
        return super(Reader, video_class).__new__(video_class)

    def __init__(self, module: Union[str, List[str]] = None, **kwargs) -> None:
        super().__init__(**kwargs)

    #
    # Iterable
    #

    def __iter__(self) -> Iterator[np.ndarray]:
        """Implementation of the :py:class:`Iterable` interface.
        """
        return self

    #
    # Iterator
    #

    def __next__(self) -> np.ndarray:
        """Implementation of the :py:class:`Iterator` interface.
        """
        raise NotImplementedError(f"{self.__class__.__name__} claims to "
                                  "be a Reader, but does not implement "
                                  "the __next__ method.")
    #
    # Context manager
    #

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

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

    # FIXME[todo]: documentation!

    @property
    def frames_per_second(self) -> float:
        """
        """
        return self._frames_per_second()

    @abstractmethod
    def _frames_per_second(self) -> float:
        """
        """
        # to be overwritten by subclasses

    @property
    @abstractmethod
    def frame(self) -> int:
        """
        """

    @abstractmethod
    def get_frame_at(self, time: Union[int, float, str]):
        """
        """

    @abstractmethod
    def frame_at(self, time: Union[int, float, str]) -> int:
        """
        """


class RandomReader(Reader):

    def __init__(self, fps: float = 25.0, size=(100, 100), **kwargs) -> None:
        super.__init__(**kwargs)
        self._frames_per_second = lambda _: fps
        self._size = size

        # The following code will use the new random generator of numpy
        # 1.17 if available and otherwise fall back to the legacy version.
        rng = np.random.default_rng()
        try:
            self._random_integers = rng.integers
        except AttributeError:
            self._random_integers = rng.randint

    def __next__(self) -> np.ndarray:
        """Implementation of the :py:class:`Iterator` interface.
        """
        return self._random_integers(0, 256, size=self._size[::-1] + (3,),
                                     dtype=np.uint8)


class VideoReader(Reader):
    """A :py:class:`VideoReader` is a :py:class:`Reader` that reads
    a video. This class adds length and index access.
    """

    def __new__(cls, filename: str = None,
                module: Union[str, List[str]] = None) -> 'Reader':
        """Create a new VideoReader.
        """
        if filename is None:
            raise ValueError("You have to specify either filename "
                             "to create a VideoReader object.")
        return super().__new__(cls, filename=filename, module=module)

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

    @property
    def frame(self) -> int:
        """The index of the frame the next read will yield.
        """
        return self._get_frame()

    @abstractmethod
    def _get_frame(self) -> int:
        """Get the index of the frame the next read will yield.
        """


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


class URLReader(VideoReader):  # FIXME[todo]: implementation
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


class Writer(Preparable):
    """A video :py:class:`Writer` is a "sink" for or "consumer" of video
    data, that is a sequence of frames (images).  A video can be
    written completely, in chunks or framewise.  There is also the
    option to asynchronously run a video :py:class:`Writer` either by
    regularly querying or by observing an :py:class:`ImageObservable`.

    A video :py:class:`Writer` can be associated with different speed
    parameters, all expressed as frame rate in frames per second
    (fps).  The kind of supported parameters depends on the type of
    the :py:class:`Writer`.  One may control the operational speed of
    the :py:class:`Writer`, meaning how many frames per second should
    be written.  If frames are written to the :py:class:`Writer` at a
    higher rate, some frames may be skipped and if the
    :py:class:`Writer` is actively querying an
    :py:class:`ImageObservable`, operational speed determines the
    maximal query frequency.

    A video :py:class:`Writer` storing videos to file (or some other
    form of storage) may save a frame rate as part of the metadata
    and will indicate the desired speed for playback. This can be set
    independent of the operational speed of the :py:class:`Writer`.

    Frames may be provided to a video :py:class:`Writer` in form of
    :py:class:`Imagelike` objects.

    The video :py:class:`Writer` is an abstract base class that has to
    be subclassed to provide actual funcitonality.  The central method
    to be overwritten is :py:meth:`_write_frame` which will get a
    frame and should perform the write operation. If the
    :py:class:`Writer` needs specific resources, those can be required
    by overwriting the :py:meth:`_open_writer` and
    :py:meth:`_close_writer` methods.

    """
    # FIXME[todo]: most of what is described in the docstring is not
    # yet implemented.


    _frame_format: ImageFormat

    def __new__(cls, filename: str = None,
                module: Union[str, List[str]] = None, **kwargs) -> 'Writer':
        """Create a new video :py:class:`Writer`. This constructor
        will decide on what class to instantiate based on the arguments
        provided.  It will use the `thirdparty` module to select an
        actual implementation.
        """
        video_class = None
        if cls.__name__ in ('NullWriter', 'Display'):
            video_class = cls
        elif filename is not None:
            classname = 'VideoWriter'
        else:
            raise ValueError("You have to specify a filename to "
                             "to create a video Writer object.")

        if video_class is None:
            video_class = thirdparty.import_class(classname, module=module)
        return super(Writer, video_class).__new__(video_class)

    def __init__(self, fps: float, size: ImageSize,
                 frame_format: ImageFormat, **kwargs) -> None:
        super().__init__(**kwargs)
        self._fps = fps
        self._size = size
        self._frame_format = frame_format

    def __del__(self) -> None:
        if self.opened:
            self.close()

    def __call__(self, video: VideoReader, **kwargs) -> None:
        self._assert_opened()
        for frame in video:
            self.write_frame(self._prepare_frame(frame), **kwargs)

    def __bool__(self) -> bool:
        return self._is_opened()

    def __iadd__(self, frame: Imagelike) -> 'VideoWriter':
        self._assert_opened()
        self.write_frame(frame)
        return self

    #
    # context manager
    #

    def __enter__(self) -> 'Writer':
        self._open()
        return self

    def __exit__(self, _exception_type, _exception_value, _traceback) -> None:
        self._close()

    #
    # public interface
    #

    @property
    def frame_format(self) -> ImageFormat:
        """The :py:class:`ImageFormat` used by this video :py:class:`Writer`.
        Frames written to this :py:class:`Writer` are assumed to be in
        that format, except if explicitly stated otherwise (for
        example by providing explicit format information when calling
        :py:class:`write_frame` or when writing an :py:class:`Image`
        of another format).

        """
        return self._frame_format

    def write_frame(self, frame: Imagelike, **kwargs) -> None:
        """Write a frame to this video :py:class:`Writer`.
        """
        self._assert_opened()
        self._write_frame(self._prepare_frame(frame, **kwargs))

    # FIXME[todo]: remove - use __bool__ instead
    # FIXME[todo]: and pepare/unprepare instead of open/close
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

    def copy(self, video, transform=None, progress=None):
        """Copy a given video to this video :py:class:`Writer`.
        """
        if progress is not None:
            video = progress(video)
        for index, frame in enumerate(video):
            if transform is not None:
                frame = transform(frame)
            self._image_writer(self.filename_for_frame(index), frame)

    #
    # private API (to be overwritten by subclasses)
    #

    def _write_frame(self, frame: np.ndarray) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} claims to "
                                  "be a VideoWriter, but provides no "
                                  "'_write_frame' method.")

    def _open(self) -> None:
        """Make sure the Writer is open, that is, ready to write.
        If sufficient arguments are given to the constructor, this
        will be called automatically, otherwise it has to be called
        explicitly before the first read.
        """

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

    def _prepare_frame(self, frame: Imagelike) -> Any:
        """Prepare a frame to be written. This includes operations
        like resizing it to fit into the video.
        """
        # FIXME[todo]: check frame (size, colors, etc)
        return Image.as_array(frame)  # may be overwrittern by subclasses

    #
    # private auxiliary methods
    #

    def _assert_opened(self) -> None:
        if not self.opened:
            raise RuntimeError("VideoWriter is not open.")


class VideoWriter(Writer):
    pass  # FIXME[hack]


class NullWriter(Writer):
    """The :py:class:`NullWriter` will discard all frames. It is
    meant a convenience class to be used when a Writer is required,
    but the written frames are not used.
    """

    def __init__(self, fps: float = None, size = None, **kwargs) -> None:
        super().__init__(fps=fps, size=size, **kwargs)

    def write_frame(self, frame: np.ndarray, **kwargs) -> None:
        """
        """
        pass


class Display(Writer):
    """
    """
    # FIXME[todo]: introduce a show method (show the video)
    # either in blocking or non-blocking mode

    def __init__(self, fps: float = None, size=None,
                 display: ImageDisplay = None, **kwargs) -> None:
        super().__init__(fps=fps, size=size, **kwargs)
        self._display = ImageDisplay() if display is None else display

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type == ValueError and self._display.closed:
            return True  # ignore the exception
        return super().__exit__(exc_type, exc_value, exc_traceback)

    def _open(self) -> None:
        """Open the video :py:class:`Writer` for writing.  In the case
        of a :py:class:`Display` this means preparing the display
        component for showing images.
        """
        # Do not use a display event loop: we will update the image
        # regularly, and if no event loop is running events will
        # be processed on each invocation of display.show(). This
        # should suffice for a smooth user interface except for
        # videos with a very low frame rate.
        self._display.blocking = None  # do not start a display event loop
        self._display._view.show()  # FIXME[hack]: design a better Display API

    def _write_frame(self, frame: np.ndarray) -> None:
        """Write a frame with this :py:class:`Writer`. In the case
        of a :py:class:`Display` this means to show the frame in the
        display component.

        Arguments
        ---------
        frame:
            The frame to be displayed as an array. The frame is
            expected to be in the standard format of this
            :py:class:`Writer`, which defaults to `np.uint8` values,
            color images being in RGB color space.
        """
        if self._display.closed:
            raise ValueError("Writing to closed display.")
        self._display.show(frame)

    def close(self):
        """Close the video :py:class:`Writer`.  In the case
        of a :py:class:`Display` this means to close the display
        component showing the images.
        """
        self._display.close()


class FileWriter(VideoWriter, FileBase):
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


class VideoDirectory:
    """The :py:class:`VideoDirectory` provides base functionality for
    storing videos as a collection of individual images stored in
    a directory.
    """

    _frames_directory: Path

    def __init__(self, directory: Union[Path, str], **kwargs) -> None:
        super().__init__(**kwargs)
        self._frames_directory = Path(directory)

    def filename_for_frame(self, index: int) -> Path:
        """Obtain the filename for a specific frame in this
        :py:class:`VideoDirectory`

        Arguments
        ---------
        index:
            The index of the frame.

        Result
        ------
            The a
        """
        return self._frames_directory / f'frame_{index}.jpg'


class DirectoryReader(Reader, VideoDirectory):
    """A :py:class:`VideoReader` for videos that are stored as individual
    images in a directory.
    """

    def __len__(self) -> int:
        if self._len is None:
            if not self._frames_directory.isdir():
                raise RuntimeError("Frames directory "
                                   f"'{self._frames_directory}' "
                                   "does not exist.")
            else:
                self._len = len(os.listdir(self._frames_directory))
        return self._len

    def __getitem__(self, index: int) -> np.ndarray:
        if not self._frames_directory.isdir():
            raise RuntimeError(f"Frame directory '{self._frames_directory}' "
                               "does not exist.")
        if not 0 <= index < len(self):
            raise IndexError(f"Invalid frame index {index}"
                             f"is not in [0, {len(self)}]")
        return self._loader(self.filename_for_frame(index))


class DirectoryWriter(VideoWriter, VideoDirectory):
    """A :py:class:`DirectoryWriter` stores a video by writing individual
    frames as images into a directory.

    """

    def __init__(self, image_writer, **kwargs) -> None:
        super().__init__(**kwargs)
        self._image_writer = image_writer
        self._frame_index = None

    def _prepare(self) -> None:
        """Prepare this :py:class:`DirectoryWriter` for writing.
        Thi will ensure that the directory exists.
        """
        super()._prepare()
        os.makedirs(self._directory, exist_ok=True)
        self._frame_index = 0

    def _unprepare(self) -> None:
        """Release resources acquired by this video :py:class:`Writer`
        and reset it into an unprepared state.
        """
        self._frame_index = None
        super()._unprepare()

    def _prepared(self) -> bool:
        """Check whether this :py:class:`DirectoryWriter` is prepared
        for writing frames.
        """
        return (self._frame_index is not None) and super()._prepared()

    def _write_frame(self, frame: np.ndarray) -> None:
        """Write an image to this :py:class:`DirectoryWriter`. This
        writes an image file into the directory and increases the
        current index by one.

        Arguments
        ---------
        frame:
            The frame to be written to the :py:class:`Writer`. This
            is assumed to be given in the correct format (dtype, colorspace,
            size, etc.) as expected by the :py:class:`ImageWriter`
            employed by this :py:class:`DirectoryWriter`.
        """
        self._image_writer(self.filename_for_frame(self._frame_index), frame)
        self._frame_index += 1
