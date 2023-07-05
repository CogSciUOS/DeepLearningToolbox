"""Definition of abstract video programming interfaces. There are
several ideas covered by these interfaces.

* A video as ImageSource: a datasource that provides images.
  As such, it can have different properties:
  - Video: it can be a sequence (providing index access). Individual
  - VideoReader: it can be iterable. It has an internal state,
      indicating the current frame. It may or may not be possible
      to rewind the reader. From a rewindable Reader a Video can be
      be constructed and from a Video, a (rewindable) VideoReader
      can be constructed.

* A video as a place where images (frames) can be deposited (VideoWriter).
  - VideoWriter:
    currently only sequential (incremental) writers are supported.

from dltb.util.logging import debug_module
debug_module('dltb.base.gui')
debug_module('dltb.base.image')
debug_module('dltb.base.video')

Example 1
---------
from dltb.base.video import RandomReader, VideoDisplay
reader = RandomReader(raw_mode=True)
for frame in reader:
    print(frame.shape, frame.mean())

Example 2
---------

from dltb.base.video import RandomReader, VideoDisplay
from tqdm import tqdm
display = VideoDisplay()
display(RandomReader(), progress=tqdm)

with display:
    for frame in reader:
        display += frame

FIXME[todo]: these ideas are not yet specified clearly and the implementation
is currently not consistent with this description.

"""

# standard imports
from typing import Union, Any, Tuple, Optional
from typing import Sequence, Iterator
#from abc import abstractmethod
from pathlib import Path
from time import time as now, sleep
import os
import sys
import logging
import datetime
import threading

# third party imports
import numpy as np

# toolbox imports
from .image import Image, Imagelike, ImageProperties, ImageDisplay
from .image import Format as ImageFormat, Size, Sizelike
from .image import ImageReader, ImageWriter
from .prepare import Preparable
from .implementation import Implementable
from ..util.time import time_str, Timelike, IndexTiming
from ..types import Pathlike, as_path

# logging
LOG = logging.getLogger(__name__)


class FileBase:
    """A base for classes that operate on a file.
    """

    def __init__(self, filename: Pathlike, **kwargs) -> None:
        super().__init__(**kwargs)
        self._path = as_path(filename)

    @property
    def filename(self) -> str:
        """The name of the file this `FileBase` is operating (as `str`).
        """
        return str(self._path)

    @property
    def path(self) -> Path:
        """The name of the file this `FileBase` is operating (as `Path`).
        """
        return self._path


Indexlike = Union[int, Timelike]


class VideoProperties(ImageProperties):
    """A collection of properties that a video may possess.

    Arguments
    ---------
    framerate:
        A float value specifying the number of frames per second.
    number_of_frames:
        Number of frames of the complete video. ``None`` if length
        is not known or undefined (e.g., in a video stream).
    raw_mode:
        A flag indicating if the frames are provided in raw format
        (``True``), e.g., as numpy array, or as :py:class:`Image`
        objects.
    """

    _timing: Optional[IndexTiming] = None
    raw_mode: bool

    def __init__(self, framerate: Optional[float] = None,
                 number_of_frames: Optional[int] = None,
                 raw_mode: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._number_of_frames = number_of_frames
        self.raw_mode = raw_mode
        if framerate is not None:
            self._timing = IndexTiming(samplerate=framerate)

    def _to_index(self, index: Indexlike) -> int:
        if isinstance(index, int):
            return index
        if self._timing is None:
            raise ValueError(f"Video has no timing - time '{index}' "
                             "cannot be mapped to index.")
        return self._timing.time_to_index(index)

    def _result_frame(self, frame: np.ndarray, index: int) -> Imagelike:
        """
        """
        if self.raw_mode:
            return frame
        image = Image(frame)
        # add video, index ...
        image.add_attribute('index', index)
        return image

    @property
    def size(self) -> Size:
        """The size of a video frame.
        """
        return self._size

    @property
    def channels(self) -> int:
        """The number of color channels.
        """
        return self._channels

    @property
    def shape(self) -> Tuple[int]:
        """The shape of single frame in the video.
        """
        size = self.size
        return (size.height, size.width, self.channels)

    @property
    def number_of_frames(self) -> Optional[int]:
        """The total number of frames the video consists of.
        """
        return self._number_of_frames

    @property
    def framerate(self) -> Optional[float]:
        """The number of frames per second.  If a framerate is
        given, its meaning depends on the specific sublcass.
        The framerate can be `None`.
        """
        return self._timing and self._timing.samplerate

    @property
    def frames_per_second(self) -> float:
        """The number of frames per second.
        """
        return self.framerate

    @property
    def duration(self) -> Optional[float]:
        """The number of frames per second.
        """
        number_of_frames = self.number_of_frames
        if number_of_frames is None:
            return None
        framerate = self.framerate
        if framerate is None:
            return None
        return number_of_frames / framerate


class Video(Sequence[Image], VideoProperties):
    """The `Video` class represents a video as a (immutable) `Sequence` of
    images (frames).

    The frames of a `Video` can be accessed by index access:
    `video[index]` will give the frame with the given index.
    If the video has a framerate, frames can also be accessed
    as `video[time]`.

    Arguments
    ---------
    framerate:
        The number of frames per second (fps).
    """

    def __getitem__(self, indexlike: Indexlike) -> Image:
        index = self._to_index(indexlike)
        if not 0 <= index < len(self):
            raise IndexError("Index '{indexlike}' ({index}) is out of Range.")
        return self._result_frame(self._frame_at(index), index=index)

    def __len__(self) -> int:
        return self.number_of_frames

    @property
    def framerate(self) -> float:
        """The Framerate of this `Video`.

        Raises
        ------
        AttributeError:
            Accessing this property may raise an `AttributeError` if
            the Video has no framerate.
        """
        if self._timing is None:
            raise AttributeError("No framerate was assigned to the video.")
        return self._timing.samplerate

    #
    # methods to be implemented by subclasses
    #

    def _frame_at(self, index: int) -> np.ndarray:
        """Get a specific frame from this video.
        """
        # to be implemented by subclasses


Videolike = Union[Video]


class VideoFile(Video, FileBase, Implementable):
    """A :py:class:`Video` that is read from a file.

    A `Videofile` provides a sequence view on a video, that is the
    frames may be accessed by index.  Depending on the underlying
    implementation and video codec, that may not be the best
    option for video playback.  The :py:class:`VideoFileReader`
    provides an alternative, realizing an iterator access to a
    video file that may be better suited for playback.  A further
    alternative is the :py:class:`VideoFilePlayer` that can be
    used of an asynchronous playback.

    """


class VideoReader(Iterator[Image], VideoProperties, Implementable):
    """An abstract interface to read videos. A :py:class:`Reader`
    allows to read a video source frame by frame.  Reading a
    frame can be done explicitly by calling `frame = next(reader)` or
    implicitly by looping over the frames (`for frame in reader: ...`).

    A `VideoReader` provides the current frame number as the property
    `index`. Certain `VideoReaders` may allow to set this property
    to seek a specific position.  If a framerate is set for the
    `VideoReader`, the current position can also be accessed as
    :py:meth:`time` (in seconds).

    If a framerate is set, iterating over the reader can also be
    forced to occur in realtime, meaning that frames are not provided
    at a rate higher than this framerate.

    The `VideoReader` implements the context manager interface.
    """

    # FIXME[old/todo]: integrate into the Implementation logic:
    #   - when constructed as VideoReader or WebCam, use an implementation
    #     of that class
    #   - when constructing the classes RandomReader and Thumbcinema, use
    #     that specific class
    #   - when a `filename` argument is provided, use a VideoReader class
    #   - when a `device` argument is provided, use a WebCam class
    #   - otherwise raise an error

    _index: int
    _realtime: bool
    _last_read_timestamp: float

    def __init__(self, realtime: bool = False, **kwargs) -> None:
        super().__init__(self, **kwargs)
        self._index = 0
        self._realtime = realtime
        self._last_read_timestamp = now()

    #
    # Iterator
    #

    def __next__(self) -> np.ndarray:
        """Implementation of the :py:class:`Iterator` interface. Will read and
        return the next frame.
        """
        if self._realtime:
            time_since_last_read = now() - self._last_read_timestamp
            remaining_time = time_since_last_read - (1/self.framerate)
            if remaining_time > 0:
                sleep(remaining_time)
        self._last_read_timestamp = now()

        index = self.index
        frame = self._next_frame()
        if frame is None:
            raise StopIteration()
        return self._result_frame(frame, index=index)

    #
    # Context manager
    #

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    #
    # Index (current positions)
    #

    @property
    def index(self) -> int:
        """The index of the next frame to be fecthed from this
        `VideoReader`.
        """
        return self._index

    @index.setter
    def index(self, index: int) -> None:
        self._set_index(index)

    def _set_index(self, index: int) -> None:
        self._index = index  # to be overwritten by subclasses

    @property
    def time(self) -> float:
        """Current position in seconds.
        """
        return self._timing.index_to_time(self.index)

    @time.setter
    def time(self, time: Timelike) -> None:
        self.index = self._timing.time_to_index(time)

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

    #
    # to be implemented by subclasses
    #

    #@abstractmethod
    def _next_frame(self) -> np.ndarray:
        """Read the next from this :py:class:`Reader`.
        """
        return self and None  # not used, but will make pylint happy ;-)


class RandomReader(VideoReader):
    """A :py:class:`VideoReader` that provides random (noise) images.
    """

    def __init__(self, size: Sizelike = (100, 100), channels: int = 3,
                 **kwargs) -> None:
        super().__init__(size=size, channels=channels, **kwargs)
        self._size = Size(size)

        # The following code will use the new random generator of numpy
        # 1.17 if available and otherwise fall back to the legacy version.
        rng = np.random.default_rng()
        try:
            self._random_integers = rng.integers
        except AttributeError:
            self._random_integers = rng.randint

    def _next_frame(self) -> np.ndarray:
        """Implementation of the :py:class:`Iterator` interface.
        """
        shape = self._size[::-1] + (3,)
        return self._random_integers(0, 256, size=shape, dtype=np.uint8)


class VideoIteratoreReader(VideoReader):
    """An Iterator for Video objects.
    """
    _video: Video

    def __init__(self, video: Video, **kwargs) -> None:
        super().__init__(**kwargs)
        self._video = video

    def _next_frame(self) -> np.ndarray:
        frame = self._video[self._index]
        self._index += 1
        return frame


class VideoArray(Video):
    """A `VideoArray` provides access to a video stored in an array.
    """
    _frames: np.ndarray

    def __init__(self, frames: Optional[np.ndarray] = None,
                 reader: Optional[VideoReader] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if frames is not None:
            self._frames = frames
        elif reader is not None:
            shape = (len(reader), reader.size.height, reader.sizewidth,
                     reader.channels)
            self._frames = np.ndarray(shape)
            for index, frame in enumerate(reader):
                self._frames[index] = frame
        else:
            raise ValueError("")

        self._number_of_frames = len(self._frames)

    def _frame_at(self, index: int) -> np.ndarray:
        """Get a specific frame from this video.
        """
        return self._frames[index]


class VideoFileReader(VideoReader, FileBase, Implementable):
    """An abstract interface to read videos. A :py:class:`FileReader`
    extends the base :py:class:`Reader` by adding the idea of length:
    a video file has an beginning and an end.  Within this interval,
    the reader points to a current position from which a frame is
    read.

    """


class Webcam(VideoReader, Implementable):
    """An abstract webcam backend.

    Attributes
    ----------
    _device: int
        Device number of the webcam to use.

    _lock: threading.Lock
        A lock to avoid race conditions due to simultanous access to
        the webcam.
    """

    # maximal acceptable delay due to buffering
    maximal_buffering_delay: float = 0.1

    def __init__(self, device: int, **kwargs) -> None:
        LOG.info("Acqiring Webcam (%d) for %s.", device, type(self))
        self._lock = None
        super().__init__(**kwargs)
        self._device = device
        self._lock = threading.Lock()
        # Store time of last capture operation
        self._last_read_timestamp = now()

    def __del__(self) -> None:
        if self._lock is not None:
            del self._lock
            self._lock = None

    @property
    def device(self) -> int:
        """The webcam device number.
        """
        return self._device

    def read_frame(self, clear_buffer: bool = None) -> np.ndarray:
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
        LOG.debug("Capturing frame from %s (clear_buffer=%s)",
                  self, clear_buffer)
        with self._lock:
            if clear_buffer is None:
                time_passed = now() - self._last_read_timestamp
                clear_buffer = time_passed > self.maximal_buffering_delay
            if clear_buffer:
                self._clear_buffer()
            frame = self._next_frame()
            self._last_read_timestamp = now()
        LOG.debug("Captured frame of shape %s, dtype=%s with shape %s",
                  frame.shape, frame.dtype, self)
        return frame

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
                self._next_frame()


class VideoWriter(Preparable, Implementable):
    """A video :py:class:`VideoWriter` is a "sink" or "consumer" of video
    data, that is a sequence of frames (images).  A video can be
    written completely, in chunks or framewise.  There is also the
    option to asynchronously run a :py:class:`VideoWriter` either by
    regularly querying or by observing an :py:class:`ImageObservable`.

    A :py:class:`VideoWriter` can be associated with different speed
    parameters, all expressed as frame rate in frames per second
    (fps).  The kind of supported parameters depends on the type of
    the :py:class:`VideoWriter`.  One may control the operational speed of
    the :py:class:`VideoWriter`, meaning how many frames per second should
    be written.  If frames are written to the :py:class:`VideoWriter` at a
    higher rate, some frames may be skipped and if the
    :py:class:`VideoWriter` is actively querying an
    :py:class:`ImageObservable`, operational speed determines the
    maximal query frequency.

    A :py:class:`VideoWriter` storing videos to file (or some other
    form of storage) may save a frame rate as part of the metadata
    and will indicate the desired speed for playback. This can be set
    independent of the operational speed of the :py:class:`VideoWriter`.

    Frames may be provided to a :py:class:`VideoWriter` in form of
    :py:class:`Imagelike` objects.

    The :py:class:`VideoWriter` is an abstract base class that has to
    be subclassed to provide actual funcitonality.  The central method
    to be overwritten is :py:meth:`_write_frame` which will get a
    frame and should perform the write operation. If the
    :py:class:`VideoWriter` needs specific resources, those can be required
    by overwriting the :py:meth:`_open_writer` and
    :py:meth:`_close_writer` methods.

    """
    # FIXME[todo]: most of what is described in the docstring is not
    # yet implemented.

    _frame_format: ImageFormat

    # FIXME[old/todo]: integrate into the Implementation logic:
    #   - when constructing the classes NullWriter or Display, use
    #     that specific class
    #   - when a `filename` argument is provided, use an implementation
    #     of the VideoReader class
    #   - otherwise raise an error

    def __init__(self, frame_format: Optional[ImageFormat] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._frame_format = frame_format

    def __del__(self) -> None:
        if self.opened:
            self.close()

    def __call__(self, video: VideoReader, progress=None, **kwargs) -> None:
        if progress is not None:
            video = progress(video)
        with self:  # ensure writer is usable ("opened")
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

    def __enter__(self) -> 'VideoWriter':
        self._open()
        return self

    def __exit__(self, _exception_type, _exception_value, _traceback) -> None:
        self._close()

    #
    # public interface
    #

    @property
    def frame_format(self) -> ImageFormat:
        """The :py:class:`ImageFormat` used by this :py:class:`VideoWriter`.
        Frames written to this :py:class:`VideoWriter` are assumed to be in
        that format, except if explicitly stated otherwise (for
        example by providing explicit format information when calling
        :py:class:`write_frame` or when writing an :py:class:`Image`
        of another format).

        """
        return self._frame_format

    def write_frame(self, frame: Imagelike, **kwargs) -> None:
        """Write a frame to this video :py:class:`VideoWriter`.
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

    def copy(self, video, transform=None, progress=None):
        """Copy a given video to this :py:class:`VideoWriter`.
        """
        # FIXME[design]: there are multiple implementations similar
        # to this method:
        #   - experiments.video: class VideoOperator
        #   - dltb.util.video: function copy()
        if progress is not None:
            video = progress(video)
        for frame in video:
            if transform is not None:
                frame = transform(frame)
            self._write_frame(frame)

    #
    # private API (to be overwritten by subclasses)
    #

    def _write_frame(self, frame: np.ndarray) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} claims to "
                                  "be a VideoWriter, but provides no "
                                  "'_write_frame' method.")

    @staticmethod
    def _open() -> None:
        """Make sure the `VideoWriter` is open, that is, ready to write.
        If sufficient arguments are given to the constructor, this
        will be called automatically, otherwise it has to be called
        explicitly before the first read.
        """

    @staticmethod
    def _is_opened() -> bool:
        """Check if this :py:class:`VideoWriter` is still open, meaning
        that it accepts more frames to be written.
        """
        return True  # may be overwritten by subclasses

    @staticmethod
    def _close() -> None:
        """Actual implementation of the closing operation.
        This should release all system resources acquired by this
        :py:class:`VideoWriter`, like file handles, etc.
        """
        # may be overwritten by subclasses

    @staticmethod
    def _prepare_frame(frame: Imagelike) -> Any:
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


class NullWriter(VideoWriter):
    """The :py:class:`NullWriter` will discard all frames. It is meant a
    convenience class to be used when a :py:class:`VideoWriter` is
    required, but the written frames are not used.
    """

    def _write_frame(self, frame: np.ndarray) -> None:
        """Write a frame to this `NullWriter`.  The `NullWriter`
        simply ignores this frame.
        """


class VideoDisplay(VideoWriter):
    """A `VideoDisplay` displays a video by successively showing its
    frames.
    """
    # FIXME[todo]: introduce a show method (show the video)
    # either in blocking or non-blocking mode

    def __init__(self, display: ImageDisplay = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._display = ImageDisplay() if display is None else display

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type == ValueError and self._display.closed:
            return True  # ignore the exception
        return super().__exit__(exc_type, exc_value, exc_traceback)

    def _open(self) -> None:
        """Open this :py:class:`VideoWriter` for writing.  In the case
        of a :py:class:`VideoDisplay` this means preparing the display
        component for showing images.
        """
        # Do not use a display event loop: we will update the image
        # regularly, and if no event loop is running events will
        # be processed on each invocation of display.show(). This
        # should suffice for a smooth user interface except for
        # videos with a very low frame rate.
        self._display.show(blocking=False)

    def _write_frame(self, frame: np.ndarray) -> None:
        """Write a frame with this :py:class:`VideoWriter`. In the case
        of a :py:class:`VideoDisplay` this means to show the frame in the
        display component.

        The display is guaranteed to be opened when this message is
        called.

        Arguments
        ---------
        frame:
            The frame to be displayed as an array. The frame is
            expected to be in the standard format of this
            :py:class:`VideoWriter`, which defaults to `np.uint8` values,
            color images being in RGB color space.

        """
        self._display.show(frame)

    def close(self):
        """Close the :py:class:`VideoWriter`.  In the case
        of a :py:class:`VideoDisplay` this means to close the display
        component showing the images.
        """
        self._display.close()


class FileWriter(VideoWriter, FileBase): # pylint: disable=abstract-method
    """A `FileWriter` can write a video into a file.
    """


class VideoDirectory:
    """The :py:class:`VideoDirectory` provides base functionality for
    storing videos as a collection of individual images stored in
    a directory.
    """

    _frames_directory: Path
    _frame_index: Optional[int]

    def __init__(self, directory: Optional[Pathlike] = None, **kwargs) -> None:
        if directory is None:
            raise ValueError(f"No directory specified for {type(self).__name__}.")
        super().__init__(**kwargs)
        self._frames_directory = as_path(directory)
        self._frame_index = None

    @property
    def directory(self) -> str:
        """The video directory.
        """
        return str(self._frames_directory)

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


class DirectoryReader(VideoReader, VideoDirectory):
    """A :py:class:`VideoReader` for videos that are stored as individual
    images in a directory.
    """
    _len: Optional[int] = None

    def __init__(self, reader: Optional[ImageReader] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._image_reader = reader or ImageReader()

    def __len__(self) -> int:
        if self._len is None:
            if not self._frames_directory.isdir():
                raise RuntimeError("Frames directory "
                                   f"'{self._frames_directory}' "
                                   "does not exist.")
            self._len = len(os.listdir(self._frames_directory))
        return self._len

    def __getitem__(self, index: int) -> np.ndarray:
        if not self._frames_directory.isdir():
            raise RuntimeError(f"Frame directory '{self._frames_directory}' "
                               "does not exist.")
        if not 0 <= index < len(self):
            raise IndexError(f"Invalid frame index {index}"
                             f"is not in [0, {len(self)}]")
        return self._image_reader.read(self.filename_for_frame(index))


class DirectoryWriter(VideoWriter, VideoDirectory):
    """A :py:class:`DirectoryWriter` stores a video by writing individual
    frames as images into a directory.

    """

    def __init__(self, writer: Optional[ImageWriter] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._image_writer = writer or ImageWriter()

    def _prepare(self) -> None:
        """Prepare this :py:class:`DirectoryWriter` for writing.
        Thi will ensure that the directory exists.
        """
        super()._prepare()
        os.makedirs(self._frames_directory, exist_ok=True)
        self._frame_index = 0

    def _unprepare(self) -> None:
        """Release resources acquired by this :py:class:`VideoWriter`
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
            The frame to be written to this :py:class:`VideoWriter`. This
            is assumed to be given in the correct format (dtype, colorspace,
            size, etc.) as expected by the :py:class:`ImageWriter`
            employed by this :py:class:`DirectoryWriter`.
        """
        self._image_writer(self.filename_for_frame(self._frame_index), frame)
        self._frame_index += 1
