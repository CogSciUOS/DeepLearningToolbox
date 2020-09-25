"""Access to the imageio library

The following main classes of imageio expose an interface for
advanced users. A brief overview:

  * imageio.FormatManager - for keeping track of registered formats.
  * imageio.Format - representation of a file format reader/writer
  * imageio.Format.Reader - object used during the reading of a file.
  * imageio.Format.Writer - object used during saving a file.
  * imageio.Request - used to store the filename and other info.
"""

# standard imports
import logging

# third party imports
import imageio
import imageio_ffmpeg
import numpy as np

# toolbox imports
from ..base import image, video

# logging
LOG = logging.getLogger(__name__)


class ImageIO(image.ImageReader, image.ImageWriter):

    def read(self, filename: str, **kwargs) -> np.ndarray:
        return imageio.imread(filename)

    def write(self, image: np.ndarray, filename: str, **kwargs) -> None:
        imageio.imwrite(filename)


class VideoReader(video.FileReader):
    """A :py:class:`BaseFileReader` realized by the imageio library.


    To use this plugin, the imageio-ffmpeg library should be installed.


    The :py:class:`imageio.plugins.ffmpeg.FfmpegFormat.Reader`
    provides the following properties:
    * closed: bool
    * request: imageio.core.request.Request
    * format: <Format FFMPEG - Many video formats and cameras (via ffmpeg)>

    methods:
    * count_frames()
        Count the number of frames.
    * get_meta_data()
        {'plugin': 'ffmpeg',
         'nframes': inf,
         'ffmpeg_version': '4.1.3 built with gcc 7.3.0 ...',
         'codec': 'h264',
         'pix_fmt': 'yuv420p',
         'fps': 25.0,
         'source_size': (460, 360),
         'size': (460, 360),
         'duration': 230.18}
    * get_data()
    * get_next_data()
    * iter_data(): generator
    * set_image_index()
        Set the internal pointer such that the next call to
        get_next_data() returns the image specified by the index
    * get_length(): can be inf
    * close()

    Attributes
    ----------
    _reader:
        The underlying imageio reader object.

    _index: int
        The index of the current frame.
    """

    def __init__(self, filename: str, **kwargs) -> None:
        self._reader = None
        super().__init__(filename=filename, **kwargs)
        self._reader = imageio.get_reader(filename)
        if self._reader is None:
            LOG.error("Opening movie file (%s) failed", filename)
            raise RuntimeError("Creating video reader object for file "
                               f"'{filename}' failed.")
        self._meta = self._reader.get_meta_data()
        self._index = -1
        LOG.debug("Reader object: %r", self._reader)
        LOG.info("Video file: %s", filename)
        LOG.info("FFMPEG backend version %s (%s)",
                 imageio_ffmpeg.get_ffmpeg_version(),
                 imageio_ffmpeg.get_ffmpeg_exe())

    def __del__(self) -> None:
        """Destructor for this :py:class:`WebcamBackend`.
        The underlying :py:class:`.imageio.Reader` object will be
        closed and deleted.
        """
        if self._reader is not None:
            LOG.info("Releasing video Reader (%r)", self._reader)
            self._reader.close()
            del self._reader
            self._reader = None

    #
    # Iterator
    #
            
    def __next__(self) -> np.ndarray:
        try:
            frame = self._reader.get_next_data()
            self._index += 1
        except IndexError as ex:
            raise StopIteration("IndexError ({ex})")

        if frame is None:
            raise RuntimeError("Reading a frame from "
                               "ImageIO Video Reader failed!")
        return frame

    def __len__(self) -> int:
        # When reading from a video, the number of available frames is
        # hard/expensive to calculate, which is why its set to inf by
        # default, indicating “stream mode”. To get the number of
        # frames before having read them all, you can use the
        # reader.count_frames() method (the reader will then use
        # imageio_ffmpeg.count_frames_and_secs() to get the exact
        # number of frames, note that this operation can take a few
        # seconds on large files). Alternatively, the number of frames
        # can be estimated from the fps and duration in the meta data
        # (though these values themselves are not always
        # present/reliable).
        return self._reader.count_frames()

    def __getitem__(self, index: int) -> np.ndarray:
        """Get the frame for a given frame number.

        Note: after getting a frame, the current frame number (obtained
        by :py:meth:`frame`) will be frame+1, that is the number of
        the next frame to read.

        Arguments
        ---------
        index: int
            The number of the frame to be read. If no frame is given,
            the next frame available will be read from the capture
            object.
        """
        if index is None:
            return next(self)
        self._index = index
        image = self._reader.get_data(-index)
        return image

    @property
    def frames_per_second(self) -> float:
        """Frames per second in this video.
        """
        return self._meta['fps']

    @property
    def frame(self) -> int:
        return self._index


class Webcam(video.Webcam):
    """A :py:class:`WebcamBackend` realized by an OpenCV
    :py:class:`imageio.Reader` object.


    Attributes
    ----------
    _reader: imageio.plugins.ffmpeg.FfmpegFormat.Reader
        A video reader object
    """

    def __init__(self, device: int = 0):
        """Constructor for this :py:class:`WebcamBackend`.
        The underlying :py:class:`imageio.Reader` object will be
        created.
        """
        super().__init__(device)
        self._reader = imageio.get_reader(f'<video{device}>')
        if not self._reader:
            LOG.error("Acqiring Webcam (%d) failed", device)
            raise RuntimeError("Creating video reader object for camera "
                               f"'{device}' failed.")
        LOG.debug("Reader object: %r", self._reader)
        LOG.info("Camera device: %d", device)

    def __del__(self):
        """Destructor for this :py:class:`WebcamBackend`.
        The underlying :py:class:`.imageio.Reader` object will be
        closed and deleted.
        """
        if self._reader is not None:
            LOG.info("Releasing Webcam Reader (%d)", self._device)
            self._reader.close()
            del self._reader
            self._reader = None
        super().__del__()

    def _get_frame(self) -> np.ndarray:
        """Get the next frame from the ImageIO Video Reader.
        """
        frame, meta = self._reader.get_next_data()
        if frame is None:
            raise RuntimeError("Reading a frame from "
                               "ImageIO Video Reader failed!")
        return frame


class VideoWriter(video.FileWriter):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._writer = None

    def _open(self) -> None:
        if self._is_opened():
            raise RuntimeError("Video was already opened.")
        if self._filename is None:
            raise RuntimeError("No video filename was provided.")
        print("Opening:", self._filename, self._fps)
        self._writer = imageio.get_writer(self._filename, fps=int(self._fps))

    def _is_opened(self) -> bool:
        return self._writer is not None

    def _close(self) -> None:
        print("Closing:", self._filename, self._fps)
        if self._is_opened():
            self._writer.close()
            self._writer = None
        
    def _write_frame(self, frame: np.ndarray) -> None:
        writer.append_data(marked_image)
