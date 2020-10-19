"""A :py:class:`Datasource` reading frames from a video.
"""

# standard imports
import logging

# toolbox imports
from dltb.base.video import Reader
from dltb.base.data import Data
from .datasource import Indexed, Imagesource, Livesource

# logging
LOG = logging.getLogger(__name__)


# FIXME[error]: when looping over the end of the video, the
# loop continues but raises errors:


class Video(Indexed, Imagesource, Livesource):
    # pylint: disable=too-many-ancestors
    """A data source fetching frames from a video.

    Attributes
    ----------
    _backend:
        The VideoBackend use for accessing the video file.
    _filename:
        The filename frome which the video is read.
    _frame:
        The currently fetched frame.
        If the frame changes, Observers of this :py:class:`Video`
        will receive a `data_changed` notification.
    """

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

    #
    # Preparable
    #

    def _prepared(self) -> bool:
        """Report if this Datasource is prepared for use.
        A Datasource has to be prepared before it can be used.
        """
        return super()._prepared() and (self._backend is not None)

    def _prepare(self) -> None:
        """Prepare this Datasource for use.
        """
        super()._prepare()
        self._backend = Reader(filename=self._filename)
        self._loop_interval = 1. / self._backend.frames_per_second

    def _unprepare(self) -> None:
        """Unprepare this Datasource. This will free resources but
        the webcam can no longer be used.  Call :py:meth:`prepare`
        to prepare the webcam for another use.
        """
        if self._backend:
            del self._backend
            self._backend = None
        self._frame = None
        super()._unprepare()

    #
    # Data
    #

    def _get_meta(self, data: Data, **kwargs) -> None:
        # pylint: disable=arguments-differ
        """Get metadata for some data.
        """
        data.add_attribute('frame', batch=True)  # usually the same a index
        data.add_attribute('time', batch=True)
        super()._get_meta(data, **kwargs)

    def _get_frame_at(self, data: Data, time: float):
        """Fetch a video frame at a given timepoint from this
        :py:class:`Video`.

        Arguments
        ---------
        time: float
            The temporal position (point in time) of the frame
            to fetch in seconds (fractions may be given).
        """
        data.array = self._backend.get_frame_at(time)
        data.index = self._backend.frame

    def _get_data(self, data: Data, frame: int = None, time: float = None,
                  index: int = None, **kwargs) -> None:
        # pylint: disable=arguments-differ
        if frame is not None and not data:
            data.datasource_argument = 'frame'
            data.datasource_value = frame
            index = frame
        elif time is not None and not data:
            data.datasource_argument = 'time'
            data.datasource_value = time
            index = self._backend.frame_at(time)
        super()._get_data(data, index=index, **kwargs)

    def _get_default(self, data: Data, **kwargs) -> None:
        """The default is to obtain the next frame from the video.
        """
        self._get_index(data, index=None, **kwargs)  # get next frame

    def _get_snapshot(self, data, snapshot: bool = True, **kwargs) -> None:
        """Reading a sanpshot from the video means reading the next frame.

        Arguments
        ---------
        snapshot: bool
            If True, try to make sure that we get a current snapshot.
            On some systems, the video driver buffers some frame, so that
            reading just reading the frame may result in outdated data and
            one should first empty the buffer before reading the data.

        """
        LOG.debug("Video._get_data(snapshot=%r)", snapshot)
        self._get_index(data, index=None, **kwargs)  # get next frame

    def _get_index(self, data: Data, index: int, **kwargs) -> None:
        """Implementation of the :py:class:`Indexed` datasource interface.
        The index will be interpreted as frame number.

        Arguments
        ---------
        index: int
            The frame number of the frame to fetch from the video.
            If no frame is given, the next frame of the video will
            be fetched (advancing the frame number by 1).
        """
        data.array = self._backend[index]
        data.frame = self._backend.frame
        data.index = index or data.frame
        data.time = self._backend.time

    #
    # Public Video API
    #

    @property
    def frame(self) -> int:
        """The current frame of this video.
        The next frame read will be frame + 1.
        """
        return self._backend.frame

    @property
    def time(self) -> float:
        """The current frame time of this video.
        """
        return self._backend.time

    @property
    def frames_per_second(self) -> float:
        """Frames per second for this video.
        """
        return self._backend.frames_per_second

    #
    # Implementation of the 'Indexed' API
    #

    def __len__(self) -> int:
        """The length of a video is the number of frames it is composed of.
        """
        if not self.prepared:
            raise RuntimeError("Applying len() to unprepare Video object.")
        return len(self._backend)
