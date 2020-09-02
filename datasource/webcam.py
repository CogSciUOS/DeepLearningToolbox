"""A datasource reading images from a webcam.

This module mainly provides the class :py:class:`Webcam` that is
a Datasource interface for accessing the a webcam.

Examples
--------

    from datasource.webcam import Webcam

    webcam = Webcam()
    webcam.prepare()

    frame = webcam.get_data()
    image = frame.data

"""
# standard imports
import sys
import time
import threading
import logging

# toolbox imports
from dltb.base.video import Reader
from .datasource import Data, Imagesource, Loop, Snapshot

# logging
LOG = logging.getLogger(__name__)


class DataWebcam(Imagesource, Loop, Snapshot):
    # pylint: disable=too-many-ancestors
    """A data source fetching images from the webcam.

    Attributes
    ----------
    _backend: WebcamBackend
        The WebcamBackend use for accessing the Webcam.

    _device: int
        Device number of the webcam to use.
    """

    def __init__(self, key: str = "Webcam", description: str = "<Webcam>",
                 device: int = 0, **kwargs) -> None:
        """Create a new DataWebcam

        Raises
        ------
        ImportError:
            The OpenCV module is not available.
        """
        super().__init__(key=key, description=description, **kwargs)
        self._device = device
        self._backend = None
        self._loop_thread = None
        self._loop_data = None

    def __str__(self) -> str:
        return "Webcam"

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
        self._backend = Reader(device=self._device)
        # FIXME[todo]: we need some mechanism to decide which backend to use
        # self._backend = OpencvWebcamBackend(self._device)
        # self._backend = ImageioWebcamBackend(self._device)

    def _unprepare(self) -> None:
        """Unprepare this Datasource. This will free resources but
        the webcam can no longer be used.  Call :py:meth:`prepare`
        to prepare the webcam for another use.
        """
        if self._backend:
            del self._backend
            self._backend = None
        super()._unprepare()

    #
    # Data
    #

    def _get_meta(self, data: Data, **kwargs) -> None:
        # pylint: disable=arguments-differ
        """Get metadata for some data.
        """
        data.add_attribute('timestamp', batch=True)
        super()._get_meta(data, **kwargs)

    def _get_snapshot(self, data, snapshot: bool = True, **kwargs) -> None:
        """The default (and only) mode of getting data from the webcam
        is reading the next frame.

        Arguments
        ---------
        snapshot: bool
            If True, try to make sure that we get a current snapshot.
            On some systems, the video driver buffers some frame, so that
            reading just reading the frame may result in outdated data and
            one should first empty the buffer before reading the data.
        """
        LOG.debug("Webcam._get_data(snapshot=%r)", snapshot)
        data.data = (self._loop_data if self.looping else
                     self._backend.get_frame(clear_buffer=snapshot))
        super()._get_snapshot(data, snapshot, **kwargs)

    #
    # Loop
    #

    def start_loop(self):
        """Start an asynchronous loop cycle. This method will return
        immediately, running the loop cycle in a background thread.
        """
        super().start_loop()
        if sys.platform == 'linux':
            self._loop_thread = threading.Thread(target=self._run_loop_linux)
            self._loop_thread.start()
            LOG.info("Webcam: loop thread started.")

    def stop_loop(self):
        """Stop a currently running loop.
        """
        super().stop_loop()
        if sys.platform == 'linux':
            self._loop_thread.join()
            self._loop_thread = None
            LOG.info("Webcam: loop thread joined.")

    def _run_loop_linux(self) -> None:
        """Under linux, the av-based linux capture code is using
        an internal fifo (5 frames, iirc), and you cannot clean (or
        say, flush) it.

        Hence we will apply another loop logic: we read frames as
        fast as possible and only report them at certain times.
        """
        LOG.info("Webcam: employing linux loop")
        fetched = 0
        start_time = time.time()
        while not self.loop_stop_event.is_set():
            self._loop_data = self._backend.get_frame(clear_buffer=False)
            last_time = time.time()
            fetched += 1
            LOG.debug("Webcam: fetched: %d, frames per second: %.1f",
                      fetched, fetched/(last_time-start_time))

    #
    # information
    #

    def _get_description(self) -> str:
        description = super()._get_description()
        description += ", backend="
        description += ("None" if self._backend is None else
                        (type(self._backend).__module__ + '.' +
                         type(self._backend).__name__))
        return description
