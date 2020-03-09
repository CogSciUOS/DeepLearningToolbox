from . import Datasource, InputData, Loop, Snapshot

import sys
import time
import importlib
import numpy as np

class DataWebcam(Loop, Snapshot):
    """A data source fetching images from the webcam.

    Attributes
    ----------
    _capture:
        A capture object
    """
    _device: int = 0
    _capture = None  # cv2.Capture
    _frame: np.ndarray = None

    @staticmethod
    def check_availability():
        """Check if this Datasource is available.

        Returns
        -------
        True if the OpenCV library is available, False otherwise.
        """
        return importlib.util.find_spec('cv2')

    def __init__(self, id: str="Webcam", description: str="<Webcam>",
                 device: int=0, **kwargs):
        """Create a new DataWebcam

        Raises
        ------
        ImportError:
            The OpenCV module is not available.
        """
        super().__init__(id=id, description=description, **kwargs)

        # FIXME[bug]: for some reason, the Loop constructor is not called!
        self._loop_interval = 0.2


    @property
    def prepared(self) -> bool:
        """Report if this Datasource prepared for use.
        A Datasource has to be prepared before it can be used.
        """
        return self._capture is not None

    def _prepare_data(self):
        """Prepare this Datasource for use.
        """
        from cv2 import VideoCapture
        self._capture = VideoCapture(self._device)
        if not self._capture:
            raise RuntimeError("Acquiring video capture failed!")

    def _unprepare_data(self):
        """Unprepare this Datasource. This will free resources but
        the webcam can no longer be used.  Call :py:meth:`prepare`
        to prepare the webcam for another use.
        """
        self._capture.release()
        self._capture = None
        self._frame = None

    @property
    def fetched(self):
        return self._frame is not None

    def _fetch(self, **kwargs):
        ret, frame = self._capture.read()
        if not ret:
            raise RuntimeError("Reading an image from video capture failed!")
        self._frame = frame[:,:,::-1]

    def run_loop(self):
        if sys.platform == 'linux':
            # Hack: under linux, the av-based linux capture code is using
            # an internal fifo (5 frames, iirc), and you cannot clean (or
            # say, flush) it.
            #
            # Hence we will apply another loop logic: we read frames as
            # fast as possible and only report them at certain times.
            last_time = time.time()
            while self._loop_running:
                if time.time() - last_time > self._loop_interval:
                    self.fetch(random=True)
                else:
                    ret, frame = self._capture.read()
            self._loopEvent = None
        else:
            super().run_loop()

    def snapshot(self) -> None:
        """Create a snapshot.
        """
        # Hack: under linux, the av-based linux capture code is using
        # an internal fifo (5 frames, iirc), and you cannot clean (or
        # say, flush) it.
        #
        # Hence we will skip some frames to really get the current image.
        if sys.platform == 'linux':
            ignore = 4
            for i in range(ignore):
                ret, frame = self._capture.read()
                if not ret:
                    raise RuntimeError("Snapshot: Reading an image from "
                                       "video capture failed!")
        super().snapshot()

    def _get_data(self):
        return self._frame

    def __str__(self):
        return "Webcam"
