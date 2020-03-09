import os
import sys
import time
import importlib
import numpy as np
import imutils

from base.observer import BusyObservable, change, busy
from base import View as BaseView, Controller as BaseController, run
from datasource import Metadata

class Detector(BusyObservable, method='detector_changed',
               changes=['data_changed', 'detection_finished']):
    """A general detector. 

    detector_changed:
        The input data given to the detector have changed.

    detection_finished:
        The detection has finished. This is reported when the
        detector has finished its work.

    """    
    _requirements = None


    def __init__(self):
        super().__init__()
        self._requirements = {}

    def _add_requirement(self, name, what, *data):
        self._requirements[name] = (what,) + data

    def _remove_requirement(self, name):
        self._requirements.pop(name, None)

    def available(self, verbose=True):
        """Check if required resources are available.
        """
        for name, requirement in self._requirements.items():
            if requirement[0] == 'file':
                if not os.path.exists(requirement[1]):
                    if verbose:
                        print(type(self).__name__ +
                              f": File '{requirement[1]}' not found")
                    return False
            if requirement[0] == 'module':
                if requirement[1] in sys.modules:
                    continue
                spec = importlib.util.find_spec(requirement[1])
                if spec is None:
                    print(type(self).__name__ +
                          f": Module '{requirement[1]}' not found")
                    return False
        return True
        
    def install(self):
        """Install the resources required for this module.
        """
        raise NotImplementedError("Installation of resources for '" +
                                  type(self).__name__ +
                                  "' is not implemented (yet).")

    @busy
    def prepare(self, install: bool=False):
        """Load the required resources.
        """
        if self.prepared():
            return

        # FIXME[concept]:
        # In some situations, one requirement has to be prepared in
        # order to check for other requirements.
        # Example: checking the availability of an OpenCV data file
        # may require the 'cv2' module to be loaded in order to construct
        # the full path to that file.

        for name, requirement in self._requirements.items():
            if requirement[0] == 'module' and requirement[1] not in globals():
                globals()[requirement[1]] = \
                    importlib.import_module(requirement[1])

        if not self.available(verbose=True):
            if install:
                self.install()
            else:
                raise RuntimeError("Resources required to prepare '" +
                                   type(self).__name__ +
                                   "' are not installed.")
        self._prepare()

    def _prepare(self):
        pass

    def prepared(self):
        return True

    def detect(self, data, **kwargs):
        if not self.prepared():
            raise RuntimeError("Running unprepared detector.")

        if data is None:
            return None
        
        start = time.time()
        detections = self._detect(data, **kwargs)
        end = time.time()

        if detections is not None:
            detections.duration = end - start

        return detections

    def _detect(self, image: np.ndarray, **kwargs) -> Metadata:
        """Do the actual detection.

        The detector will return a Metadata structure containing the
        detections as a list of :py:class:`Location`s (usually of type
        :py:class:`BoundingBox`) in the 'regions' property.
        """
        raise NotImplementedError("Detector class '" +
                                  type(self).__name__ +
                                  "' is not implemented (yet).")


class DetectorView(BaseView, view_type=Detector):
    """Viewer for :py:class:`Engine`.

    Attributes
    ----------
    _engine: Engine
        The engine controlled by this Controller

    """

    def __init__(self, engine: Detector=None, **kwargs):
        super().__init__(observable=engine, **kwargs)


class DetectorController(DetectorView, BaseController):
    """Controller for :py:class:`Engine`.
    This class contains callbacks for all kinds of events which are
    effected by the user in the ``MaximizationPanel``.

    Attributes
    ----------
    _engine: Engine
        The engine controlled by this Controller
    """
    
    def __init__(self, engine: Detector, **kwargs) -> None:
        """
        Parameters
        ----------
        engine: Engine
        """
        super().__init__(engine=engine, **kwargs)
        self._data = None
        self._next_data = None
        self._detections = None

    @run
    def prepare(self, install: bool=False):
        print(f"Prepare {self.__class__.__name__} for {self._detector.__class__.__name__}:")
        start = time.time()
        self._detector.prepare(install)
        end = time.time()
        print(f"Preparation of {self.__class__.__name__} for {self._detector.__class__.__name__} finished after {end-start:.4f}s")

    def process(self, data):
        """Process the given data.

        """
        self._next_data = data
        if not self._detector.busy:
            self._detector.busy = True  # FIXME[todo]: better API
            self._process()

    @run
    def _process(self):
        #self._detector.busy = True
        while self._next_data is not None:

            self._data = self._next_data
            self._next_data = None
            self._detector.change(data_changed=True)

            print(f"Detect {self.__class__.__name__} for {self._detector.__class__.__name__}:")
            
            self._detections = self._detector.detect(self._data)
            print(f"Detection of {self.__class__.__name__} for {self._detector.__class__.__name__} finished after {self._detections.duration:.4f}s")
            self._detector.change(detection_finished=True)

        self._detector.busy = False

    @property
    def data(self):
        return self._data

    @property
    def detections(self) -> Metadata:
        """A metadata holding the results of the last invocation
        of the detector.
        """
        return self._detections   


class ImageDetector(Detector):

    
    def detect(self, image, **kwargs):
        if image is None:
            return None
        
        resize_ratio = image.shape[1]/400.0
        image = imutils.resize(image, width=400)

        if image.ndim != 2 and image.ndim != 3:
            raise ValueError("The image provided has an illegal format: "
                             f"shape={image.shape}, dtype={image.dtype}")

        detections = super().detect(image, **kwargs)
        
        if detections is not None and resize_ratio != 1.0:
            detections.scale(resize_ratio)

        return detections


# FIXME[todo]: Controller logic: deriving a Controller
# make sure that this Controller controls an
# ImageDetector (not any general Detector)
class ImageController(DetectorController):

    @property
    def image(self):
        return self.data
