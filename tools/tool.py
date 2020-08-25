"""Abstract base class for tools.
"""

# standard imports
import os
import sys
import time
import importlib
import logging

# toolbox imports
from base import Preparable, MetaRegister, busy
from datasource import Data

# logging
LOG = logging.getLogger(__name__)


# FIXME[todo]: move Resource to some other package (e.g. base) and
# merge with other Resource classes ...
class Resource(Preparable):
    """A resource may have some requirements to be used.
    It may provide methods to install such requirements.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._requirements = {}

    #
    # Requirements
    #

    def _add_requirement(self, name, what, *data) -> None:
        """Add a requirement for this :py:class:`Tool`.
        """
        self._requirements[name] = (what,) + data

    #
    # Preparable
    #

    def _preparable(self) -> bool:
        """Check if required resources are available.
        """
        for name, requirement in self._requirements.items():
            if requirement[0] == 'file':
                if not os.path.exists(requirement[1]):
                    LOG.warning("File requirement '%s' (filename='%s') "
                                "for resource '%s' (%s) not found.", name,
                                requirement[1], self.key, type(self).__name__)
                    return False
            if requirement[0] == 'module':
                if requirement[1] in sys.modules:
                    continue
                spec = importlib.util.find_spec(requirement[1])
                if spec is None:
                    LOG.warning("Module requirement '%s' (module=%s) "
                                "for resource '%s' (%s) not found.", name,
                                requirement[1], self.key, type(self).__name__)
                    return False
        return True

    def _prepare(self, install: bool = False, **kwargs):
        # pylint: disable=arguments-differ
        """Load the required resources.
        """
        super()._prepare(**kwargs)

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

        if not self.preparable:
            if install:
                self.install()
            else:
                raise RuntimeError("Resources required to prepare '" +
                                   type(self).__name__ +
                                   "' are not installed.")

    #
    # Installation
    #

    def install(self) -> None:
        """Install the resources required for this module.
        """
        LOG.info("Installing requirements for resource '%s'.",
                 self.__class__.__name__)
        start = time.time()
        self._install()
        end = time.time()
        LOG.info("Installation of requirements for resource '%s' "
                 "finished after %.2fs",
                 self.__class__.__name__, end-start)

    def _install(self) -> None:
        # FIXME[concept]: what is this method supposed to do
        # and which (sub)classes should implement this method.
        """Actual implementation of the installation procedure.
        """
        pass
        # raise NotImplementedError("Installation of resources for '" +
        #                           type(self).__name__ +
        #                           "' is not implemented (yet).")


class Tool(Resource, metaclass=MetaRegister):
    # pylint: disable=too-many-ancestors
    """:py:class:`Tool` is an abstract base class for tools.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        LOG.info("New tool: %s (%s)", self.key, type(self).__name__)

    def _prepare(self, **kwargs):
        # pylint: disable=arguments-differ
        """Prepare this tool.
        """
        super()._prepare(**kwargs)


class Processor(Tool):

    def process(self, data: Data) -> None:
        """Run a data processing loop. This will set the detector
        into a busy state ("processing"), in which new input data
        are processed until no more new data are provided.
        If new data is given, before the previously provided data
        was processed, the previous data will be skipped.
        When processing one data item finishes, observers will
        receive a 'detection_finished' notification, and can obtain
        the data object including the detections via the
        :py:meth:`data` property. The detections can be accessed
        as a data attribute named by the :py:meth:`detector` property.

        The main motivation for this method is to process data from
        a data loop (like a webcam or a video) in real-time, always
        processing the most recent data available.
        """
        LOG.info("Detector '%s' processes data %r", self.key, data)
        self._next_data = data
        if not self.busy:
            self._process()

    @busy("processing")
    def _process(self):
        """The implementation of the process loop. This method
        is assumed to run in a background thread. It will
        check the property `_next_data` for fresh data and
        if present, it will hand (a copy) of this data to the
        detector and otherwise it will end the loop.

        The data object passed to the detector will also
        be stored as attribute (using the unique key of the detector
        as attribute name) in the original data passed to the
        detector. The results of the detector will be stored
        in detector data under the attribute `detections`.
        """
        while self._next_data is not None:

            data = self._next_data
            self._next_data = None
            self._process_data(data)

    def _process_data(self, data: Data) -> None:
        """Adapt the data.
        To be implemented by subclasses. Subclasses may augment
        the given data object.
        """
        self._data = data 
