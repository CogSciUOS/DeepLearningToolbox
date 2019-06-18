from toolbox import Toolbox, View as ToolboxView
from tools.activation import (Engine as ActivationEngine,
                              View as ActivationView)
from ..utils import QImageView, QObserver

import numpy as np
from scipy.misc import imresize

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget


# FIXME[todo]: add docstrings!
# FIXME[todo]: rename: QModelImageView is an old name ...


class QModelImageView(QImageView, QObserver, Toolbox.Observer,
                      ActivationEngine.Observer):
    """A :py:class:`QImageView` to display the input image of the
    Toolbox or ActivationEngine.


    Toolbox.Observer
    ----------------
    The QModelImageView can observe a Toolbox to always display the
    current input image. If a 'input_changed' is reported, the
    QModelImageView is updated to display the new input image.


    ActivationEngine.Observer
    -------------------------
    FIXME[todo]: Currently not implemented!
    

    Attributes
    ----------
    _toolbox: ToolboxView
        The toolbox we want to observe.
    _processed: bool
        A flag indicating if the raw or the preprocessed input
        data should be shown.

    _activation: ActivationEngine
        The activation Engine observed by this QModelImageView.

    Signals
    -------
    """
    modeChanged = pyqtSignal(bool)

    _toolbox: ToolboxView = None
    _activation: ActivationView = None
    _processed: bool = False
    
    def __init__(self, toolbox: ToolboxView=None,
                 activations: ActivationView=None,
                 parent: QWidget=None):
        """Initialize this QModelImageView.

        Arguments
        ---------
        parent: QWidget
        """
        super().__init__(parent)
        self.modeChanged.connect(self.onModeChanged)
        self.setToolboxView(toolbox)
        self.setActivationView(activations)

    #
    # Toolbox.Observer
    #

    def setToolboxView(self, toolbox: ToolboxView) -> None:
        self._exchangeView('_toolbox', toolbox,
                           interests=Toolbox.Change('input_changed'))

    def toolbox_changed(self, toolbox: Toolbox, info: Toolbox.Change) -> None:
        if info.input_changed:
            # Just use the image from the Toolbox if no ActivationView
            # is available - otherwise we will use the image(s) from the
            # ActivationView (which will inform us via activation_changed ...)
            if self._activation is None:
                self._updateImage()
                self.setMask(None)

    #
    # ActivationEngine.Observer
    #

    def setActivationView(self, toolbox: ActivationView) -> None:
        interests = ActivationEngine.\
            Change('activation_changed', 'input_changed', 'unit_changed')
        self._exchangeView('_activation', toolbox, interests=interests)

    def activation_changed(self, engine: ActivationEngine,
                           info: ActivationEngine.Change) -> None:
        """The :py:class:`QModelImageView` is interested in the
        input iamges, activations and units.
        """

        if info.input_changed:
            self._updateImage()

        if info.activation_changed or info.unit_changed:
            try:
                activation = engine.get_activation()
                unit = engine.unit
            except:
                activation = None
                unit = None

            # For convolutional layers add a activation mask on top of the
            # image, if a unit is selected
            if (activation is not None and unit is not None and
                activation.ndim > 1):
                # exclude dense layers
                from util import grayscaleNormalized
                activation_mask = grayscaleNormalized(activation[..., unit])
                self.setMask(activation_mask)
            else:
                self.setMask(None)

    #
    # Update
    #

    def _updateImage(self) -> None:
        """Set the image to be displayed either based on the current
        state of the Toolbox or the ActivationView. This will also
        take the current display mode (raw or processed) into account.
        """
        if self._activation is None:
            image = self._toolbox.input_data if self._toolbox else None
        elif self._processed:
            image = self._activation.input_data
        else:
            image = self._activation.raw_input_data
            
        self.setImage(image)


    @pyqtSlot(bool)
    def onModeChanged(self, processed: bool) -> None:
        """The display mode was changed. There are two possible modes:
        (1) processed=False (raw): the input is shown as provided by the source,
        (2) processed=True: the input is shown as presented to the network.

        Arguments
        ---------
        precessed: bool
            The new display mode (False=raw, True=processed).
        """
        self.setMode(processed)

    def setMode(self, processed: bool) -> None:
        if processed != self._processed:
            self._processed = processed
            self.modeChanged.emit(processed)
            self._updateImage()

    def mousePressEvent(self, event):
        """A mouse press toggles between raw and processed mode.
        """
        self.setMode(not self._processed)

    # FIXME[todo]: does not work!
    def keyPressEvent(self, event):
        key = event.key()
        print(f"{self.__class__.__name__}.keyPressEvent({key})")
        if key == Qt.Key_Space:
            self.setMode(not self._processed)

