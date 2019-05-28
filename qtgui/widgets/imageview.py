from toolbox import Toolbox, View as ToolboxView
from tools.activation import Engine as ActivationEngine
from ..utils import QImageView, QObserver

import numpy as np
from scipy.misc import imresize

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget


# FIXME[todo]: add docstrings!

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
    _processed: bool = False
    
    def __init__(self, toolbox: ToolboxView=None, parent: QWidget=None):
        """Initialize this QModelImageView.

        Arguments
        ---------
        parent: QWidget
        """
        super().__init__(parent)
        self.modeChanged.connect(self.onModeChanged)
        self.setToolboxView(toolbox)

    def setToolboxView(self, toolbox: ToolboxView) -> None:
        self._exchangeView('_toolbox', toolbox,
                           interests=Toolbox.Change('input_changed'))

    def _setImageFromToolbox(self):
        """Set the image to be displayed based on the current state of
        the Toolbox. This will also take the current display mode (raw
        or processed) into account.
        """
        image = self._toolbox.input_data if self._toolbox else None
        self.setImage(image)
        return  # FIXME[hack]: switch between
        if self._activation is None:
            image = None
        elif self._processed:
            image = self._activation.input_data
        else:
            image = self._activation.raw_input_data
        self.setImage(image)
        
    def toolbox_changed(self, toolbox: Toolbox,
                        change: Toolbox.Change) -> None:
        self._setImageFromToolbox()

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
            self._setImageFromToolbox()

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

    # FIXME[old] but some parts should be recycled: convolution mask
    def _setImageFromModel(self):
        """Set the image to be displayed based on the current state of
        the model. This will also take the current display mode (raw
        or processed) into account.
        """
        print("FIXME: QModelImageView._setImageFromModel was ignored!")
        return
        if self._activation is None:
            image = None
        elif self._show_raw:
            image = self._activation.raw_input_data
        else:
            image = self._activation.input_data
        self.setImage(image)

    def activation_changed(self, activation: ActivationEngine,
                     info: ActivationEngine.Change) -> None:
        """
        The QModelImageView is mainly interested in 'input_changed'
        events. 
        """
        print("FIXME: QModelImageView.activation_changed was ignored!")
        return
        # FIXME[hack]: this is not an appropriate way to set the model!
        self._activation = activation

        # If the input changed, we will display the new input image
        if info.input_changed:
            self._setImageFromModel()

        # For convolutional layers add a activation mask on top of the
        # image, if a unit is selected
        activation = activation.get_activation()
        unit = activation.unit
        if (activation is not None and unit is not None and
            activation.ndim > 1):  # exclude dens layers
            from util import grayscaleNormalized
            activation_mask = grayscaleNormalized(activation[..., unit])
            self.setMask(activation_mask)
        else:
            self.setMask(None)

            
