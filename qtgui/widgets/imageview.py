import numpy as np
from scipy.misc import imresize

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget

from model import Model, ModelObserver, ModelChange
from qtgui.utils import QImageView

# FIXME[todo]: add docstrings!


class QModelImageView(QImageView, ModelObserver):
    """A QImageView to display the input image of the model.

    Attributes
    ----------
    _model: Model
        The model observed by this QModelImageView.
    _show_raw: bool
        A flag indicating if the raw or the preprocessed input
        to the network is shown.

    Signals
    -------
    """

    modeChange = pyqtSignal(bool)

    def __init__(self, parent: QWidget=None):
        """Initialize this QModelImageView.

        Arguments
        ---------
        parent: QWidget
        """
        super().__init__(parent)
        self._show_raw: bool = False
        self.modeChange.connect(self.onModeChange)
        # FIXME[hack]: remove
        self._info_box = None
        
    def _setImageFromModel(self):
        """Set the image to be displayed based on the current state of
        the model. This will also take the current display mode (raw
        or reshaped) into account.
        """
        if self._model is None:
            image = None
        elif self._show_raw:
            image = self._model.raw_input_data
        else:
            image = self._model.input_data
        self.setImage(image)

    def modelChanged(self, model: Model, info: ModelChange):
        """
        The QModelImageView is mainly interested in 'input_changed'
        events. 
        """
        # FIXME[hack]: this is not an appropriate way to set the model!
        self._model = model

        # If the input changed, we will display the new input image
        if info.input_changed:
            self._setImageFromModel()

        # For convolutional layers add a activation mask on top of the
        # image, if a unit is selected
        activation = model._current_activation
        unit = model.unit
        if (activation is not None and unit is not None and
            activation.ndim > 1):  # exclude dens layers
            from util import grayscaleNormalized
            activation_mask = grayscaleNormalized(activation[..., unit])
            self.setMask(activation_mask)
        else:
            self.setMask(None)

    def onModeChange(self, showRaw: bool):
        """The display mode was changed.

        Arguments
        ---------
        mode: bool
            The new display mode (False=raw, True=reshaped).
        """
        self._show_raw = showRaw
        self._setImageFromModel()

    def mousePressEvent(self, event):
        """A mouse press toggles between raw and reshaped mode.
        """
        self.modeChange.emit(not self._show_raw)

    # FIXME[todo]: does not work!
    def keyPressEvent(self, event):
        key = event.key()
        print(f"{self.__class__.__name__}.keyPressEvent(key)")
        if key == Qt.Key_Space:
            self.modeChange.emit(not self._show_raw)
