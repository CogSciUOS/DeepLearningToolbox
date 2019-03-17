import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget
from qtgui.widgets import QMatrixView
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QSplitter

from qtgui.widgets import QActivationView
from qtgui.widgets import QInputSelector, QInputInfoBox, QModelImageView
from qtgui.utils import QImageView

from .panel import Panel


class OcclusionPanel(Panel):
    '''Panel for visualization of unit occlusion.'''

    def __init__(self, parent=None):
        '''Initialization of the OcclusionPanel.

        Parameters
        ---------
        parent  :   QWidget
                    The parent argument is sent to the QWidget constructor.
        '''

        super().__init__(parent)
        self._initUI()
        self._layoutUI()

    def _initUI(self):
        '''Initialise all UI elements. These are

            * The ``QImageView`` showing the occlusion overlay
            * The ``QImageView`` showing the current input image
            * A ``QInputInfoBox`` to display information about the input

        '''
        #super().initUI()
        self._occlusion_view = QImageView(self)

    def _layoutUI(self):
        # FIXME[layout]
        self._occlusion_view.setMinimumWidth(300)
        self._occlusion_view.resize(600, self._occlusion_view.height())
        occlusionLayout = QVBoxLayout()
        occlusionLayout.addWidget(self._occlusion_view)
        occlusionBox = QGroupBox('Occlusion')
        occlusionBox.setLayout(occlusionLayout)

        #######################################################################
        #                      Attach widgets to window                       #
        #######################################################################
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(occlusionBox)
        #splitter.addWidget(self._input_box)
        layout = QHBoxLayout()
        layout.addWidget(splitter)
        #layout.addWidget(self._network_box)
        self.setLayout(layout)

    def updateOcclusion(self):
        if not self._network or self._data is None:
            occlusion = None
        else:
            win_size = 1
            # occlusion = self._network.get_occlusion_map(self._data, win_size)

            # self._occlusion_view.setImage(self._data)
            # self._occlusion_view.setActivationMask(occlusion[0, :, :, 0])

    def setInput(self, input: int = None):
        '''Set the current input stimulus for the network.
        The input stimulus is take from the internal data collection.

        Parameters
        ----------
        input:
            The index of the input stimulus in the data collection.
        '''
        if input is None or self.data is None:
            self._data_index = None
        else:
            self._data_index = input
            self.updateInput()

    def setLayer(self, layer=None):
        super().setLayer(layer=layer)
        self.updateOcclusion()


    def setInputData(self, raw: np.ndarray=None, fitted: np.ndarray=None,
                     description: str=None):
        super().setInputData(raw, fitted, description)
        self.updateOcclusion()
