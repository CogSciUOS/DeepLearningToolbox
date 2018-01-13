import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QComboBox
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QSplitter

from qtgui.widgets import QActivationView
from qtgui.widgets import QInputSelector, QInputInfoBox, QImageView
from qtgui.widgets import QNetworkView, QNetworkInfoBox

class Panel(QWidget):
    '''Base class for different visualisation panels. In the future, the type of
    visualisation should be a component in the panel, not a separate panel
    class.
    '''

    def __init__(self, parent=None):
        '''Initialization of the ActivationsView.

        Parameters
        ----------
        parent  :   QWidget
                    The parent argument is sent to the QWidget constructor.
        '''
        super().__init__(parent)

    def initUI(self):
        '''Initialise all UI elements. These are

            * The ``QImageView`` showing the current input image
            * A ``QInputSelector`` to show input controls
            * A ``QNetworkView``, a widget to select a layer in a network
            * A ``QInputInfoBox`` to display information about the input
        '''
        ########################################################################
        #                              User input                              #
        ########################################################################
        self._input_view = QImageView(self)
        # FIXME[layout]
        # keep image view square (TODO: does this make sense for every input?)
        self._input_view.heightForWidth = lambda w: w
        self._input_view.hasHeightForWidth = lambda: True

        # QNetworkInfoBox: a widget to select the input to the network
        # (data array, image directory, webcam, ...)
        # the 'next' button: used to load the next image
        self._input_selector = QInputSelector()

        self._input_info = QInputInfoBox()
        # FIXME[layout]
        self._input_info.setMinimumWidth(300)

        input_layout = QVBoxLayout()
        # FIXME[layout]
        input_layout.setSpacing(0)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.addWidget(self._input_view)
        input_layout.addWidget(self._input_info)
        input_layout.addWidget(self._input_selector)

        input_box = QGroupBox('Input')
        input_box.setLayout(input_layout)
        self._input_box = input_box

        ########################################################################
        #                            Network                                   #
        ########################################################################
        # networkview: a widget to select a network
        self._network_view = QNetworkView()

        self._network_selector = QComboBox()

        self._network_layout = QVBoxLayout()
        self._network_layout.addWidget(self._network_selector)
        self._network_layout.addWidget(self._network_view)

        self._network_box = QGroupBox('Network')
        self._network_box.setLayout(self._network_layout)

    def updateInput(self, data):
        self._input_view.setImage(data)

    def modelChanged(self, model, info):
        current_input = model.get_input(model._current_index)
        self.updateInput(current_input.data)
