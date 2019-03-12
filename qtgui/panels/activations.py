"""
File: activations.py
Author: Petr Byvshev, Ulf Krumnack, Rasmus Diederichsen
Email: rdiederichse@uni-osnabrueck.de
Github: https://github.com/themightyoarfish
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QSplitter

from qtgui.widgets import QActivationView
from qtgui.widgets import QInputSelector, QInputInfoBox, QModelImageView
from qtgui.widgets import QNetworkBox, QNetworkView, QNetworkSelector

from .panel import Panel
from controller import ActivationsController


class ActivationsPanel(Panel):
    """A complex panel containing elements to display activations in
    different layers of a network. The panel offers controls to select
    different input stimuli, and to select a specific network layer.

    Attributes:
    -----------
    _activation_view
    _network_view
    _input_view: QModelImageView
    _input_selector: QInputSelector
    _input_info: QInputInfoBox
    """

    def __init__(self, parent=None):
        """Initialization of the ActivationsPael.

        Parameters
        ----------
        parent  :   QWidget
                    The parent argument is sent to the QWidget constructor.
        """
        super().__init__(parent)
        _network_map = {}
        self.initUI()

    def initUI(self):
        """Initialise all UI elements. These are
            * The ``QActivationView`` showing the unit activations on the left
            * The ``QModelImageView`` showing the current input image
            * A ``QInputSelector`` to show input controls
            * A ``QNetworkView``, a widget to select a layer in a network
            * A ``QInputInfoBox`` to display information about the input
        """

        #
        # Input data
        #

        # QModelImageView: a widget to display the input data
        self._input_view = QModelImageView(self)
        # FIXME[layout]
        # keep image view square (TODO: does this make sense for every input?)
        self._input_view.heightForWidth = lambda w: w
        self._input_view.hasHeightForWidth = lambda: True

        # QInputSelector: a widget to select the input to the network
        # (data array, image directory, webcam, ...)
        # the 'next' button: used to load the next image
        self._input_selector = QInputSelector()

        # FIXME[hack]
        self._input_info = QInputInfoBox(imageView=self._input_view)
        self._input_view.modeChange.connect(self._input_info.onModeChange)

        # FIXME[layout]
        self._input_info.setMinimumWidth(200)

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

        #
        # Network
        #

        # QNetworkSelector: a widget to select a network
        self._network_selector = QNetworkSelector()

        # QNetworkView: a widget to select a network
        self._network_view = QNetworkView()

        # Now put the network stuff into one group
        network_layout = QVBoxLayout()
        network_layout.addWidget(self._network_selector)
        network_layout.addWidget(self._network_view)

        self._network_box = QGroupBox('Network')
        self._network_box.setLayout(network_layout)

        #
        # Activations
        #

        # ActivationView: a canvas to display a layer activation
        self._activation_view = QActivationView()
        # FIXME[layout]
        self._activation_view.setMinimumWidth(200)
        self._activation_view.resize(400, self._activation_view.height())

        activation_layout = QVBoxLayout()
        activation_layout.addWidget(self._activation_view)

        activation_box = QGroupBox('Activation')
        activation_box.setLayout(activation_layout)

        #
        # Attach widgets to window
        #
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(activation_box)
        splitter.addWidget(self._input_box)
        layout = QHBoxLayout()
        layout.addWidget(splitter)
        layout.addWidget(self._network_box)
        self.setLayout(layout)
