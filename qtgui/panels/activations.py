"""
File: activations.py
Author: Petr Byvshev, Ulf Krumnack, Rasmus Diederichsen
Email: rdiederichse@uni-osnabrueck.de
Github: https://github.com/themightyoarfish
"""
from toolbox import Controller as ToolboxController
from network import Controller as NetworkController
from datasources import Controller as DatasourceController
from controller import ActivationsController
from tools.activation import (Engine as ActivationEngine,
                              View as ActivationView,
                              Controller as ActivationController)

from .panel import Panel
from ..utils import QObserver
from ..widgets.activationview import QActivationView
from ..widgets.inputselector import QInputNavigator, QInputInfoBox
from ..widgets.imageview import QModelImageView
from ..widgets.networkview import QNetworkView, QNetworkSelector
from ..widgets.classesview import QClassesView

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QSplitter


class ActivationsPanel(Panel, QObserver, ActivationEngine.Observer):
    """A complex panel containing elements to display activations in
    different layers of a network. The panel offers controls to select
    different input stimuli, and to select a specific network layer.

    Attributes:
    -----------
    _activationView: QActivationView
        A widget displaying the activations for a selected layer.
    _network_view: QNetworkView
        A widget for different network related activities: selecting
        a network from a list of networks, selecting a layer for a network,
        displaying network information and network output.
    _inputView: QModelImageView
    _inputInfoBox: QInputInfoBox
    _inputNavigator: QInputNavigator
    _classesView: QClassesView = None
    """
    _activation: ActivationView = None

    _inputView: QModelImageView = None
    _inputInfoBox: QInputInfoBox = None
    _inputNavigator: QInputNavigator = None
    _network_view: QNetworkView = None
    _classesView: QClassesView = None
    _activationView: QActivationView = None

    def __init__(self, toolbox: ToolboxController=None,
                 network: NetworkController=None,
                 datasource: DatasourceController=None,
                 activations: ActivationsController=None,
                 **kwargs) -> None:
        """Initialization of the ActivationsPael.

        Parameters
        ----------
        toolbox: ToolboxController
        network: NetworkController
        parent: QWidget
            The parent argument is sent to the QWidget constructor.
        """
        super().__init__(**kwargs)
        _network_map = {}
        self._initUI()
        self._layoutUI()
        self.setToolboxController(toolbox)
        self.setNetworkController(network)
        self.setActivationsController(activations)
        self.setDatasourceController(datasource)

    def setActivationsController(self, activations: ActivationsController
                                 ) -> None:
        interests = ActivationEngine.Change('network_changed')
        self._exchangeView('_activation', activations, interests=interests)
        self._networkView.setActivationsController(activations)
        self._activationView.setActivationController(activations)
        self._classesView.setActivationView(activations)

    def setToolboxController(self, toolbox: ToolboxController) -> None:
        self._networkSelector.setToolboxView(toolbox)
        self._inputView.setToolboxView(toolbox)
        self._inputInfoBox.setToolboxView(toolbox)
            
    def setNetworkController(self, network: NetworkController) -> None:
        self._networkSelector.setNetworkView(network)
        self._networkView.setNetworkView(network)

    def setDatasourceController(self,
                                datasource: DatasourceController) -> None:
        self._inputNavigator.setDatasourceController(datasource)



    def _initUI(self):
        """Initialise all UI elements. These are
            * The ``QActivationView`` showing the unit activations on the left
            * The ``QModelImageView`` showing the current input image
            * A ``QInputNavigator`` to show input controls
            * A ``QNetworkView``, a widget to select a layer in a network
            * A ``QInputInfoBox`` to display information about the input
        """

        #
        # Input data
        #

        # QModelImageView: a widget to display the input data
        self._inputView = QModelImageView()

        # QInputSelector: a widget to select the input
        # (a datasource navigator)
        self._inputNavigator = QInputNavigator()

        self._inputInfoBox = QInputInfoBox()
        self._inputView.modeChanged.connect(self._inputInfoBox.onModeChanged)

        #
        # Network
        #

        # QNetworkSelector: a widget to select a network
        self._networkSelector = QNetworkSelector()

        # QNetworkView: a widget to select a network layer
        self._networkView = QNetworkView()

        #
        # Activations
        #

        # ActivationView: a canvas to display a layer activation
        self._activationView = QActivationView()

        # QClassesView: display classification results
        self._classesView = QClassesView()
        self._classesViewBox = QGroupBox('Classification')
        self._classesViewBox.setCheckable(True)
        classesViewLayout = QVBoxLayout()
        classesViewLayout.addWidget(self._classesView)
        self._classesViewBox.setLayout(classesViewLayout)

    def _layoutUI(self):
        #
        # Input data
        #
        inputLayout = QVBoxLayout()

        # FIXME[layout]
        # keep image view square (TODO: does this make sense for every input?)
        self._inputView.heightForWidth = lambda w: w
        self._inputView.hasHeightForWidth = lambda: True
        # FIXME[hack]
        self._inputView.setMaximumSize(500,500)
        
        # FIXME[layout]
        inputLayout.setSpacing(0)
        inputLayout.setContentsMargins(0, 0, 0, 0)
        inputLayout.addWidget(self._inputView)
        inputLayout.addWidget(self._inputInfoBox)
        # FIXME[layout]
        self._inputInfoBox.setMinimumWidth(200)
        inputLayout.addWidget(self._inputNavigator)

        inputBox = QGroupBox('Input')
        inputBox.setLayout(inputLayout)

        #
        # Network
        #
        networkBox = QGroupBox('Network')
        networkLayout = QVBoxLayout()
        networkLayout.addWidget(self._networkSelector)
        networkLayout.addWidget(self._networkView)
        networkBox.setLayout(networkLayout)

        #
        # classes
        #
        rightLayout = QVBoxLayout()
        rightLayout.addWidget(networkBox)
        rightLayout.addWidget(self._classesViewBox)

        #
        # Activations
        #
        activationLayout = QVBoxLayout()
        activationLayout.addWidget(self._activationView)
        # FIXME[layout]
        self._activationView.setMinimumWidth(200)
        self._activationView.resize(400, self._activationView.height())

        activationBox = QGroupBox('Activation')
        activationBox.setLayout(activationLayout)

        #
        # Attach widgets to window
        #
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(activationBox)
        splitter.addWidget(inputBox)
        layout = QHBoxLayout()
        layout.addWidget(splitter)
        layout.addLayout(rightLayout)
        self.setLayout(layout)

    def activation_changed(self, activation: ActivationEngine,
                           info: ActivationEngine.Change) -> None:
        """The QClassesView is only interested if the classification result
        changes.
        """
        if info.network_changed:
            network = activation.network
            enabled = network and network.is_classifier()
            self._classesViewBox.setEnabled(enabled)
