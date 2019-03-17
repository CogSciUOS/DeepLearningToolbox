"""
File: maximization.py
Author: Ulf Krumnack, Antonia Hain
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
"""

from toolbox import Toolbox, Controller as ToolboxController
from network import Network, Controller as NetworkController
from tools.am import (Engine as MaximizationEngine,
                      Controller as MaximizationController)

from .panel import Panel
from ..utils import QObserver
from ..widgets.maximization import (QMaximizationConfig,
                                    QMaximizationControls,
                                    QMaximizationDisplay)

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QSplitter)


class MaximizationPanel(Panel, QObserver, MaximizationEngine.Observer):
    """A panel containing elements to configure and run activation
    maximization and to display results.

    * MaximizationControls: 

    * 

    Attributes
    ----------
    _maximization_controls: QMaximizationControls
        A set of controls to start and stop the maximization process
        and to display intermediate results.
    _maximization_config_view: QMaximizationConfig

    """
    _maximization_controller: MaximizationController = None
    _maximization_controls: QMaximizationControls = None
    _maximization_display: QMaximizationDisplay = None
    _maximization_config_view: QMaximizationConfig = None

    def __init__(self, toolbox: ToolboxController=None,
                 network: NetworkController=None, **kwargs):
        """Initialization of the ActivationsPael.

        Parameters
        ----------
        parent  :   QWidget
                    The parent argument is sent to the QWidget constructor.
        """
        super().__init__(**kwargs)
        self._initUI()
        self._layoutUI()
        self.setToolboxController(toolbox)
        self.setNetworkController(network)

    def _initUI(self):
        """Add the UI elements

            * The ``QActivationView`` showing the unit activations on the left

        """
        #
        # Controls
        #
        self._maximization_controls = QMaximizationControls()

        #
        # Display
        #
        self._maximization_display = QMaximizationDisplay()
        self._maximization_controls.display = self._maximization_display

        #
        # Configuration
        #

        # QMaximizationConfig: a widget to configure the maximization process
        self._maximization_config_view = QMaximizationConfig()

    def _layoutUI(self):
        """Layout the UI elements.

            * The ``QActivationView`` showing the unit activations on the left

        """

        #
        # Set dimensions for the QMaximizationConfig
        #
        self._maximization_config_view.setMinimumWidth(200)
        height = self._maximization_config_view.height()
        self._maximization_config_view.resize(400, height)


        config_layout = QVBoxLayout()
        config_layout.addWidget(self._maximization_config_view)

        config_box = QGroupBox('Configuration')
        config_box.setLayout(config_layout)

        #
        # Panel
        #

        layout = QVBoxLayout()
        layout.addWidget(self._maximization_controls)
        layout.addWidget(self._maximization_display)
        layout.addStretch(1)
        layout.addWidget(config_box)

        self.setLayout(layout)

    def setToolboxController(self, toolbox: ToolboxController) -> None:
        self._maximization_config_view.setToolboxController(toolbox)

    def setNetworkController(self, network: NetworkController) -> None:
        self._maximization_config_view.setNetworkController(network)
        self._maximization_controls.setNetworkController(network)

    def setMaximizationController(self, maximization: MaximizationController
                                  ) -> None:
        interests=MaximizationEngine.Change('observable_changed')
        self._exchangeView('_maximization_controller', maximization,
                           interests=interests)
        self._maximization_controls.setMaximizationController(maximization)
        # FIXME[hack]: we also need to disconnect ...
        if maximization is not None:
            self._maximization_display.\
                connectToConfigViews(maximization.onConfigViewSelected)

    def engineChanged(self, engine: MaximizationEngine,
                      change: MaximizationEngine.Change) -> None:
        if change.observable_changed:
            self._maximization_config_view.setConfig(engine.config)
