"""
File: maximization.py
Author: Ulf Krumnack, Antonia Hain
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QSplitter)

from qtgui.widgets.maximization import (QMaximizationConfig,
                                        QMaximizationControls,
                                        QMaximizationDisplay)
from .panel import Panel

import numpy as np
from tools.am import Config, Engine
from controller import MaximizationController, BaseController
from observer import Observer

class MaximizationPanel(Panel):
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

    def __init__(self, parent=None):
        """Initialization of the ActivationsPael.

        Parameters
        ----------
        parent  :   QWidget
                    The parent argument is sent to the QWidget constructor.
        """
        super().__init__(parent)
        self._initUI()

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

        self._layoutComponents()


    def _layoutComponents(self):
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

    def setController(self, controller: BaseController,
                      observerType: type=Observer):
        super().setController(controller, observerType)
        if isinstance(controller, MaximizationController):
            self._maximization_display.connectToConfigViews(controller.onConfigViewSelected)
