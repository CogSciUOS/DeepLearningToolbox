'''
File: maximization.py
Author: Ulf Krumnack, Antonia Hain
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
'''

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QSplitter

from qtgui.widgets.maximization import QMaximizationConfig
from .panel import Panel

import numpy as np


class MaximizationPanel(Panel):
    '''A complex panel containing elements to configure the
    activation maximization of individual unites.

    Attributes
    ----------
    _network_map    :   dict A dictionary mapping the strings displayed in the network selector
                        dropdown to actual network objects.
    '''

    _network_map    :   dict = {}

    def __init__(self, parent=None):
        '''Initialization of the ActivationsPael.

        Parameters
        ----------
        parent  :   QWidget
                    The parent argument is sent to the QWidget constructor.
        '''
        super().__init__(parent)
        self._initUI()

    def _initUI(self):
        '''Add additional UI elements

            * The ``QActivationView`` showing the unit activations on the left

        '''

        #
        # Configuration
        #

        # QMaximizationConfig: a widget to configure the maximization process
        self._maximization_config_view = QMaximizationConfig()

        # FIXME[layout]
        self._maximization_config_view.setMinimumWidth(200)
        self._maximization_config_view.resize(400, self._maximization_config_view.height())

        config_layout = QVBoxLayout()
        config_layout.addWidget(self._maximization_config_view)

        config_box = QGroupBox('Configuration')
        config_box.setLayout(config_layout)

        layout = QHBoxLayout()
        layout.addWidget(config_box)
        self.setLayout(layout)
