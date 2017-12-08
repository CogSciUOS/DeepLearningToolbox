'''
File: activations.py
Author: Petr Byvshev, Ulf Krumnack, Rasmus Diederichsen
Email: rdiederichse@uni-osnabrueck.de
Github: https://github.com/themightyoarfish
'''

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QSplitter

from qtgui.widgets import QActivationView
from qtgui.widgets import QNetworkInfoBox
from .panel import Panel
from controller import ActivationsController

import numpy as np

# FIXME[todo]: rearrange the layer selection on network change!


class ActivationsPanel(Panel):
    '''A complex panel containing elements to display activations in
    different layers of a network. The panel offers controls to select
    different input stimuli, and to select a specific network layer.

    Attributes
    ----------
    _network :  network.network.Network
                The current model to visualise. This network is used
                for all computations in this ActivationsPanel. Can be None if no
                network is selected.
    _layer   :   str
                Layer id of the currently selected layer. Activations are shown
                for the respective layer. The layer can be None if no layer is
                selected.
    _data   :   np.ndarray
                The current input data. This data should always match the input
                size of the current network. May be None if no input data is
                available.
    '''

    def __init__(self, model, parent=None):
        '''Initialization of the ActivationsView.

        Parameters
        ----------
        parent  :   QWidget
                    The parent argument is sent to the QWidget constructor.
        model   :   model.Model
                    The backing model. Communication will be handled by a
                    controller.
        '''
        self._controller = ActivationsController(model)
        super().__init__(parent)

    def initUI(self):
        '''Add additional UI elements

            * The ``QActivationView`` showing the unit activations on the left

        '''
        super().initUI()
        ########################################################################
        #                             Activations                              #
        ########################################################################
        # ActivationView: a canvas to display a layer activation
        self._activation_view = QActivationView()
        self._activation_view.setController(self._controller)

        # FIXME[layout]
        self._activation_view.setMinimumWidth(300)
        self._activation_view.resize(600, self._activation_view.height())

        activation_layout = QVBoxLayout()
        activation_layout.addWidget(self._activation_view)

        activation_box = QGroupBox('Activation')
        activation_box.setLayout(activation_layout)

        ########################################################################
        #                            Network stuff                             #
        ########################################################################
        self._network_view.selected.connect(self.setLayer)

        # network info: a widget to select a layer
        self._network_info = QNetworkInfoBox()
        # FIXME[layout]
        self._network_info.setMinimumWidth(300)
        self._network_layout.addWidget(self._network_info)

        ########################################################################
        #                       Attach widgets to window                       #
        ########################################################################
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(activation_box)
        splitter.addWidget(self._input_box)
        layout = QHBoxLayout()
        layout.addWidget(splitter)
        layout.addWidget(self._network_box)
        self.setLayout(layout)

    def updateActivation(self, activations):
        '''Update this panel in response to new activation values.
        New activation values can be caused by the selection of another
        layer in the network, but also by the change of the input image.
        '''
        self._activation_view.setActivation(activations)


    def setInputData(self, raw: np.ndarray=None, fitted: np.ndarray=None,
                     description: str=None):
        super().setInputData(raw, fitted, description)
        self.updateActivation()
