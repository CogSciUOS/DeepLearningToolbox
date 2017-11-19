'''
File: activations.py
Author: Petr Byvshev, Ulf Krumnack, Rasmus Diederichsen
Email: rdiederichse@uni-osnabrueck.de
Github: https://github.com/themightyoarfish
'''

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QComboBox
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QSplitter

from qtgui.widgets import QActivationView
from qtgui.widgets import QInputSelector, QInputInfoBox, QImageView
from qtgui.widgets import QNetworkView, QNetworkInfoBox
from .panel import Panel

import numpy as np
from scipy.misc import imresize

# FIXME[todo]: rearrange the layer selection on network change!
# FIXME[todo]: add docstrings!


class ActivationsPanel(Panel):
    '''A complex panel containing elements to display activations in
    different layers of a network. The panel offers controls to select
    different input stimuli, and to select a specific network layer.

    Attributes
    ----------
    _network :   network.network.Network
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

    def on_activation_view_selected(self, callback):
        '''Connect a callback to the activation view.

        Parameters
        ----------
        callback    :   function
                        Function to call when activation view gets selected
        '''
        self._activation_view.on_unit_selected(callback)

    def __init__(self, parent=None):
        '''Initialization of the ActivationsView.

        Parameters
        ----------
        parent  :   QWidget
                    The parent argument is sent to the QWidget constructor.
        '''
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
        self.on_activation_view_selected(self.setUnit)

        # FIXME[layout]
        self._activation_view.setMinimumWidth(300)
        self._activation_view.resize(600, self._activation_view.height())

        activation_layout = QVBoxLayout()
        activation_layout.addWidget(self._activation_view)

        activation_box = QGroupBox('Activation')
        activation_box.setLayout(activation_layout)
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



    def setLayer(self, layer=None):
        super().setLayer(layer=layer)
        # We update the activation on every invocation, no matter if
        # the selected layer changed. This allows for some dynamics
        # if layers like dropout are involved. However, if such
        # layers do not exist, it will only waste computing power ...
        self.updateActivation()

    def updateActivation(self):
        '''Update this panel in response to a new activation values.
        New activation values can be caused by the selection of another
        layer in the network, but also by the change of the input image.
        '''
        if not (self._network and self._layer and self._data is not None):
            activations = None
        else:
            activations = self._network.get_activations(
                [self._layer], self._data)[0]
            # ff or conv layer w/ batch dim
            if activations.ndim in {2, 4}:
                assert activations.shape[0] == 1, 'Attempting to visualiase batch.'
                activations = np.squeeze(activations, axis=0)

        self._activation_view.setActivation(activations)

    def setUnit(self, unit: int=None):
        '''Change the currently visualised unit. This should be called when the
        user clicks on a unit in the ActivationView. The activation mask will be
        nearest-neighbour-interpolated to the shape of the input data.

        Parameters
        ----------
        unit    :   int
                    Index of the unit in the layer (0-based)
        '''
        activation_mask = self._activation_view.getUnitActivation(unit)
        if activation_mask is not None:
            if activation_mask.shape == self._data.shape:
                activation_mask = imresize(activation_mask, self._data.shape,
                                           interp='nearest')
            self._input_view.setActivationMask(activation_mask)

    def setInputData(self, raw: np.ndarray=None, fitted: np.ndarray=None,
                     description: str=None):
        super().setInputData(raw, fitted, description)
        self.updateActivation()
