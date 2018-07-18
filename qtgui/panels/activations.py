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
import controller
from observer import Observer

import numpy as np


class ActivationsPanel(Panel, Observer):
    '''A complex panel containing elements to display activations in
    different layers of a network. The panel offers controls to select
    different input stimuli, and to select a specific network layer.

    Attributes
    ----------
    _network_map    :   dict A dictionary mapping the strings displayed in the network selector
                        dropdown to actual network objects.
    '''

    _network_map    :   dict = {}

    def __init__(self, model, parent=None):
        '''Initialization of the ActivationsPael.

        Parameters
        ----------
        parent  :   QWidget
                    The parent argument is sent to the QWidget constructor.
        model   :   model.Model
                    The backing model. Communication will be handled by a
                    controller.
        '''
        from controller import ActivationsController
        super().__init__(parent)
        self.initUI()
        self.setController(ActivationsController(model))
        model


    def setController(self, controller):
        super().setController(controller)
        controllable_widgets = [self._activation_view,
                   self._network_view,
                   self._input_view,
                   self._input_selector,
                   self._input_info,
                   self._network_view]
        for widget in controllable_widgets:
            widget.setController(controller)

        # What is this for? Since the Model is initialised with the network, the first call to
        # setNetwork must somehow communicate that an update of all observers is desired despite
        # the fact that the new network is identical to the old one. Another approach would be to
        # make the network an optional Model initialiser param, but this requires some further
        # tweaks. So here I just capture a local variable in a closure which then gets flipped on
        # first call. I need a closure anyway for the name -> network map, so might as well put this
        # in as well.
        first_call = True
        def select_net(name):
            '''closure for _network_map and first_call'''
            nonlocal first_call # 'global' does not work, dunno why.
            controller.onNetworkSelected(self._network_map[name], first_call)
            first_call = False

        self._network_selector.activated[str].connect(select_net)
        for key, network in controller._model._networks.items():
            display_name = str(network.__class__)
            self._network_selector.addItem(display_name)
            self._network_map[display_name] = network

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

        # FIXME[layout]
        self._activation_view.setMinimumWidth(200)
        self._activation_view.resize(400, self._activation_view.height())

        activation_layout = QVBoxLayout()
        activation_layout.addWidget(self._activation_view)

        activation_box = QGroupBox('Activation')
        activation_box.setLayout(activation_layout)

        ########################################################################
        #                            Network stuff                             #
        ########################################################################

        # network info: a widget to select a layer
        self._network_info = QNetworkInfoBox()
        # FIXME[layout]
        # self._network_info.setMinimumWidth(300)
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
