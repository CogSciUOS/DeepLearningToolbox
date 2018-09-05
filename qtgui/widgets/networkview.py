from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout

from network import Network, Layer, Conv2D
from observer import Observer


class QNetworkView(QWidget, Observer):
    '''A simple widget to display information on a network and select a
    layer.

    Attributes
    ----------
    _controller :   controller.InputController
                    Controller for this UI element
    _current_selected   :   int
                            Layer id of the currently selected layer
    '''

    _controller:   'controller.InputController' = None
    _current_selected:  int = None

    # FIXME[hack]: make a nicer solution and then remove this!
    _label_input = None
    _label_output = None
    
    def __init__(self, parent=None):
        '''Initialization of the QNetworkView.

        Parameters
        ----------
        parent : QWidget
            The parent argument is sent to the QWidget constructor.
        '''
        super().__init__(parent)
        self.initUI()

    def initUI(self) -> None:
        '''Initialize the user interface.'''
        self._info_layout = QVBoxLayout()
        self._layer_layout = QVBoxLayout()
        self._network_info = QNetworkInfoBox(self)
        self._layer_buttons = []
        self._network_info.setMinimumWidth(300)
        self._info_layout.addWidget(self._network_info)
        layout = QVBoxLayout()
        layout.addLayout(self._layer_layout)
        layout.addLayout(self._info_layout)
        self.setLayout(layout)

    def setController(self, controller):
        super().setController(controller)
        self._network_info.setController(self._controller)

    def layerClicked(self):
        '''Callback for clicking one of the layer buttons.'''
        self._controller.onLayerSelected(self.sender().text())

    def modelChanged(self, model, info):

        if info.network_changed:
            ###############################
            #  Respond to network change  #
            ###############################
            layout = self._layer_layout
            network = model._network
            self._current_selected = None
            if network:
                # remove the old network buttons
                # FIXME[todo]: (still not sure what is the correct way to do:
                # widget.deleteLater(), widget.close(), widget.setParent(None), or
                # layout.removeWidget(widget) + widget.setParent(None))
                for button in self._layer_buttons:
                    button.deleteLater()
                self._layer_buttons = []
                if self._label_input is not None:
                    self._label_input.deleteLater()
                    self._label_input = None
                if self._label_output is not None:
                    self._label_output.deleteLater()
                    self._label_output = None                   

                self._label_input = QLabel("input = " + str(network.get_input_shape(False)))
                layout.addWidget(self._label_input)

                # a column of buttons to select the network layer
                for name in network.layer_dict.keys():
                    button = QPushButton(name, self)
                    self._layer_buttons.append(button)
                    layout.addWidget(button)
                    button.resize(button.sizeHint())
                    button.clicked.connect(self.layerClicked)

                self._label_output = QLabel("output = " + str(network.get_output_shape(False)))
                layout.addWidget(self._label_output)

                # choose the active layer for the new network
                # self.setActiveLayer(0)
                self._controller.onLayerSelected(None)

        if info.network_changed or info.layer_changed:
            #############################
            #  Respond to layer change  #
            #############################
            layer = model._layer
            if layer:
                # unmark the previously active layer
                if self._current_selected is not None:
                    oldItem = self._layer_buttons[self._current_selected]
                    oldItem.setStyleSheet('')

                self._current_selected = model.idForLayer(layer)
                if self._current_selected is not None:
                    # and mark the new layer
                    newItem = self._layer_buttons[self._current_selected]
                    if newItem is not None:
                        newItem.setStyleSheet('background-color: red')

        self.update()

####################################################################################################
#                                 QNetworkInfoBox class definition                                 #
####################################################################################################
from PyQt5.QtWidgets import QLabel
from collections import OrderedDict
from network import Network


class QNetworkInfoBox(QLabel, Observer):
    '''
    .. class:: QNetworkInfoBox

    A simple widget to display information on a network and a selected
    layer'''

    def __init__(self, parent=None):
        '''Create a new :py:class:``QNetworkInfoBox``'''
        super().__init__(parent)
        self.setWordWrap(True)

    def modelChanged(self, model, info):
        network = model._network
        network_text = ''
        layer_text = ''
        if info.network_changed:
            if not network:
                return
            ############################################################################################
            #                                  Set Network info text                                   #
            ############################################################################################
            network_text = '<b>Network info:</b> '
            network_name = type(network).__name__
            network_text += 'FIXME[todo]: obtain network information ...'
            network_text += f'<br>\n<b>class:</b> {network_name}'

        if info.network_changed or info.layer_changed:
            ############################################################################################
            #                                   Set Layer info text                                    #
            ############################################################################################
            if not network:
                return
            if model._layer:
                layer = network.layer_dict[model._layer]
                layer_text = '<br>\n<br>\n'
                layer_text += '<b>Layer info:</b><br>\n'

                layer_info = '\n'.join('<b>{}</b>: {}<br>'.format(key, val) for key, val in
                                       layer.info.items())
                layer_text += layer_info

        self.setText(network_text + layer_text)
