from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout

from network import Network, Layer, Conv2D
from model import Model, ModelObserver, ModelChange


from .classesview import QClassesView

from PyQt5.QtWidgets import QWidget, QLabel, QGridLayout


class QNetworkView(QWidget, ModelObserver):
    '''A simple widget to display information on a network and select a
    layer.

    Attributes
    ----------
    _controller :   controller.ActivationsController
                    Controller for this UI element
    _current_selected   :   int
                            Layer id of the currently selected layer
    '''

    _current_selected:  int = None

    # FIXME[hack]: make a nicer solution and then remove this!
    _label_input = None
    _label_output = None

    _classes_view = None
    
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
        info_layout = QVBoxLayout()
        self._network_info = QNetworkBox(self)
        self._network_info.setMinimumWidth(300)
        info_layout.addWidget(self._network_info)
        
        self._layer_layout = QVBoxLayout()
        self._layer_buttons = []
        
        self._classes_view = QClassesView(self)

        layout = QVBoxLayout()
        layout.addLayout(self._layer_layout)
        layout.addLayout(info_layout)
        layout.addStretch(1)
        layout.addWidget(self._classes_view)
        self.setLayout(layout)

    def setController(self, controller):
        super().setController(controller)
        self._network_info.setController(controller)
        self._classes_view.setController(controller)


    def _layerButtonClicked(self):
        '''Callback for clicking one of the layer buttons.'''
        if (self._current_selected and
            self._layer_buttons[self._current_selected] == self.sender()):
            layer = None
        else:
            layer = self.sender().text()
        self._controller.onLayerSelected(layer)

    def modelChanged(self, model: Model, info: ModelChange) -> None:
        """The QNetworkView is interested in network related changes, i.e.
        changes of the network itself or the current layer.
        
        """

        if info.network_changed:
            #
            # Respond to network change
            #
            layout = self._layer_layout
            network = model._network
            self._current_selected = None
            if network:
                #
                # remove the old network buttons
                #
                
                # FIXME[todo]: (still not sure what is the correct way to do:
                # widget.deleteLater(), widget.close(),
                # widget.setParent(None), or
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

                #
                # a column of buttons to select the network layer
                #
                for name in network.layer_dict.keys():
                    button = QPushButton(name, self)
                    self._layer_buttons.append(button)
                    layout.addWidget(button)
                    button.resize(button.sizeHint())
                    button.clicked.connect(self._layerButtonClicked)

                self._label_output = QLabel("output = " + str(network.get_output_shape(False)))
                layout.addWidget(self._label_output)

                # choose the active layer for the new network
                # self.setActiveLayer(0)
                self._controller.onLayerSelected(None)

        if info.network_changed or info.layer_changed:
            #
            # Respond to layer change
            #
            layer_id = model.layer_id
            try:
                layer_index = model.idForLayer(layer_id)
            except ValueError as error:
                print(f"ERROR: {self.__class__.__name__}.modelChanged(): "
                      "{error}")
                layer_index = None
            if layer_index != self._current_selected:
                
                # unmark the previously active layer
                if self._current_selected is not None:
                    oldItem = self._layer_buttons[self._current_selected]
                    oldItem.setStyleSheet('')

                self._current_selected = layer_index
                if self._current_selected is not None:
                    # and mark the new layer
                    print(f"debug: self._current_selected:{self._current_selected}")
                    print(f"debug: self._layer_buttons:{len(self._layer_buttons)}")
                    newItem = self._layer_buttons[self._current_selected]
                    if newItem is not None:
                        newItem.setStyleSheet('background-color: red')

        self.update()


###############################################################################
###                           QNetworkBox                                   ###
###############################################################################


from PyQt5.QtWidgets import QLabel
from collections import OrderedDict
from network import Network


class QNetworkBox(QLabel, ModelObserver):
    '''
    .. class:: QNetworkBox

    A simple widget to display information on a network and a selected
    layer'''

    def __init__(self, parent=None):
        '''Create a new :py:class:``QNetworkInfoBox``'''
        super().__init__(parent)
        self._network_text = ''
        self._layer_text = ''
        self.setWordWrap(True)

    def modelChanged(self, model: Model, info: ModelChange) -> None:
        network_text = '<b>Network Info Box</b><br>'
        layer_text = ''
        
        #
        # Set Network info text
        #
        network = model._network
        if info.network_changed:
            network_text += '<b>Network info:</b> '
            if network is not None:
                network_name = type(network).__name__
                network_text += 'FIXME[todo]: obtain network information ...'
                network_text += ('<br>\n<b>class:</b> '
                                 f'{network.__class__.__name__}')
                network_text += f'<br>\n<b>name:</b> {network}'
            else:
                network_text += "No network"
            self._network_text = network_text

        #
        # Set Layer info text
        #
        layer_id = model._layer
        if info.layer_changed:
            layer_text += '<br>\n<br>\n'
            layer_text += '<b>Layer info:</b><br>\n'
            if layer_id:
                layer = network.layer_dict[layer_id]
                layer_info = '\n'.join('<b>{}</b>: {}<br>'.format(key, val)
                                       for key, val in layer.info.items())
                layer_text += layer_info
            else:
                layer_text += "No layer selected"
            self._layer_text = layer_text

        self.setText(self._network_text + self._layer_text)


###############################################################################
###                           QNetworkSelector                              ###
###############################################################################

from PyQt5.QtWidgets import QComboBox
from typing import Dict, Iterable


class QNetworkSelector(QComboBox, ModelObserver):
    """
    Attributes
    ----------
    _network_map : Dict[str,Network]
        A dictionary mapping the strings displayed in the network selector
        dropdown to actual network objects.

    """
    def __init__(self, parent=None) -> None:
        '''Initialization of the QNetworkView.

        Parameters
        ----------
        parent : QWidget
            The parent argument is sent to the QComboBox constructor.
        '''
        super().__init__(parent)
        self._network_map = {}

    def setController(self, controller) -> None:
        super().setController(controller)
        self.activated[str].connect(controller.onNetworkSelected)

    def modelChanged(self, model: Model, info: ModelChange) -> None:
        if info.network_changed:
            self._update_networks(model.networks)

            network = model.network
            if network is not None:
                new_index = self.findText(network.get_id())
                self.setCurrentIndex(new_index)

    def _update_networks(self, networks: Iterable[Network]) -> None:
        self.clear()
        for network in networks:
            self.addItem(network.get_id(), network)

    @property
    def network(self) -> Network:
        return self.currentData()
