from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout

from network import Network, Layer, Conv2D

from observer import Observer
from controller import NetworkController


class QNetworkView(QWidget, Observer):
    '''A simple widget to display information on a network and select a
    layer.

    Attributes
    ----------
    _network :  network.Network
                The network currently displayed in this ``QNetworkView``.

    _active :   str  or int
                The identifier of the currently selected layer.  The actual type
                of the identifier depends on the network implementation. The
                value None indicates that no layer is currently selectd.
    '''

    _network:   Network = None

    _active:   str = None

    _controller:   NetworkController = None

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
        self.setLayout(QVBoxLayout())

    def layerClicked(self):
        '''Callback for clicking one of the layer buttons.'''
        self._controller.on_layer_selected(self.sender().text())

    def modelChanged(self, model):
        ############################################################################################
        #                                 Respond to layer change                                  #
        ############################################################################################

        layer = model._layer
        layer_id = model.id_for_layer(layer)
        # unmark the previously active layer
        oldItem = self.layout().itemAt(layer_id)
        if oldItem is not None:
            oldItem.widget().setStyleSheet('')

        # and mark the new layer
        newItem = self.layout().itemAt(layer_id)
        if newItem is not None:
            newItem.widget().setStyleSheet('background-color: red')

        ############################################################################################
        #                                Respond to network change                                 #
        ############################################################################################
        layout = self.layout()
        network = model._network

        # remove the old network buttons
        # FIXME[todo]: (still not sure what is the correct way to do:
        # widget.deleteLater(), widget.close(), widget.setParent(None), or
        # layout.removeWidget(widget) + widget.setParent(None))
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if network:
            # a column of buttons to select the network layer
            for name in network.layer_dict.keys():
                button = QPushButton(name, self)
                layout.addWidget(button)
                button.resize(button.sizeHint())
                button.clicked.connect(self.layerClicked)

            # choose the active layer for the new network
            # self.setActiveLayer(0)
            self._controller.setLayer(None)

        self.update()

####################################################################################################
#                                 QNetworkInfoBox class definition                                 #
####################################################################################################
from PyQt5.QtWidgets import QLabel
from collections import OrderedDict
from network import Network


class QNetworkInfoBox(QLabel, Observer):
    '''A simple widget to display information on a network and a selected
    layer.'''

    def __init__(self, parent=None):
        '''Create a new ``QNetworkInfoBox``'''
        super().__init__(parent)
        self.setWordWrap(True)

    def modelChanged(self, model):
        network = model._network
        layer = network.layer_dict[model._layer]
        ############################################################################################
        #                                  Set Network info text                                   #
        ############################################################################################

        network_text = '<b>Network info:</b> '
        if not network:
            network_text += 'No network selected'
        else:
            network_name = type(network).__name__
            network_text += 'FIXME[todo]: obtain network information ...'
            network_text += f'<br>\n<b>class:</b> {network_name}'

        ############################################################################################
        #                                   Set Layer info text                                    #
        ############################################################################################
        if not network:
            layer_text = ''
        else:
            layer_text = '<br>\n<br>\n'
            layer_text += '<b>Layer info:</b><br>\n'

            if not layer:
                layer_text += 'No layer selected'
            else:
                layer_info = '\n'.join('<b>{}</b>: {}<br>'.format(key, val) for key, val in
                                       layer.info.items())
                layer_text += layer_info

        self.setText(network_text + layer_text)
