from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout

from network import Network, Layer, Conv2D


class QNetworkView(QWidget):
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

    selected    :   pyqtSignal(object)
                    A signal that will be emitted everytime a layer is selected
                    in this view. The signal will be emitted multiple times if
                    the same layer is pressed multiple times in a row, as some
                    layers (like dropout), may change on every activation. The
                    argument provided by the signal will be the layer_id, i.e.
                    the identifier of the layer in the current network.
    '''

    _network    :   Network = None

    _active     :   str = None

    selected = pyqtSignal(object)


    def __init__(self, parent=None):
        '''Initialization of the QNetworkView.

        Parameters
        ----------
        parent : QWidget
            The parent argument is sent to the QWidget constructor.
        '''
        super().__init__(parent)
        self.initUI()
        self.setNetwork()


    def initUI(self) -> None:
        '''Initialize the user interface.'''
        self.setLayout(QVBoxLayout())


    def setNetwork(self, network=None):
        '''Set the network to display in this widget

        Parameters
        ----------
        network :   network.Network
                    Network to display
        '''
        self._network = network
        self._active = None

        layout = self.layout()

        ## remove the old network buttons
        # FIXME[todo]: (still not sure what is the correct way to do:
        # widget.deleteLater(), widget.close(), widget.setParent(None), or
        # layout.removeWidget(widget) + widget.setParent(None))
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

        if self._network is not None:
            ## a column of buttons to select the network layer
            for name in self._network.layer_dict.keys():
                button = QPushButton(name, self)
                layout.addWidget(button)
                button.resize(button.sizeHint())
                button.clicked.connect(self.layerClicked)

            ## choose the active layer for the new network
            #self.setActiveLayer(0)
            self.setActiveLayer(None)
        else:
            self.setActiveLayer(None)

        self.update()


    def layerClicked(self):
        '''Callback for clicking one of the layer buttons.
        '''
        self.setActiveLayer(self.sender().text())


    def setActiveLayer(self, active):
        '''Set the active layer.

        Parameters
        ----------
        layer : int or string
            The index or the name of the layer to activate.
        '''
        if active != self._active:
            ## unmark the previously active layer
            if self._active is not None:
                oldItem = self.layout().itemAt(self._active)
                if oldItem is not None:
                    oldItem.widget().setStyleSheet('')

            ## assign the new active layer
            self._active = active
            if isinstance(self._active, str):
                for index, label in enumerate(self._network.layer_dict.keys()):
                    if active == label:
                        self._active = index

                if isinstance(self._active, str):
                    self._active = None
                    # FIXME: Maybe throw exception!

            ## and mark the new active layer
            if self._active is not None:
                newItem = self.layout().itemAt(self._active)
                if newItem is not None:
                    newItem.widget().setStyleSheet('background-color: red')

        # FIXME[question]: what to emit: index or label?
        #self.selected.emit(self._active)
        self.selected.emit(active)


##
## QNetworkInfoBox
##

from PyQt5.QtWidgets import QLabel
from collections import OrderedDict
from network import Network

class QNetworkInfoBox(QLabel):
    '''A simple widget to display information on a network and a selected
    layer.

    Attributes
    ----------
    _network    :   Network
                    Network to show info about
    _network_text    :   str
                        Info string
    _layer_id   :   str
                    Identifier of the current layer.
    _layer_text  :   str
                    Info text about the current layer

    '''

    _network : Network = None
    _network_text = ''

    _layer_id = None
    _layer_text = ''

    def __init__(self, parent = None):
        '''Create a new ``QNetworkInfoBox``'''
        super().__init__(parent)
        self.setWordWrap(True)
        self.setNetwork()


    def setNetwork(self, network : Network = None, layer_id = None) -> None:
        '''Set the network for which information is displayed.

        Parameters
        ----------
        network :   network.Network
                    The network
        layer_id    :   str
                        Identifier of the layer to show
        '''
        if self._network != network or not self._network_text:
            self._network = network
            self._network_text = '<b>Network info:</b> '
            if self._network is None:
                self._network_text += 'No network selected'
            else:
                network_name = type(self._network).__name__
                self._network_text += 'FIXME[todo]: obtain network information ...'
                self._network_text += f'<br>\n<b>class:</b> {network_name}'

        self.setLayer(layer_id)


    def setLayer(self, layer_id=None) -> None:
        '''Set the layer for which information is displayed.

        Parameters
        ----------
        layer_id    :   str
                        The identifier of the network layer.
        '''

        if layer_id != self._layer_id or not self._layer_text:
            if not self._network:
                self._layer_text = ''
                self._layer_id = None
            else:
                self._layer_id = layer_id
                self._layer_text = '<br>\n<br>\n'
                self._layer_text += '<b>Layer info:</b><br>\n'

                if self._layer_id is None:
                    self._layer_text += 'No layer selected'
                else:
                    layer = self._network.layer_dict[self._layer_id]
                    self._layer_text += self._layerInfoString(layer)

        self.setText(self._network_text + self._layer_text)

    def _layerInfoString(self, layer: Layer) -> str:
        '''Provide a string with information about a network layer.

        Parameters
        -----------
        layer   :   network.Layer
                    The network layer

        Returns
        -------
        str
            A string containing information on that layer.
        '''
        return '\n'.join('<b>{}</b>: {}<br>'.format(key, val) for key, val in
                         layer.info.items())
