from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout

from network import BaseNetwork
from network.layers.layers import Layer, Conv2D


class QNetworkView(QWidget):
    """A simple widget to display information on a network and select a
    layer.
    """

    network: BaseNetwork = None
    """The network currently displayed in this QNetworkView.
    """

    active = None
    """The identifier of the currently selected layer.  The actual type of
    the identifier depends on the network implementation. The value
    None indicates that no layer is currently selectd.
    """

    selected = pyqtSignal(object)
    """A signal that will be emitted everytime a layer is selected in this
    view. The signal will be emitted multiple times if the same layer
    is pressed multiple times in a raw, as some layers (like dropout),
    may change on every activation. The argument provided by the
    signal will be the layer_id, i.e. the identifier of the label in
    the current network.
    """

    
    def __init__(self, parent=None):
        '''Initialization of the QNetworkView.

        Arguments
        ---------
        parent : QWidget
            The parent argument is sent to the QWidget constructor.
        '''
        super().__init__(parent)
        self.initUI()
        self.setNetwork()


    def initUI(self) -> None:
        '''Initialize the user interface.
        '''
        self.setLayout(QVBoxLayout())


    def setNetwork(self, network=None):
        '''Set the network to display in this widget
        '''
        self.network = network
        self.active = None

        layout = self.layout()

        ## remove the old network buttons
        # FIXME[todo]: (still not sure what is the correct way to do:
        # widget.deleteLater(), widget.close(), widget.setParent(None), or
        # layout.removeWidget(widget) + widget.setParent(None))
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

        if self.network is not None:
            ## a column of buttons to select the network layer
            for name in self.network.layer_dict.keys():
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

        Arguments
        ---------
        layer : int or string
            The index or the name of the layer to activate.
        '''
        if active != self.active:
            ## unmark the previously active layer
            if self.active is not None:
                oldItem = self.layout().itemAt(self.active)
                if oldItem is not None:
                    oldItem.widget().setStyleSheet("")

            ## assign the new active layer
            self.active = active
            if isinstance(self.active, str):
                for index, label in enumerate(self.network.layer_dict.keys()):
                    if active == label:
                        self.active = index

                if isinstance(self.active, str):
                    self.active = None
                    # FIXME: Maybe throw exception!

            ## and mark the new active layer
            if self.active is not None:
                newItem = self.layout().itemAt(self.active)
                if newItem is not None:
                    newItem.widget().setStyleSheet("background-color: red")

        # FIXME[question]: what to emit: index or label?
        #self.selected.emit(self.active)
        self.selected.emit(active)


##
## QNetworkInfoBox
##

from PyQt5.QtWidgets import QLabel
from collections import OrderedDict
from network import BaseNetwork

class QNetworkInfoBox(QLabel):
    """A simple widget to display information on a network and a selected
    layer.
    """
    
    network : BaseNetwork = None
    networkText = ""

    layer_id = None
    layerText = ""

    def __init__(self, parent = None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setNetwork()


    def setNetwork(self, network : BaseNetwork = None, layer_id = None) -> None:
        """Set the network for which information are displayed.

        Arguments
        ---------
        network
            The network.
        """
        if self.network != network or not self.networkText:
            self.network = network
            self.networkText = "<b>Network info:</b> "
            if self.network is None:
                self.networkText += "No network selected"
            else:
                self.networkText += "FIXME[todo]: obtain network information ..."
                self.networkText += "<br>\n<b>class:</b> {}".format(type(self.network).__name__)
        self.setLayer(layer_id)


    def setLayer(self, layer_id=None) -> None:
        """Set the layer for which information are displayed.

        Arguments
        ---------
        layer_id
            The identifier of the network layer.
        """

        if layer_id != self.layer_id or not self.layerText:
            if self.network is None:
                self.layerText = ""
                self.layer_id = None
            else:
                self.layer_id = layer_id
                self.layerText = "<br>\n<br>\n"
                self.layerText += "<b>Layer info:</b><br>\n"

                if self.layer_id is None:
                    self.layerText += "No layer selected"
                else:
                    self.layerText += self._layerInfoString(self.network.layer_dict[self.layer_id])

        self.setText(self.networkText + self.layerText)

    def _layerInfoString(self, layer: Layer) -> str:
        """Provide a string with information about a network layer.

        Parameters
        ----------
        layer
            The network layer.

        Returns
        -------
        A string containing information on that layer.
        """
        info_str = ''
        for key, val in layer.info.items():
            info_str += '<b>{}</b>: {}<br>\n'.format(key, val)
        return info_str

    def _convolutionalLayerInfoString(self, layer_id) -> str:
        """Provide a string with information on a convolutional layer.

        Arguments
        ---------
        layer_id
            The identifier of the network layer. The layer should be
            convolutional.

        Returns
        -------
        str
            A string containing information on that layer.
        """
        features = OrderedDict()
        features['Input shape'] = 'get_layer_input_shape'
        features['Output shape'] = 'get_layer_output_shape'
        features['Input channels'] = 'get_layer_input_channels'
        features['Output channels'] = 'get_layer_output_channels'
        features['Input units'] = 'get_layer_input_units'
        features['Output units'] = 'get_layer_output_units'
        features['Kernel size'] = 'get_layer_kernel_size'
        features['Stride'] = 'get_layer_stride'
        features['Padding'] = 'get_layer_padding'
        features['Output padding'] = 'get_layer_output_padding'
        features['Dilation'] = 'get_layer_dilation'

        text = ""
        for feature, method in features.items():
            try:
                method_to_call = getattr(self.network, method)
                text += ("{}: {}<br>\n".
                     format(feature, method_to_call(layer_id)))
            except NotImplementedError:
                text += ("{}: not implemented ({})<br>\n".
                     format(feature, format(type(self.network).__name__)))
        return text
