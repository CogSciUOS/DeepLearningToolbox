from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout

# FIXME[todo]: add docstrings!

class QNetworkView(QWidget):

    # FIXME: int (index) or str (label)?
    selected = pyqtSignal(str)


    def __init__(self, parent = None):
        '''Initialization of the QNetworkView.

        Arguments
        ---------
        parent : QWidget
            The parent argument is sent to the QWidget constructor.
        '''
        super().__init__(parent)
        self.initUI()
        self.setNetwork()


    def initUI(self):
        self.setLayout(QVBoxLayout())


    def setNetwork(self, network = None):
        self.network = network
        self.active = None

        layout = self.layout()

        # remove the old network buttons
        # (still not sure what is the correct way to do:
        # widget.deleteLater(), widget.close(), widget.setParent(None), or
        # layout.removeWidget(widget) + widget.setParent(None))
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

        if self.network is not None:
            # a column of buttons to select the network layer
            for i,name in enumerate(self.network.layer_ids):
                button = QPushButton(name, self)
                layout.addWidget(button)
                button.resize(button.sizeHint())
                button.clicked.connect(self.layerClicked)

            # choose the active layer for the new network
            #self.setActiveLayer(0)

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
            # unmark the previously active layer
            if self.active is not None:
                oldItem = self.layout().itemAt(self.active)
                if oldItem is not None:
                    oldItem.widget().setStyleSheet("")

            # assign the new active layer
            self.active = active
            if isinstance(self.active, str):
                for index, label in enumerate(self.network.layer_ids):
                    if active == label:
                        self.active = index

                if isinstance(self.active, str):
                    self.active = None
                    # FIXME: Maybe throw exception!

            # and mark the new active layer
            if self.active is not None:
                newItem = self.layout().itemAt(self.active)
                if newItem is not None:
                    newItem.widget().setStyleSheet("background-color: red")

            # FIXME[question]: what to emit: index or label?
            #self.selected.emit(self.active)
            self.selected.emit(active)


# FIXME[todo]: add docstrings!

from PyQt5.QtWidgets import QLabel

class QNetworkInfoBox(QLabel):

    network : object = None
    networkText = ""

    layer = None
    layerText = ""

    def __init__(self, parent = None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setNetwork()

    def setNetwork(self, network = None, layer = None):
        if self.network != network or not self.networkText:
            self.network = network
            self.networkText = "<b>Network info:</b> "
            if self.network is None:
                self.networkText += "No network selected"
            else:
                self.networkText += "FIXME[todo]: obtain network information ..."
                self.networkText += "<br>\n<b>class:</b> {}".format(type(self.network).__name__)
        self.setLayer(layer)


    def setLayer(self, layer = None):
        # FIXME: shape should be obtain from self.network, not passed as argument!
        print("setLayer2 : {} -> {}".format(self.layer,layer))

        if layer != self.layer or not self.layerText:
            if self.network is None:
                self.layerText = ""
                self.layer = None
            else:
                self.layer = layer
                self.layerText = "<br>\n<br>\n"
                self.layerText += "<b>Layer info:</b> "

                if self.layer is None:
                    self.layerText += "No layer selected"
                else:
                    shape = None
                    try:
                        shape = self.network.get_layer_output_shape(self.layer)
                        info = self.network.get_layer_info(self.layer)
                        self.layerText += info['name'] + "<br>\n"
                    except NotImplementedError:
                        self.layerText += "The implementation provides no information for this layer.<br>\n"
                    self.layerText += ("Fully connected" if len(shape) == 2 else "Convolutional") + "<br>\n"
                    self.layerText += "Shape: {}<br>\n".format(shape)
                    print(shape)
                    if shape is not None and len(shape) > 2:
                        try:
                            weights, bias = self.network.get_layer_weights(self.layer)
                            self.layerText += "{} kernels of size {}x{}<br>\n".format(weights.shape[3],weights.shape[1],weights.shape[0])
                        except NotImplementedError:
                            self.layerText += "The implementation provides no further information.<br>\n"
                        except ValueError:
                            self.layerText += "FIXME: ValueError!<br>\n"

        self.setText(self.networkText + self.layerText)
