from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QComboBox
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QSplitter

from qtgui.widgets import QActivationView
from qtgui.widgets import QInputSelector, QInputInfoBox, QImageView
from qtgui.widgets import QNetworkView, QNetworkInfoBox

import numpy as np

# FIXME[todo]: rearrange the layer selection on network change!
# FIXME[todo]: add docstrings!


class ActivationsPanel(QWidget):
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

    def on_input_selected(self, callback):
        self._input_selector.selected.connect(callback)

    def __init__(self, parent=None):
        '''Initialization of the ActivationsView.

        Parameters
        ---------
        parent  :   QWidget
                    The parent argument is sent to the QWidget constructor.
        '''
        super().__init__(parent)
        self.initUI()

        self.sample_index = 0

        self._network = None

        # FIXME[question]: int (index) or str (label)?
        self._layer = None

        self._data = None

    def initUI(self):

        #
        # Activations
        #

        # activationview: a canvas to display a layer activation
        self._activation_view = QActivationView()
        self._activation_view.selected.connect(self.setUnit)

        # FIXME[layout]
        self._activation_view.setMinimumWidth(300)
        self._activation_view.resize(600, self._activation_view.height())

        activationLayout = QVBoxLayout()
        activationLayout.addWidget(self._activation_view)

        activationBox = QGroupBox("Activation")
        activationBox.setLayout(activationLayout)

        #
        # Input
        #

        # FIXME[old]
        # inputview: a canvas to display the input image
        # QImageView: another canvas to display the input image
        # (mayb be more efficient - check!)
        self._input_view = QImageView(self)
        # FIXME[layout]
        # self._input_view.setMinimumWidth(200)
        self._input_view.heightForWidth = lambda w: w
        self._input_view.hasHeightForWidth = lambda: True

        # inputselect: a widget to select the input to the network
        # (data array, image directory, webcam, ...)
        # the "next" button: used to load the next image
        self._input_selector = QInputSelector()

        self._input_info = QInputInfoBox()
        # FIXME[layout]
        self._input_info.setMinimumWidth(300)

        inputLayout = QVBoxLayout()
        # FIXME[layout]
        inputLayout.setSpacing(0)
        inputLayout.setContentsMargins(0, 0, 0, 0)
        inputLayout.addWidget(self._input_view)
        inputLayout.addWidget(self._input_info)
        inputLayout.addWidget(self._input_selector)

        inputBox = QGroupBox("Input")
        inputBox.setLayout(inputLayout)

        #
        # Network
        #

        # networkview: a widget to select a layer in a network
        self._network_view = QNetworkView()
        self._network_view.selected.connect(self.setLayer)
        # FIXME[hack]
        # self._network_view.setMinimumSize(300,400)

        self._network_selector = QComboBox()
        self._networks = {'None': None}
        self._network_selector.addItems(self._networks.keys())
        self._network_selector.activated.connect(
            lambda i: self.setNetwork(
                self._networks[self._network_selector.currentText()]))

        # layerinfo: display input and layer info
        self._network_info = QNetworkInfoBox()
        # FIXME[layout]
        self._network_info.setMinimumWidth(300)

        networkLayout = QVBoxLayout()
        networkLayout.addWidget(self._network_selector)
        networkLayout.addWidget(self._network_view)
        networkLayout.addWidget(self._network_info)

        networkBox = QGroupBox("Network")
        networkBox.setLayout(networkLayout)

        #
        # Putting all together
        #

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(activationBox)
        splitter.addWidget(inputBox)
        layout = QHBoxLayout()
        # layout.addWidget(activationBox)
        # layout.addWidget(inputBox)
        layout.addWidget(splitter)
        layout.addWidget(networkBox)
        self.setLayout(layout)

    def addNetwork(self, network):
        name = "Network " + str(self._network_selector.count())
        self._networks[name] = network
        self._network_selector.addItem(name)
        self.setNetwork(network)

    def getNetworkName(self, network):
        name = None
        for n, net in self._networks.items():
            if net == network:
                name = n
        return name

    def setNetwork(self, network=None):

        if self._network != network:
            self._network = network
            # change the network selector to reflect the network
            name = self.getNetworkName(network)
            index = self._network_selector.findText(name)
            self._network_selector.setCurrentIndex(index)

            self._network_view.setNetwork(network)
            self._network_info.setNetwork(network)
            self.setLayer(None)

    def setLayer(self, layer=None):
        if self._layer != layer:
            self._layer = layer
            self._network_info.setLayer(layer)

        # We update the activation on every invocation, no matter if
        # the selected layer changed. This allows for some dynamics
        # if layers like dropout are involved. However, if such
        # layers do not exist, it will only waste computing power ...
        # self.layerSelected.emit(layer)
        self.updateActivation()

    def updateActivation(self):
        '''Update this panel in response to a new activation values.
        New activation values can be caused by the selection of another
        layer in the network, but also by the change of the input image.
        '''
        if self._network is None:
            activations = None
        elif self._layer is None:
            activations = None
        elif self._data is None:
            activations = None
        else:
            activations = self._network.get_activations(
                [self._layer], self._data)[0]

        self._activation_view.setActivation(activations)

    def setUnit(self, unit: int = None):
        """This methode is involved when the currently selected unit (e.g., in
        the activationview) has changed. This change should be
        reflected in other widgets.
        """

        activationMask = self._activation_view.getUnitActivation(unit)
        if activationMask is None:
            activationMask = None
        else:
            ratio = self._data.shape[1]//activationMask.shape[0]
            if ratio > 1:
                activationMask = self.resizemask(activationMask, ratio)
            self._input_view.setActivationMask(activationMask)

    def setInputDataArray(self, data: np.ndarray = None):
        '''Set the input data to be used.
        '''
        self._input_selector.setDataArray(data)

    def setInputDataFile(self, filename: str):
        '''Set the input data to be used.
        '''
        self._input_selector.setDataFile(filename)

    def setInputDataDirectory(self, dirname: str):
        '''Set the input data to be used.
        '''
        self._input_selector.setDataDirectory(dirname)

    def setInputDataSet(self, name: str):
        '''Set the input data set to be used.
        '''
        self._input_selector.setDataSet(name)

    def setInputData(
            self,
            raw: np.ndarray = None,
            fitted: np.ndarray = None,
            description: str = None):
        '''Set the current input stimulus for the network.
        The input stimulus is take from the internal data collection.

        Argruments
        ----------
        raw:
            The raw input data, as provided by the data source.
        fitted:
            The input data transformed to fit the network.
        description:
            A string describing where the origin of the input data.
        '''

        self._input_view.setImage(raw)
        self._input_info.showInfo(raw, description)

        self._data = fitted
        self.updateActivation()

    def resizemask(self, mask, factor):
        newmask = np.repeat(np.repeat(mask, factor, 0), factor, 1)
        return newmask
