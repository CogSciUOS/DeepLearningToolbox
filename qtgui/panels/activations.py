from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QComboBox
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
    '''

    '''The current network. This network is used for all computations
    in this ActivationsPanel. Can be None if no network is selected.
    '''
    network: object = None

    '''The current network layer. Activations are shown for the respective
    layer. The layer can be None if no layer is selected.
    '''
    # FIXME[question]: int (index) or str (label)?
    layer: str = None

    '''The current input data. This data should always map the input size
    of the current network. May be None if no input data is available.'''
    _data: np.ndarray = None

    '''A signal that is emitted each time some new input data have been
    selected via the GUI. (np.ndarray, str)'''
    inputSelected = pyqtSignal(object, str)

    '''A signal that is emitted each time a new network layer has been
    selected via the GUI.'''
    layerSelected = pyqtSignal(object)

    def __init__(self, parent=None):
        '''Initialization of the ActivationsView.

        Arguments
        ---------
        parent : QWidget
            The parent argument is sent to the QWidget constructor.
        '''
        super().__init__(parent)
        self.initUI()
        self.setNetwork()
        self.setInputData()

        self.sample_index = 0

    def initUI(self):

        #
        # Activations
        #

        # activationview: a canvas to display a layer activation
        #self.activationview = PlotCanvas(None, width=9, height=9)
        self.activationview = QActivationView()
        self.activationview.selected.connect(self.setUnit)

        # FIXME[layout]
        self.activationview.setMinimumWidth(300)
        self.activationview.resize(600, self.activationview.height())

        activationLayout = QVBoxLayout()
        activationLayout.addWidget(self.activationview)

        activationBox = QGroupBox("Activation")
        activationBox.setLayout(activationLayout)

        #
        # Input
        #

        # FIXME[old]
        # inputview: a canvas to display the input image
        #self.inputview = QInputViewMatplotlib(self, width=4, height=4)
        # QImageView: another canvas to display the input image
        # (mayb be more efficient - check!)
        self.inputview = QImageView(self)
        # FIXME[layout]
        # self.inputview.setMinimumWidth(200)
        self.inputview.heightForWidth = lambda w: w
        self.inputview.hasHeightForWidth = lambda: True

        # inputselect: a widget to select the input to the network
        # (data array, image directory, webcam, ...)
        # the "next" button: used to load the next image
        self.inputselector = QInputSelector()
        #self.inputselector.selected = self.inputSelected

        self.inputinfo = QInputInfoBox()
        # FIXME[layout]
        self.inputinfo.setMinimumWidth(300)

        inputLayout = QVBoxLayout()
        # FIXME[layout]
        inputLayout.setSpacing(0)
        inputLayout.setContentsMargins(0, 0, 0, 0)
        inputLayout.addWidget(self.inputview)
        inputLayout.addWidget(self.inputinfo)
        inputLayout.addWidget(self.inputselector)

        inputBox = QGroupBox("Input")
        inputBox.setLayout(inputLayout)

        #
        # Network
        #

        # networkview: a widget to select a layer in a network
        self.networkview = QNetworkView()
        self.networkview.selected.connect(self.setLayer)
        # FIXME[hack]
        # self.networkview.setMinimumSize(300,400)

        self.networkselector = QComboBox()
        self.networks = {'None': None}
        self.networkselector.addItems(self.networks.keys())
        self.networkselector.activated.connect(
            lambda i: self.setNetwork(
                self.networks[self.networkselector.currentText()]))

        # layerinfo: display input and layer info
        self.networkinfo = QNetworkInfoBox()
        # FIXME[layout]
        self.networkinfo.setMinimumWidth(300)

        networkLayout = QVBoxLayout()
        networkLayout.addWidget(self.networkselector)
        networkLayout.addWidget(self.networkview)
        networkLayout.addWidget(self.networkinfo)

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
        name = "Network " + str(self.networkselector.count())
        self.networks[name] = network
        self.networkselector.addItem(name)
        self.setNetwork(network)

    def getNetworkName(self, network):
        name = None
        for n, net in self.networks.items():
            if net == network:
                name = n
        return name

    def setNetwork(self, network=None):

        if self.network != network:
            self.network = network
            # change the network selector to reflect the network
            name = self.getNetworkName(network)
            index = self.networkselector.findText(name)
            self.networkselector.setCurrentIndex(index)

            self.networkview.setNetwork(network)
            self.networkinfo.setNetwork(network)
            self.setLayer(None)

    def setLayer(self, layer=None):
        if layer != self.layer:
            self.layer = layer
            self.networkinfo.setLayer(self.layer)

        # We update the activation on every invocation, no matter if
        # the selected layer changed. This allows for some dynamics
        # if layers like dropout are involved. However, if such
        # layers do not exist, it will only waste computing power ...
        self.layerSelected.emit(layer)
        self.updateActivation()

    def updateActivation(self):
        '''Update this panel in response to a new activation values.
        New activation values can be caused by the selection of another
        layer in the network, but also by the change of the input image.
        '''
        if self.network is None:
            activations = None
        elif self.layer is None:
            activations = None
        elif self._data is None:
            activations = None
        else:
            activations = self.network.get_activations(
                [self.layer], self._data)[0]

        self.activationview.setActivation(activations)

    def setUnit(self, unit: int = None):
        """This methode is involved when the currently selected unit (e.g., in
        the activationview) has changed. This change should be
        reflected in other widgets.
        """

        activationMask = self.activationview.getUnitActivation(unit)
        if activationMask is None:
            activationMask = None
        else:
            ratio = self._data.shape[1]//activationMask.shape[0]
            if ratio > 1:
                activationMask = self.resizemask(activationMask, ratio)
            self.inputview.setActivationMask(activationMask)

    def setInputDataArray(self, data: np.ndarray = None):
        '''Set the input data to be used.
        '''
        self.inputselector.setDataArray(data)

    def setInputDataFile(self, filename: str):
        '''Set the input data to be used.
        '''
        self.inputselector.setDataFile(filename)

    def setInputDataDirectory(self, dirname: str):
        '''Set the input data to be used.
        '''
        self.inputselector.setDataDirectory(dirname)

    def setInputDataSet(self, name: str):
        '''Set the input data set to be used.
        '''
        self.inputselector.setDataSet(name)

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

        self.inputview.setImage(raw)
        self.inputinfo.showInfo(raw, description)

        self._data = fitted
        self.updateActivation()

    def resizemask(self, mask, factor):
        newmask = np.repeat(np.repeat(mask, factor, 0), factor, 1)
        return newmask
