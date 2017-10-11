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
    different layer of a network. The panel offers controls to select
    different input stimuli, and to select a specific network layer.
    '''

    network : object = None
    # FIXME[question]: int (index) or str (label)?
    layer : str = None

    # FIXME[move]: not part of the action view, but rather input handling ...
    data : object = None
    dataIndex : int = None

    inputSelected = pyqtSignal(object)
    layerSelected = pyqtSignal(object)


    def __init__(self, parent = None):
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
        #self.inputview.setMinimumWidth(200)
        self.inputview.heightForWidth = lambda w : w
        self.inputview.hasHeightForWidth = lambda : True
        #print("heightForWidth({})={} [{}]".format(200,self.inputview.heightForWidth(200), self.inputview.hasHeightForWidth()))

        # inputselect: a widget to select the input to the network
        # (data array, image directory, webcam, ...)
        # the "next" button: used to load the next image
        self.inputselector = QInputSelector()
        self.inputselector.setNumberOfElements(20) # FIXME[hack]
        self.inputselector.selected.connect(self.setInput)

        self.inputinfo = QInputInfoBox()
        # FIXME[layout]
        self.inputinfo.setMinimumWidth(300)

        inputLayout = QVBoxLayout()
        # FIXME[layout]
        inputLayout.setSpacing(0)
        inputLayout.setContentsMargins(0,0,0,0)
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
        #self.networkview.setMinimumSize(300,400)

        self.networkselector = QComboBox()
        self.networks = { 'None': None }
        self.networkselector.addItems(self.networks.keys())
        self.networkselector.activated.connect(lambda i:self.setNetwork(self.networks[self.networkselector.currentText()]))

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
        #layout.addWidget(activationBox)
        #layout.addWidget(inputBox)
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


    def setNetwork(self, network = None):

        if self.network != network:
            self.network = network
            # change the network selector to reflect the network
            name = self.getNetworkName(network)
            index = self.networkselector.findText(name)
            self.networkselector.setCurrentIndex(index)

            self.networkview.setNetwork(network)
            self.networkinfo.setNetwork(network)
            self.setLayer(None)


    def setLayer(self, layer = None):
        if layer != self.layer:
            self.layer = layer
            self.networkinfo.setLayer(self.layer)

        ## We update the activation on every invocation, no matter if
        ## the selected layer changed. This allows for some dynamics
        ## if layers like dropout are involved. However, if such
        ## layers do not exist, it will only waste computing power ...
        self.layerSelected.emit(layer)
        self.updateActivation()


    def updateActivation(self):

        if self.network is None:
            activations = None
        elif self.layer is None:
            activations = None
        elif self.dataIndex is None:
            activations = None
        else:
            input = self.data[self.dataIndex:self.dataIndex+1,:,:,0:1]
            activations = self.network.get_activations([self.layer], input)[0]

        self.activationview.setActivation(activations)


    def setUnit(self, unit : int = None):
        """This methode is involved when the currently selected unit (e.g., in
        the activationview) has changed. This change should be
        reflected in other widgets.
        """

        activationMask = self.activationview.getUnitActivation(unit)
        if activationMask is None:
            activationMask=None
        else:
            if self.data.shape[1]//activationMask.shape[0]>1:
                activationMask=self.resizemask(activationMask,self.data.shape[1]//activationMask.shape[0])
            self.inputview.setActivationMask(activationMask)


    def setInputData(self, data = None):
        '''Provide a collection of input data for the network.
        '''
        number = None if data is None else len(data)
        self.data = data
        self.inputselector.setNumberOfElements(number)


    def setInput(self, input : int = None):
        '''Set the current input stimulus for the network.
        The input stimulus is take from the internal data collection.

        Argruments
        ----------
        input:
            The index of the input stimulus in the data collection.
        '''
        if input is None or self.data is None:
            self.dataIndex = None
        else:
            self.dataIndex = input
            self.updateInput()

        if self.dataIndex is None:
            self.inputSelected.emit(None)
        else:
            # FIXME[hack]: just provide image data without additional axes ...
            self.inputSelected.emit(self.data[self.dataIndex:self.dataIndex+1,:,:,0:1])


    def updateInput(self):
        if self.dataIndex is not None:
            self.inputview.setImage(self.data[self.dataIndex,:,:,0])
            self.inputinfo.showInfo("{}/{}".
                                    format(self.dataIndex,len(self.data)),
                                    self.data.shape[1:3])
            self.updateActivation()
        else:
            print("FIXME: no input data selected!")
    def resizemask(self,mask,factor):
        newmask=np.repeat(np.repeat(mask,factor,0),factor,1)
        return newmask
