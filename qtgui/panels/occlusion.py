import numpy as np

from PyQt5.QtWidgets import QWidget
from qtgui.widgets import QMatrixView
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QComboBox
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox
import matplotlib.pyplot as plt
from qtgui.widgets import QActivationView
from qtgui.widgets import QInputSelector, QInputInfoBox, QImageView
from qtgui.widgets import QNetworkView, QNetworkInfoBox

# FIXME[todo]: add docstrings!

class OcclusionPanel(QWidget):
    '''This Panel is intended visualization of occlusion algorithm
    '''

    network : object = None
    data : object = None
    dataIndex : int = None
    layer : str = None

    def __init__(self, parent = None):
        '''Initialization of the ExperimentsView.

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
        '''occlusion'''
        self.occlusionview=QImageView(self)
        occlusionLayout=QVBoxLayout()
        occlusionLayout.addWidget(self.occlusionview)
        occlusionBox = QGroupBox("Oclussion")
        occlusionBox.setLayout(occlusionLayout)



        self.inputview = QImageView(self)

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
        '''
        # networkview: a widget to select a layer in a network
        self.networkview = QNetworkView()
        self.networkview.selected.connect(self.setLayer)
        # FIXME[hack]
        #self.networkview.setMinimumSize(300,400)
        '''
        self.networkselector = QComboBox()
        self.networks = { 'None': None }
        self.networkselector.addItems(self.networks.keys())
        self.networkselector.activated.connect(lambda i:self.setNetwork(self.networks[self.networkselector.currentText()]))
        '''
        # layerinfo: display input and layer info
        self.networkinfo = QNetworkInfoBox()
        # FIXME[layout]
        self.networkinfo.setMinimumWidth(300)

        networkLayout = QVBoxLayout()
        networkLayout.addWidget(self.networkselector)
        networkLayout.addWidget(self.networkview)
        networkLayout.addWidget(self.networkinfo)
        '''
        #networkBox = QGroupBox("Network")
        #networkBox.setLayout(networkLayout)


        #
        # Putting all together
        #

        layout = QHBoxLayout()

        layout.addWidget(inputBox)
        layout.addWidget(occlusionBox)

        #layout.addWidget(networkBox)
        self.setLayout(layout)

    def addNetwork(self, network):
        name = "Network " + str(self.networkselector.count())
        self.networks[name] = network
        self.networkselector.addItem(name)
        self.setNetwork(network)

    def setInputData(self, data = None):
        '''Provide a collection of input data for the network.
        '''
        number = None if data is None else len(data)
        self.data = data
        self.inputselector.setNumberOfElements(number)

    def setNetwork(self, network = None):

        if self.network != network:
            self.network = network
            # change the network selector to reflect the network
            name = self.getNetworkName(network)
            index = self.networkselector.findText(name)
            self.networkselector.setCurrentIndex(index)

            #self.networkview.setNetwork(network)
            #self.networkinfo.setNetwork(network)
            #self.setLayer(None)

    def updateOcclusion(self):

        if self.network is None:
            occlusion = None
        elif self.dataIndex is None:
            occlusion = None
        else:
            input = self.data[self.dataIndex:self.dataIndex+1,:,:,0:1]
            #FIXME kernerlsize
            slidingwindowsize=1
            occlusion = self.network.get_occlussion_map(input, slidingwindowsize)


        plt.imshow(occlusion[0,:,:,0],cmap='gray')
        self.occlusionview.setImage(input[0,:,:,0])


        self.occlusionview.setActivationMask(occlusion[0,:,:,0])

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



    def setLayer(self, layer = None):
        print("setLayer : {} -> {}".format(self.layer,layer))
        if layer != self.layer:
            self.layer = layer
            self.networkinfo.setLayer(self.layer)
            self.updateActivation()

    def getNetworkName(self, network):
        name = None
        for n, net in self.networks.items():
            if net == network:
                name = n
        return name


    def updateInput(self):
        if self.dataIndex is not None:
            self.inputview.setImage(self.data[self.dataIndex,:,:,0])
            self.inputinfo.showInfo("{}/{}".
                                    format(self.dataIndex,len(self.data)),
                                    self.data.shape[1:3])
            self.updateOcclusion()
        else:
            print("FIXME: no input data selected!")
