import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt

from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QMainWindow, QTabWidget, QAction, QStatusBar
from PyQt5.QtGui import QIcon

#from qtgui.panels import ActivationsPanel, ExperimentsPanel, OcclusionPanel
from qtgui.panels import ActivationsPanel, ExperimentsPanel

# FIXME[todo]: add docstrings!

class DeepVisMainWindow(QMainWindow):
    '''The main window of the deep visualization toolbox.  The window is
    intended to hold different panels that allow for different kinds
    of visualizations.

    This class also provides central entry points to programmatically
    set certain aspects, like loading a network or input data,
    switching between different panels, etc.
    '''

    def __init__(self, network=None, data=None):
        super().__init__()

        # FIXME[matplotlib]: only needed if using matplotlib for ploting ...
        # prepare matplotlib for interactive plotting on the screen
        plt.ion()

        self.title = 'Activations'

        self.left = 10
        self.top = 10
        self.width = 1800
        self.height = 900

        self.initUI()


    def initUI(self):
        '''Initialize the graphical components of this user interface.
        '''
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        ##
        ## Initialize the menu bar
        ##
        exitAction = QAction(QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QCoreApplication.quit)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)

        ##
        ## Initialize the panels
        ##
        self.activations = ActivationsPanel()
        self.experiments = ExperimentsPanel()
#        self.occlusion = OcclusionPanel()

        self.tabs = QTabWidget(self);
        self.tabs.addTab(self.activations, "Main")
        self.tabs.addTab(self.experiments, "Experiments")
#        self.tabs.addTab(self.occlusion, "Occlusion")

        self.setCentralWidget(self.tabs)

        ##
        ## Initialize the status bar
        ##
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)


        ##
        ## Connect the signals
        ##
        # FIXME[concept]:
        self.activations.inputselector.selected.connect(self.setInputData)
        self.activations.inputSelected.connect(self.setInputData)
        self.activations.layerSelected.connect(self.setLayer)
        #self.activations.networkSelected.connect(self.setNetwork)


    def setNetwork(self, network):
        self._network = network
        self.activations.addNetwork(network)
        self.experiments.setNetwork(network)
#        self.occlusion.addNetwork(network)


    def setInputDataArray(self, data : np.ndarray):
        '''Set the input data to be used.
        '''
        self.activations.setInputDataArray(data)
#        self.occlusion.setInputData(data)

    def setInputDataFile(self, filename : str):
        '''Set the input data to be used.
        '''
        self.activations.setInputDataFile(filename)

    def setInputDataDirectory(self, dirname : str):
        '''Set the input data to be used.
        '''
        self.activations.setInputDataDirectory(dirname)

    def setInputDataSet(self, name : str):
        '''Set the input data to be used.
        '''
        self.activations.setInputDataSet(name)


    def showStatusMessage(self, message):
        self.statusBar.showMessage(message, 2000)


    def setLayer(self, layer = None) -> None:
        self.experiments.setLayer(layer)


    def setInputData(self, data : np.ndarray = None, description : str = None):
        '''Provide one data vector as input for the network.
        '''
        # FIXME[hack]: there seems to be a bug in PyQt forbidding to emit
        # signals with None parameter. See code in "widget/inputselector.py"
        if not data.ndim: data = None

        raw_data = data
        if self._network is not None and data is not None:
            network_shape = self._network.get_input_shape(include_batch=False)
            invalid = None
            if data.ndim > 4 or data.ndim < 2:
                data = None
                invalid = "Do not understand {}-dimensional data".format(data.ndim)

            if data is not None and data.ndim == 4:
                if data.shape[0] == 1:
                    data = data.squeeze(0)
                else:
                    data = None
                    invalid = "Cannot visualize batch of images"

            if data is not None and data.ndim == 2:
                data = data[..., np.newaxis].repeat(3,axis=2).copy()

            if data is not None and data.shape[0:2] != network_shape[0:2]:
                data = imresize(data, network_shape[0:2])

            if data is not None and data.shape[2] != network_shape[2]:
                # FIXME[hack]: find better way to doe RGB <-> grayscale conversion
                if network_shape[2] == 1:
                    data = data[:,:,0:1]
                elif network_shape[2] == 3 and data.shape[2] == 1:
                    data = data.repeat(3,axis=2)
                else:
                    invalid = "Cannot map {}-data to {}-network.".format(data.shape, network_shape)
                    data = None


        self.activations.setInputData(raw_data, data, description)
        self.experiments.setInputData(data)
