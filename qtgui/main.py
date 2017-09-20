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
        self.activations.inputSelected.connect(self.setInputData)
        self.activations.layerSelected.connect(self.setLayer)
        #self.activations.networkSelected.connect(self.setNetwork)


    # FIXME[hack]: split up into sensible functions or rename ...
    def setNetwork(self, network, data):
        self.activations.addNetwork(network)
        self.activations.setInputData(data)
        self.experiments.setNetwork(network)
#        self.occlusion.addNetwork(network)
#        self.occlusion.setInputData(data)
        self.update()


    def showStatusMessage(self, message):
        self.statusBar.showMessage(message, 2000)

        
    def setLayer(self, layer = None) -> None:
        self.experiments.setLayer(layer)

    def setInputData(self, data = None):
        '''Provide one data vector as input for the network.
        '''
        self.experiments.setInputData(data)
