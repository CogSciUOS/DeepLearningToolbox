import numpy as np
from scipy.misc import imresize

from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QMainWindow, QTabWidget, QAction, QStatusBar
from PyQt5.QtGui import QIcon

from qtgui.panels import ActivationsPanel
from util import ArgumentError

class DeepVisMainWindow(QMainWindow):
    '''The main window of the deep visualization toolbox. The window is
    intended to hold different panels that allow for different kinds
    of visualizations.

    This class also provides central entry points to programmatically
    set certain aspects, like loading a network or input data,
    switching between different panels, etc.

    Attributes
    ----------
    _title   :   str
                Window title

    '''

    def __init__(self, network=None, data=None, title='QtPyVis'):
        super().__init__()

        self._title = title

        self.initUI()

    def initUI(self):
        '''Initialize the graphical components of this user interface.'''
        self.setWindowTitle(self._title)

        menubar = self.menuBar()
        ########################################################################
        #                            Add exit menu                             #
        ########################################################################
        exitAction = QAction(QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcuts(['Ctrl+W', 'Ctrl+Q'])
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QCoreApplication.quit)

        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)

        ##
        # Initialize the panels ##
        self._activations = ActivationsPanel()
        self.initActivationPanel()

        self._tabs = QTabWidget(self)
        self._tabs.addTab(self._activations, 'Main')

        self.setCentralWidget(self._tabs)

        ##
        # Initialize the status bar
        ##
        self._statusBar = QStatusBar()
        self.setStatusBar(self._statusBar)

    def initActivationPanel(self):
        '''Initialiase the actiations panel.
        This will connect the Input button and the Layer buttons
        '''
        self._activations.inputselector.selected.connect(self.setInputData)
        self._activations.inputSelected.connect(self.setInputData)

    def setNetwork(self, network):
        '''Set the model to visualise
        Parameters
        ----------
        network :   network.network.Network
                    The network to use
        '''
        self._network = network
        self._activations.addNetwork(network)

    def setInputDataArray(self, data: np.ndarray):
        '''Set the input data to be used.

        Parameters
        ----------
        data    :   np.ndarray
                    Input data array
        '''
        self._activations.setInputDataArray(data)

    def setInputDataFile(self, filename: str):
        '''Set the input data to be used (read from file)

        Parameters
        ----------
        filename    :   str
                        Input filename

        '''
        self._activations.setInputDataFile(filename)

    def setInputDataDirectory(self, dirname: str):
        '''Set input data directory.

        Parameters
        ----------
        dirname     :   str
                        Directory to read from

        '''
        self._activations.setInputDataDirectory(dirname)

    def setInputDataSet(self, name: str):
        '''Set the input data to be used.'''
        self._activations.setInputDataSet(name)

    def showStatusMessage(self, message):
        self._statusBar.showMessage(message, 2000)

    def setInputData(self, data: np.ndarray=None, description: str=None):
        '''Provide one data vector as input for the network.'''
        # FIXME[hack]: there seems to be a bug in PyQt forbidding to emit
        # signals with None parameter. See code in 'widget/inputselector.py'

        if not data.ndim:
            raise ArgumentError('Data cannot be none.')

        raw_data = data
        if self._network is not None and data is not None:
            network_shape = self._network.get_input_shape(include_batch=False)
            if data.ndim > 4 or data.ndim < 2:
                raise ArgumentError('Data must have between 2 and 4 dimensions.')

            if data.ndim == 4:
                if data.shape[0] == 1:
                    # first dimension has size of 1 -> remove
                    data = data.squeeze(0)
                else:
                    raise ArgumentError('Cannot visualize batch of images')

            if data.ndim == 2:
                data = data[..., np.newaxis].repeat(3, axis=2).copy()

            if data is not None and data.shape[0:2] != network_shape[0:2]:
                data = imresize(data, network_shape[0:2])

            if data is not None and data.shape[2] != network_shape[2]:
                # FIXME[hack]: find better way to doe RGB <-> grayscale
                # conversion
                if network_shape[2] == 1:
                    data = data[:, :, 0:1]
                elif network_shape[2] == 3 and data.shape[2] == 1:
                    data = data.repeat(3, axis=2)
                else:
                    data = None

        self._activations.setInputData(raw_data, data, description)
