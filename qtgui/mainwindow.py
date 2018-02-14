import numpy as np
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QAction, QMainWindow, QStatusBar, QTabWidget,
                             QWidget)
from scipy.misc import imresize

from controller import MainWindowController
from model import Model
from qtgui.panels import ActivationsPanel, OcclusionPanel
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

    def __init__(self, network, data=None, title='QtPyVis'):
        super().__init__()
        self._controller = MainWindowController()
        self._title = title
        self._model = Model(network)
        self.initUI()

    def getModel(self):
        return self._model

    def initUI(self):
        '''Initialize the graphical components of this user interface.'''
        self.setWindowTitle(self._title)
        self._createMenu()

        # Initialize the panels
        self.initActivationPanel()

        self._createTabWidget()

        self.setCentralWidget(self._tabs)

        # Initialize the status bar
        self._statusBar = QStatusBar()

    def _createTabWidget(self):
        self._tabs = QTabWidget(self)
        self._tabs.currentChanged.connect(self._controller.on_panel_selected)
        self._tabs.addTab(self._activations, 'Activations')

    def getTab(self, index: int) -> QWidget:
        '''Get tab for index.

        Parameters
        ----------
        index   :   int
                    Index of the tab (0-based)

        Returns
        -------
        QWidget
            Widget sitting under the selected tab or ``None``, if out of range
        '''

        return self._tabs.widget(index)

    def _setAppIcon(self):
        # add an icon
        self.setWindowIcon(QIcon('assets/logo.png'))

    def _createMenu(self):
        menubar = self.menuBar()
        ########################################################################
        #                            Add exit menu                             #
        ########################################################################
        exitAction = QAction(QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcuts(['Ctrl+W', 'Ctrl+Q'])
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self._controller.on_exit_clicked)

        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)

    def closeEvent(self, event):
        '''Callback for x button click.'''
        self._controller.on_exit_clicked()

    def initActivationPanel(self):
        '''Initialiase the actiations panel.
        This will connect the Input button and the Layer buttons
        '''
        self._activations = ActivationsPanel(self._model)
        self._controller.add_child_controller(self._activations,
                                              self._activations._controller)

    def showStatusMessage(self, message):
        self._statusBar.showMessage(message, 2000)

