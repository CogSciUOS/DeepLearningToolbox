import numpy as np

from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QAction, QMainWindow, QStatusBar, QTabWidget,
                             QWidget)

from model import Model, ModelObserver
from qtgui.utils import QtAsyncRunner
from qtgui.panels import (ActivationsPanel, OcclusionPanel,
                          MaximizationPanel, InternalsPanel)

from controller import ActivationsController
from controller import DataSourceController, DataSourceObserver
from datasources import DataSource
from util import ArgumentError


class DeepVisMainWindow(QMainWindow):
    """The main window of the Deep Visualization Toolbox. The window
    is intended to hold different panels that allow for different
    kinds of visualizations.

    This class also provides central entry points to programmatically
    set certain aspects, like loading a network or input data,
    switching between different panels, etc.

    Attributes
    ----------
    _title : str
        Window title
    _activations_controller : ActivationsController
        An ActivationsController for this Application
    _datasource_controller : DataSourceController
        An ActivationsController for this Application
    _model : Model
        Backing model instance
    _current_panel : Panel
        The currently selected Panel (FIXME: currently not used!)
    """

    def __init__(self, model, data=None, title='QtPyVis'):
        super().__init__()
        self._title = title
        self._model = model
        self._runner = QtAsyncRunner(model)
        self._activations_controller = \
            ActivationsController(model, runner=self._runner)
        self._datasource_controller = \
            DataSourceController(model, runner=self._runner)
        self._current_panel = None
        self._initUI()

    def getModel(self):
        return self._model

    def setDataSource(self, datasource: DataSource) -> None:
        self._datasource_controller.set_datasource(datasource)
    
    def _initUI(self):
        """Initialize the graphical components of this user interface."""
        self.setWindowTitle(self._title)
        self._createMenu()
        self._setAppIcon()

        # Initialize the panels
        self.initActivationPanel()
        self.initMaximizationPanel()
        self.initInternalsPanel()

        self._createTabWidget()

        self.setCentralWidget(self._tabs)

        # Initialize the status bar
        self._statusBar = QStatusBar()
        

    def _createTabWidget(self):
        self._tabs = QTabWidget(self)
        if self._activations is not None:
            self._tabs.addTab(self._activations, 'Activations')
        if self._maximization is not None:
            self._tabs.addTab(self._maximization, 'Maximization')
        if self._internals is not None:
            self._tabs.addTab(self._internals, 'Internals')
        self._tabs.currentChanged.connect(self.onPanelSelected)

    def getTab(self, index: int) -> QWidget:
        """Get tab for index.

        Parameters
        ----------
        index   :   int
                    Index of the tab (0-based)

        Returns
        -------
        QWidget
            Widget sitting under the selected tab or ``None``, if out of range
        """
        return self._tabs.widget(index)

    def _setAppIcon(self):
        self.setWindowIcon(QIcon('assets/logo.png'))

    def _createMenu(self):
        menubar = self.menuBar()

        #
        # Add exit menu
        #
        exitAction = QAction(QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcuts(['Ctrl+W', 'Ctrl+Q'])
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.onExitClicked)

        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)

        #
        # Network Menu
        #
        networkMenu = menubar.addMenu('&Network')

        #
        # Datasource Menu
        #
        datasourceMenu = menubar.addMenu('&Data')

    def closeEvent(self, event):
        """Callback for x button click."""
        self.onExitClicked()

    def initActivationPanel(self):
        """Initialise the activations panel.  This will connect the Input
        button and the Layer buttons.

        """
        self._activations = ActivationsPanel()
        self._activations.setController(self._activations_controller,
                                        ModelObserver)
        self._activations.setController(self._datasource_controller,
                                        DataSourceObserver)

    def initMaximizationPanel(self):
        """Initialise the maximization panel.
        """
        self._maximization = MaximizationPanel()
        self._activations.setController(self._activations_controller,
                                        ModelObserver)

    def initInternalsPanel(self):
        """Initialise the 'internals' panel.
        """
        self._internals = InternalsPanel()

    def showStatusMessage(self, message):
        self._statusBar.showMessage(message, 2000)

    ###########################################################################
    #                        MainWindowController                             #
    ###########################################################################

    # Original idea: Controller for the main GUI window. Will form the
    # base handler for all events and aggregate subcontrollers for
    # individual widgets.

    def onPanelSelected(self, panel: 'qtgui.panels.Panel'):
        '''Callback for selecting a new panel in the main window.

        Parameters
        ----------
        panel   :   qtgui.panels.Panel
                    The newly selected panel
        '''
        self._panel = panel

    def _saveState(self):
        '''Callback for saving any application state inb4 quitting.'''
        pass

    def onExitClicked(self):
        '''Callback for clicking the exit button. This will save state and
        then terminate the Qt application.

        '''
        self._saveState()
        QCoreApplication.quit()
