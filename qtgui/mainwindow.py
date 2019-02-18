import numpy as np

from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QAction, QMainWindow, QStatusBar, QTabWidget,
                             QWidget, QLabel)


from model import Model, ModelObserver
from qtgui.utils import QtAsyncRunner
from qtgui import panels
from qtgui.panels import (ActivationsPanel, OcclusionPanel,
                          MaximizationPanel, InternalsPanel,
                          LoggingPanel)

from controller import ActivationsController
from controller import DataSourceController, DataSourceObserver
from controller import MaximizationController
from tools.am import Engine as MaximizationEngine, EngineObserver as MaximizationEngineObserver


import time
import logging
import util
from datasources import DataSource
from util import ArgumentError, resources, addons

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
    _maximization_controller : MaximizationController
        An MaximizationController for this Application
    _maximization_engine : MaximizationEngine
        The activation maximzation Engine
    _current_panel : Panel
        The currently selected Panel (FIXME: currently not used!)
    """

    def __init__(self, title:str='QtPyVis') -> None:
        """Initialize the main window.

        Parameters
        ----------
        title: str
        """
        super().__init__()
        self._title = title
        self._runner = QtAsyncRunner()

        import util
        util.runner = self._runner # FIXME[hack]

        self._current_panel = None      
        self._activations = None
        self._maximization = None
        self._internals = None
        self._lucid = None
        self._logging = None
        self._initUI()

    def setModel(self, model: Model) -> None:
        self._activations_controller = \
            ActivationsController(model, runner=self._runner)

        if self._activations is not None:
            self._activations.setController(self._activations_controller,
                                            ModelObserver)
        if self._maximization is not None:
            self._maximization.setController(self._activations_controller,
                                             ModelObserver)

        self._datasource_controller = \
            DataSourceController(model, runner=self._runner)

        if self._activations is not None:
            self._activations.setController(self._datasource_controller,
                                            DataSourceObserver)

    def setMaximizationEngine(self, engine:MaximizationEngine=None) -> None:
        self._maximization_engine = engine
        if self._maximization_engine is not None:
            self._maximization_controller = \
                MaximizationController(self._maximization_engine,
                                       runner=self._runner)
        else:
            self._maximization_controller = None

        if self._maximization is not None:
            self._maximization.setController(self._maximization_controller,
                                             MaximizationEngineObserver)


    def setLucidEngine(self, engine:'LucidEngine'=None) -> None:
        self._lucid_engine = engine
        from controller import LucidController
        self._lucid_controller = LucidController(self._lucid_engine,
                                                 runner=self._runner)
        if self._lucid is not None:
            self._lucid.setController(self._lucid_controller)

    def setDataSource(self, datasource: DataSource) -> None:
        """Set the datasource.
        """
        self._datasource_controller.set_datasource(datasource)
        if self._activations is not None:
            self._activations.setController(self._datasource_controller,
                                            DataSourceObserver)
    
    def _initUI(self):
        """Initialize the graphical components of this user interface."""
        self.setWindowTitle(self._title)
        self._createMenu()
        self._setAppIcon()

        # Initialize the panels
        self.initActivationPanel()
        self.initMaximizationPanel()
        self.initInternalsPanel()
        self.initLoggingPanel()
        if addons.use('lucid'):
            self._lucid = self.initLucidPanel()
        self.initAutoencoderPanel()

        self._createTabWidget()

        self.setCentralWidget(self._tabs)

        # Initialize the status bar
        self._statusResources = QLabel()
        self.statusBar().addWidget(self._statusResources)

    def _createTabWidget(self):
        self._tabs = QTabWidget(self)
        if self._activations is not None:
            self._tabs.addTab(self._activations, 'Activations')
        if self._maximization is not None:
            self._tabs.addTab(self._maximization, 'Maximization')
        if self._lucid is not None:
            self._tabs.addTab(self._lucid, 'Lucid')
        if self._autoencoder is not None:
            self._tabs.addTab(self._autoencoder, 'Autoencoder')
        if self._internals is not None:
            self._tabs.addTab(self._internals, 'Internals')
        if self._logging is not None:
            self._tabs.addTab(self._logging, 'Logging')
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
        # FIXME[problem]: currently this prints the following method:
        # "libpng warning: iCCP: extra compressed data"
        # probably due to some problem with the file 'logo.png'
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

        #
        # Tools Menu
        #
        toolsMenu = menubar.addMenu('&Tools')

        loggingAction = QAction('Logging', self)
        loggingAction.setStatusTip('Show the logging panel')
        # loggingAction.triggered.connect(self.)
        toolsMenu.addAction(loggingAction)

    def closeEvent(self, event):
        """Callback for x button click."""
        self.onExitClicked()

    def initActivationPanel(self):
        """Initialise the activations panel.  This will connect the Input
        button and the Layer buttons.

        """
        self._activations = ActivationsPanel()

    def initMaximizationPanel(self):
        """Initialise the maximization panel.
        """
        self._maximization = MaximizationPanel()

    def initInternalsPanel(self):
        """Initialise the 'internals' panel.
        """
        self._internals = InternalsPanel()

    def initLucidPanel(self):
        """Initialise the 'lucid' panel.
        """
        from .panels.lucid import LucidPanel
        return LucidPanel()

    def initAutoencoderPanel(self):
        """Initialise the autoencoder panel.

        """
        if addons.use('autoencoder'):
            from .panels.autoencoder import AutoencoderPanel
            self._autoencoder = AutoencoderPanel()
        else:
            self._autoencoder = None

    def initLoggingPanel(self):
        """Initialise the log panel.

        """
        self._logging = LoggingPanel()

        import logging
        self._logging.addLogger(logging.getLogger())

    def showStatusMessage(self, message):
        self.statusBar().showMessage(message, 2000)

    def showStatusResources(self):
        message = f"{time.ctime()}"
        message += (", Memory: " +
                    "Shared={:,} kiB, ".format(resources.mem.shared) +
                    "Unshared={:,} kiB, ".format(resources.mem.unshared) +
                    "Peak={:,} kiB".format(resources.mem.peak))
        if len(resources.gpus) > 0:
            message += (f", GPU: temperature={resources.gpus[0].temperature}/"
                        f"{resources.gpus[0].temperature_max}\u00b0C")
            message += (", memory={:,}/{:,}MiB".
                        format(resources.gpus[0].mem,
                               resources.gpus[0].mem_total))
        self._statusResources.setText(message)

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

    ##########################################################################
    #                          Public Interface                              #
    ##########################################################################

    def activateLogging(self, logger: logging.Logger,
                        recorder: util.RecorderHandler=None,
                        show: bool=False) -> None:
        if self._logging is not None:
            if recorder is not None:
                self._logging.setLoggingRecorder(recorder)
            self._logging.addLogger(logger)
            if show:
                self._tabs.setCurrentWidget(self._logging)
