import time
import logging
import collections
import importlib

from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QAction, QMainWindow, QStatusBar, QTabWidget,
                             QWidget, QLabel)

from base import Runner
from toolbox import ToolboxController

from qtgui.utils import QtAsyncRunner
from qtgui.panels import Panel

import util
from util import resources, addons

# FIXME[old]: this should net be needed for the MainWindow
from model import Model, ModelObserver
from controller import ActivationsController
from controller import MaximizationController
from tools.am import Engine as MaximizationEngine, EngineObserver as MaximizationEngineObserver
from datasources import Datasource, Controller as DatasourceController


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
    _datasource_controller : DatasourceController
        An ActivationsController for this Application
    _maximization_controller : MaximizationController
        An MaximizationController for this Application
    _maximization_engine : MaximizationEngine
        The activation maximzation Engine
    _current_panel : Panel
        The currently selected Panel (FIXME: currently not used!)
    """
    PanelMeta = collections.namedtuple('PanelMeta',
                                       'id label cls tooltip addons')
    _panelMetas = [
        PanelMeta('activations', 'Activations',
                  '.panels.activations.ActivationPanel',
                  'Show the activations panel', []),
        PanelMeta('autoencoder', 'Autoencoder',
                  '.panels.autoencoder.AutoencoderPanel',
                  'Show the autoencoder panel', []),
        PanelMeta('maximization', 'Maximization',
                  '.panels.maximization.MaximizationPanel',
                  'Show the activation maximization panel', []),
        PanelMeta('lucid', 'Lucid',
                  '.panels.lucid.LucidPanel',
                  'Show the Lucid panel', ['lucid']), #  FIXME[todo]: if addons.use('lucid'):
        PanelMeta('advexample', 'Adv. Examples',
                  '.panels.advexample.AdversarialExamplePanel',
                  'Show the adversarial example panel', ['advexample']),  # FIXME[todo]: if addons.use('advexample'):
        PanelMeta('occlusion', 'Occlusion',
                  '.panels.occlusion.OcclusionPanel',
                  'Show the occlusion panel', []),
        PanelMeta('resources', 'Resources',
                  '.panels.resources.ResourcesPanel',
                  'Show the resources panel', []),
        PanelMeta('experiments', 'Experiments',
                  '.panels.experiments.ExperimentsPanel',
                  'Show the Experiments panel', []),
        PanelMeta('internals', 'Internals',
                  '.panels.internals.InternalsPanel',
                  'Show the internals panel', []),
        PanelMeta('logging', 'Logging',
                  '.panels.logging.LoggingPanel',
                  'Show the logging panel', [])
    ]

    def __init__(self, toolboxController: ToolboxController,
                 title: str='QtPyVis') -> None:
        """Initialize the main window.

        Parameters
        ----------
        title: str
        """
        super().__init__()
        self._toolboxController = toolboxController
        self._title = title
        self._runner = QtAsyncRunner()
        self._initUI()

    ###########################################################################
    #                           Initialization                                #
    ###########################################################################

    def _initUI(self):
        """Initialize the graphical components of this user interface."""
        #
        # Initialize the Window
        #
        self.setWindowTitle(self._title)
        self._createMenu()
        self._setAppIcon()

        #
        # Initialize the Tabs
        #
        self._tabs = QTabWidget(self)
        self._tabs.currentChanged.connect(self.onPanelSelected)
        self.setCentralWidget(self._tabs)

        #
        # Initialize the status bar
        #
        self._statusResources = QLabel()
        self.statusBar().addWidget(self._statusResources)
       
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

        def slot(id):
            return lambda checked: self.panel(id, create=True, show=True)
        for meta in self._panelMetas:
            action = QAction(meta.label, self)
            action.setStatusTip(meta.tooltip)
            action.triggered.connect(slot(meta.id))
            toolsMenu.addAction(action)

    ###########################################################################
    #                        MainWindowController                             #
    ###########################################################################

    # Original idea: Controller for the main GUI window. Will form the
    # base handler for all events and aggregate subcontrollers for
    # individual widgets.

    def onPanelSelected(self, index: int) -> None:
        """Callback for selecting a new panel in the main window.

        Arguments
        ---------
        index: int
            Index of the newly selected panel.
        """
        panel = self._tabs.widget(index)
        panel.attention(False)

    def _saveState(self) -> None:
        """Callback for saving any application state inb4 quitting."""
        pass

    def onExitClicked(self) -> None:
        """Callback for clicking the exit button. This will save state and
        then terminate the Qt application.

        """
        self._saveState()
        QCoreApplication.quit()

    def closeEvent(self, event) -> None:
        """Callback for x button click."""
        self.onExitClicked()

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

    ##########################################################################
    #                          Public Interface                              #
    ##########################################################################

    def getRunner(self) -> Runner:
        """Get the runner set up by this MainWindow. This will
        be an instance of a special Qt Runner that can manage
        inter thread communication with Qt threads.

        Result
        ------
        runner:
            The runner set up by this :py:class:`DeepVisMainWindow`.
        """
        return self._runner

    def panel(self, panel_id: str,
              create: bool=False, show: bool=False) -> Panel:
        """Get the panel for a given panel identifier. Optionally
        create that panel if it does not yet exist.

        Arguments
        ---------
        panel_id:
            The panel identifier. This must be one of the identifiers
            from the _panelMetas list.
        create:
            A flag indicating if the panel should be created in case
            it does not exist yet.
        show:
            A flag indicatin if the panel should be shown, i.e., made
            the current panel.

        Raises
        ------
        ValueError:
            The given panel identifier is not known.
        """
        print(f"!!! panel({panel_id}, create={create}, show={show})")
        meta = self._meta(panel_id)
        panel = None
        if create:
            index = 0
            label = self._tabs.tabText(index)
            for m in self._panelMetas:
                print(f"!!! panel({panel_id}): {m}, index={index}, '{self._tabs.tabText(index)}'")
                if label == meta.label:
                    panel = self._tabs.widget(index)
                    break
                if m.id == meta.id:
                    panel = self._newPanel(meta)
                    self._tabs.insertTab(index, panel, m.label)
                    break
                if label == m.label:
                    index += 1
                    label = self._tabs.tabText(index)
        else:
            for index in range(self._tabs.count()):
                if self._tabs.tabText(index) == meta.label:
                    panel = self._tabs.widget(index)
                    break
        print(f"!!! panel({panel_id}): index={index}, panel={panel}")
        if show and panel is not None:
            self._tabs.setCurrentIndex(index)
        return panel

    def activateLogging(self, logger: logging.Logger,
                        recorder: util.RecorderHandler=None,
                        show: bool=False) -> None:
        logging = self.panel('logging', create=True, show=True)
        if logging is not None:  # can only fail if create=False
            if recorder is not None:
                logging.setLoggingRecorder(recorder)
            logging.addLogger(logger)

    ###########################################################################
    #                               Panels                                    #
    ###########################################################################

    def _meta(self, panel_id: str):
        try:
            return next(filter(lambda m: m.id == panel_id, self._panelMetas))
        except StopIteration:
            raise ValueError(f"'{self}' does not know a panel identified "
                             f"by '{panel_id}'. Known panels are: '" +
                             "', '".join([m.id for m in self._panelMetas]) +
                             "'")

    def _newPanel(self, meta):
        package, name = meta.cls.rsplit('.', 1)
        if package[0] == '.':  # relative import
            package = __name__.rsplit('.', 1)[0] + package
        module = importlib.import_module(package)
        cls = getattr(module, name)
        if hasattr(type(self), '_new' + name):
            panel = getattr(type(self), '_new' + name)(self, cls)
        else:
            panel = cls()
            if hasattr(type(self), '_init' + name):
                getattr(type(self), '_init' + name)(self, panel)
        return panel

    def _newAutoencoderPanel(self, AutoencoderPanel: type) -> Panel:
        autoencoder = self._toolboxController.autoencoder_controller
        training = self._toolboxController.training_controller
        return AutoencoderPanel(toolboxController=self._toolboxController,
                                autoencoderController=autoencoder,
                                trainingController=training)

    def _initMaximizationPanel(self, maximization: Panel) -> None:
        """Initialise the activation maximization panel.
        """
        engineController = self._toolboxController.maximization_engine
        maximization.setController(engineController,
                                   MaximizationEngineObserver)

    def _initLoggingPanel(self, loggingPanel: Panel) -> None:
        """Initialise the log panel.

        """
        loggingPanel.addLogger(logging.getLogger())

    def _newResourcesPanel(self, ResourcesPanel: type) -> Panel:
        autoencoder = self._toolboxController.autoencoder_controller
        datasource = self._toolboxController.datasource_controller
        return ResourcesPanel(toolbox=self._toolboxController,
                              network1=autoencoder,
                              datasource1=datasource)

    ##########################################################################
    #                           FIXME[old]                                   #
    ##########################################################################

    def setModel(self, model: Model) -> None:

        self._activations_controller = \
            ActivationsController(model, runner=self._runner)
        self._datasource_controller = \
            DatasourceController(model, runner=self._runner)

        activationsPanel = self.panel('activations')
        if activationsPanel is not None:
            activationsPanel.setController(activations_controller,
                                           ModelObserver)
            activationsPanel.setController(self._datasource_controller,
                                           Datasource.Observer)

        maximizationPanel = self.panel('maximization')
        if maximizationPanel is not None:
            maximizationPanel.setController(self._activations_controller,
                                            ModelObserver)

    def setLucidEngine(self, engine:'LucidEngine'=None) -> None:
        lucidPanel = self.panel('lucid')
        if lucidPanel is not None:
            from controller import LucidController
            controller = LucidController(engine, runner=self._runner)
            lucidPanel.setController(controller)

    def setDatasource(self, datasource: Datasource) -> None:
        """Set the datasource.
        """
        self._datasource_controller.set_datasource(datasource)

        activationsPanel = self.panel('activations')
        if activationsPanel is not None:
            activationsPanel.setController(self._datasource_controller,
                                           Datasource.Observer)


    def _oldCreateTabWidget(self):
        if self._activations is not None:
            self._tabs.addTab(self._activations, 'Activations')
        if self._maximization is not None:
            self._tabs.addTab(self._maximization, 'Maximization')
        if self._lucid is not None:
            self._tabs.addTab(self._lucid, 'Lucid')
        if self._adversarial_example is not None:
            self._tabs.addTab(self._adversarial_example, 'Adv. Examples')
        if self._internals is not None:
            self._tabs.addTab(self._internals, 'Internals')
        if self._logging is not None:
            self._tabs.addTab(self._logging, 'Logging')
