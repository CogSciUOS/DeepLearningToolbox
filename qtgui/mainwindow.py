"""
File: mainwindow.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
"""


# Generic imports
import time
import logging
import collections
import importlib

# Toolbox imports
from base import Runner
from toolbox import ToolboxController


# Toolbox GUI imports
from qtgui.utils import QtAsyncRunner, protect
from qtgui.panels import Panel

# FIXME[old]: this should not be needed for the MainWindow
import util
from util import resources, addons
from tools.activation import Engine as ActivationEngine

from tools.am import Engine as MaximizationEngine  # FIXME[old]
from tools.am import Controller as MaximizationController
from datasources import Datasource, Controller as DatasourceController

# Qt imports
from PyQt5.QtCore import Qt, QTimer, QCoreApplication
from PyQt5.QtGui import (QPixmap, QIcon, QDragEnterEvent, QDropEvent,
                         QCloseEvent)
from PyQt5.QtWidgets import (QAction, QMainWindow, QStatusBar, QTabWidget,
                             QWidget, QLabel, QApplication)

class DeepVisMainWindow(QMainWindow):
    """The main window of the Deep Visualization Toolbox. The window
    is intended to hold different panels that allow for different
    kinds of visualizations.

    This class also provides central entry points to programmatically
    set certain aspects, like loading a network or input data,
    switching between different panels, etc.

    Class Attributes
    ----------------
    PanelMeta: type
        A named tuple (id, label, cls, tooltip, addons). 'id' is a unique
        identifier for this panel and can be used to create or
        access that panel via the :py:meth:`panel` method.
    
    _panelMetas: list
        A list of PanelMeta entries supported by this MainWindow.

    Attributes
    ----------
    _application: QApplication
        The Qt application to which this window belongs
    _toolbox: ToolboxController
        A reference to the Toolbox operated by this MainWindow. 
    _runner: QtAsyncRunner
        A dedicated :py:class:`Runner` for this Window.
    _tabs: QTabWidget
        A tabbed container for the panels displayed by this MainWindow.

    _maximization_controller: MaximizationController
        An MaximizationController for this Application
    _maximization_engine: MaximizationEngine
        The activation maximzation Engine
    _current_panel: Panel
        The currently selected Panel (FIXME: currently not used!)
    """
    PanelMeta = collections.namedtuple('PanelMeta',
                                       'id label cls tooltip addons')
    _panelMetas = [
        PanelMeta('activations', 'Activations',
                  '.panels.activations.ActivationsPanel',
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
        PanelMeta('face', 'Face detection',
                  '.panels.face.FacePanel',
                  'Show the Experiments panel', ['dlib', 'imutils']),
        PanelMeta('ikkuna', 'Ikkuna',
                  '.panels.ikkuna.IkkunaPanel',
                  'Show the Ikkuna panel', ['ikkuna']),
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

    # FIXME[problem]: currently this prints the following message
    # (when calling MainWindow.setWindowIcon('assets/logo.png')):
    # "libpng warning: iCCP: extra compressed data"
    # probably due to some problem with the file 'logo.png'
    #icon = 'assets/logo.png'
    def __init__(self, application: QApplication, toolbox: ToolboxController,
                 title: str='QtPyVis', icon: str='assets/logo.png') -> None:
        """Initialize the main window.

        Parameters
        ----------
        title: str
            Window title.
        icon: str
            (Filename of) the Window icon.
        """
        super().__init__()
        self._app = application
        
        self._logger = logging.getLogger(__name__)
        self._toolbox = toolbox
        self._runner = QtAsyncRunner()
        self._initUI(title, icon)

    def run(self, **kwargs):
        self.start_timer()

        # Initialize the Toolbox in the background, while the (event
        # loop of the) GUI is already started in the foreground.
        self._runner.runTask(self._toolbox.setup, **kwargs)

        # This function will only return once the main event loop is
        # finished.
        return self._app.exec_()

    def stop(self):
        """Quit the application.
        """
        self._saveState()

        # Stop the Qt main event loop of this application
        # QCoreApplication.quit()
        self._app.quit()

    def start_timer(self, timeout: int=1000) -> None:
        """
        Create a Qtimer that is safe against garbage collection
        and overlapping calls.
        See: http://ralsina.me/weblog/posts/BB974.html

        Parameter
        ---------
        timeout: int
            Time interval in millisecond between timer invocations.
        """
        def timer_event():
            if self._timerIsRunning:
                try:
                    self.showStatusResources()
                finally:
                    QTimer.singleShot(timeout, timer_event)
            else:
                print("GUI: QTimer was stopped.")

        self._timerIsRunning = True
        QTimer.singleShot(timeout, timer_event)

    def stop_timer(self):
        """
        """
        print("GUI: Stopping the QTimer ...")
        self._timerIsRunning = False
        
    ##########################################################################
    #                          Public Interface                              #
    ##########################################################################

    def setToolbox(toolbox: ToolboxController) -> None:
        """Set the Toolbox controlled by this MainWindow.

        Arguments
        ---------
        toolbox: ToolboxController
            The ToolboxController.
        """
        self._toolbox = toolbox
        # FIXME[todo]: also inform the Panels that the toolbox was changed.

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
        meta = self._meta(panel_id)
        panel = None
        if create:
            index = 0
            label = self._tabs.tabText(index)
            for m in self._panelMetas:
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
    #                           Initialization                                #
    ###########################################################################

    def _initUI(self, title: str, icon: str) -> None:
        """Initialize the graphical components of this user interface."""
        #
        # Initialize the Window
        #
        self.setWindowTitle(title)
        self.setWindowIcon(QIcon(icon))
        self.setAcceptDrops(True)

        #
        # Initialize the Tabs
        #
        self._tabs = QTabWidget(self)
        self._tabs.currentChanged.connect(self.onPanelSelected)
        self.setCentralWidget(self._tabs)

        #
        # Create the menu
        #
        self._createMenu()
        
        #
        # Initialize the status bar
        #
        self._statusResources = QLabel()
        self.statusBar().addWidget(self._statusResources)

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
            panel = lambda checked: self.panel(id, create=True, show=True)
            return protect(panel)
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

    @protect
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

    def showStatusMessage(self, message):
        self.statusBar().showMessage(message, 2000)

    def showStatusResources(self):
        message = f"{time.ctime()}"
        # message += (f", {self._runner.active_workers}/"
        #             f"{self._runner.max_workers} threads")
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
    #                       Handler for Qt Events                             #
    ###########################################################################

    @protect
    def closeEvent(self, event: QCloseEvent) -> None:
        """Callback for [x] button click."""
        print("MainWindow: Window button -> exiting the main window now ...")
        self._toolbox.quit()
        print("MainWindow: toolbox.quit() returned ...")

    @protect
    def onExitClicked(self, checked: bool) -> None:
        """Callback for clicking the exit button. This will save state and
        then terminate the Qt application.

        """
        print("MainWindow: Menu/exit -> exiting the main window now ...")
        self._toolbox.quit()
        print("MainWindow: toolbox.quit() returned ...")

    @protect
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle Drag and Drop events. The dragEnterEvent is sent to
        check the acceptance of a following drop operation.
        We want to allow dragging in images.
        
        """
        # Images (from the Desktop) are provided as file ULRs
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    @protect
    def dropEvent(self, e: QDropEvent):
        """Handle Drag and Drop events. The :py:class:`QDropEvent`
        performs the actual drop operation.
        
        We support the following operations: if an Image is dropped,
        we will use it as input image for the :py:class:`Toolbox`.
        """
        mimeData = e.mimeData()
        if mimeData.hasUrls() and self._toolbox:
            # 1. We just consider the first URL, even if multiple URLs
            #    are provided.
            url = mimeData.urls()[0]
            self._logger.info(f"Drag and dropped len(mimeData.urls() URLs, "
                              f"URL[0]: {url}")

            # 2. Convert the URL to a local filename
            filename = url.toLocalFile()
            description = f"Dropped in image ({filename})"
            self._logger.info(f"Converted URL to local filename: '{filename}'")

            # 3. Set this file as input image for the toolbox.
            self._toolbox.set_input_from_file(filename,
                                              description=description)

            # 4. Mark the Drop action as accepted (we actually perform
            #    a CopyAction, no matter what was proposed)
            if e.proposedAction() == Qt.CopyAction:
                e.acceptProposedAction()
            else:
                # If you set a drop action that is not one of the
                # possible actions, the drag and drop operation will
                # default to a copy operation.
                e.setDropAction(Qt.CopyAction)
                e.accept()

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
        autoencoder = self._toolbox.autoencoder_controller
        training = self._toolbox.training_controller
        return AutoencoderPanel(toolboxController=self._toolbox,
                                autoencoderController=autoencoder,
                                trainingController=training)

    def _newActivationsPanel(self, ActivationsPanel: type) -> Panel:
        activation_tool = self._toolbox.add_tool('activation')
        network = self._toolbox.autoencoder_controller
        return ActivationsPanel(toolbox=self._toolbox, network=network,
                                activations=activation_tool,
                                datasource=self._toolbox.datasource_controller)

    def _initMaximizationPanel(self, maximization: Panel) -> None:
        """Initialise the activation maximization panel.
        """
        networkController = self._toolbox.autoencoder_controller  # FIXME[hack]
        maximizationController = self._toolbox.maximization_engine
        print(f"_initMaximizationPanel: maximizationController={maximizationController}")
        maximization.setToolboxController(self._toolbox)
        maximization.setMaximizationController(maximizationController)
        maximization.setNetworkController(networkController)

    def _initLoggingPanel(self, loggingPanel: Panel) -> None:
        """Initialise the log panel.

        """
        loggingPanel.addLogger(logging.getLogger())

    def _newResourcesPanel(self, ResourcesPanel: type) -> Panel:
        autoencoder = self._toolbox.autoencoder_controller
        datasource = self._toolbox.datasource_controller
        return ResourcesPanel(toolbox=self._toolbox,
                              network1=autoencoder,
                              datasource1=datasource)

    def _newFacePanel(self, FacePanel: type) -> Panel:
        datasource = self._toolbox.datasource_controller
        return FacePanel(toolbox=self._toolbox, datasource=datasource)

    ##########################################################################
    #                           FIXME[old]                                   #
    ##########################################################################

    def setActivationEngine(self) -> None:
        print("FIXME[old]: MainWindow.setActivationEngine() was called!")
        activationsPanel = self.panel('activations')
        if activationsPanel is not None:
            activationsPanel.setController(self._toolbox.get_tool('activation'),
                                           ActivationEngine.Observer)
            activationsPanel.setController(self._toolbox.datasource_controller,
                                           Datasource.Observer)

        maximizationPanel = self.panel('maximization')
        if maximizationPanel is not None:
            maximizationPanel.setController(self._toolbox.activation_controller,
                                            ActivationEngine.Observer)

    def setLucidEngine(self, engine:'LucidEngine'=None) -> None:
        print("FIXME[old]: MainWindow.setLucidEngine() was called!")
        lucidPanel = self.panel('lucid')
        if lucidPanel is not None:
            from controller import LucidController
            controller = LucidController(engine, runner=self._runner)
            lucidPanel.setController(controller)

    def setDatasource(self, datasource: Datasource) -> None:
        """Set the datasource.
        """
        print("FIXME[old]: MainWindow.setDatasource() was called!")
        self._toolbox.datasource_controller(datasource)

        activationsPanel = self.panel('activations')
        if activationsPanel is not None:
            activationsPanel.setController(self._toolbox.datasource_controller,
                                           Datasource.Observer)
