"""
File: mainwindow.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
"""

# Generic imports
from typing import Tuple, Iterable
import sys
import time
import logging
import collections
import importlib

# Qt imports
from PyQt5.QtCore import (Qt, QTimer, QThread, QTimerEvent,
                          pyqtSignal, pyqtSlot)
from PyQt5.QtGui import (QIcon, QDragEnterEvent, QDropEvent,
                         QCloseEvent, QHideEvent, QShowEvent)
from PyQt5.QtWidgets import (QMenu, QAction, QMainWindow,
                             QTabWidget, QLabel, QApplication,
                             QSystemTrayIcon)

# Toolbox imports
from base import Runner
from toolbox import Toolbox
from network import Network

# FIXME[old]: this should not be needed for the MainWindow
from util import resources
from datasource import Datasource

# Toolbox GUI imports
from .utils import QtAsyncRunner, QObserver, QBusyWidget, QDebug, protect
from .panels import Panel

# logging
LOG = logging.getLogger(__name__)

# FIXME[bug]: statusBar-bug:
# There seems to be a problem with QStatusBar: creating a QStatusBar
# will cause the CPU load to go up to 100% on all CPUs!  (strange
# enough, atop reports that firefox uses all this CPU%).
#
# How to reproduce: start ipython and type
#
#   from PyQt5.QtWidgets import QStatusBar, QApplication
#   app = QApplication([])
#   bar = QStatusBar()     # <- after this CPU load will go up
#
# The same effect can be observed when obtaining a statusBar from
# QMainWindow.statusBar(). However, the same also happens when commenting
# out all invocations of self.statusBar() in DeepVisMainWindow, but
# then it does not occur upon construction, but when calling mainWindow.show()
#
#    from qtgui import create_gui
#    gui = create_gui([], None)   # <- invocation of self.statusBar() commented
#    gui.show()                   # <- after this CPU load will go up
#
# but showing a QMainWindow (without Menu) does NOT cause the problem:
#
#   from PyQt5.QtWidgets import QMainWindow, QApplication
#   app = QApplication([])
#   gui = QMainWindow()
#   gui.show()                    # <- no problem here
#
# Currently I have no idea where this problem originates, or how it
# may be fixed - no suitable references found on the internet.
#
# Analysis: the bug only occurs when my firefox (73.0.1, 64-bit on an
# Ubuntu 18.04 on my home system) is running - it does not occur with
# the same version of firefox running under Ubuntu 16.04 at the university).
#  * if no Firefox is is not running the problem does not occur
#  * if my Firefox (with many tabs) is running, the full problem occurs
#  * if running a Firefox private browsing window (with one tab),
#    the problem occurs at reduced scale (3 of 4 CPUs get 100% load)
#  * if running in Safe mode (with many tabs) the problem occurs
#    at full scale.
#  * if creating and running a new (fresh) Firefox profile, the problem
#    occurs at full scale.
#
# Set the following flag (BUG) to True, if the bug occurs: it will avoid
# all GUI features (menu, statusBar), that result in using up all CPU.
# However, some graphical features will be missing ...
#
# from qtgui import mainwindow
# mainwindow.BUG = True
BUG = False


class DeepVisMainWindow(QMainWindow, QObserver, QDebug, qobservables={
        Toolbox: {'datasources_changed', 'datasource_changed',
                  'server_changed', 'shell_changed'}}):
    """The main window of the Deep Visualization Toolbox. The window
    is intended to hold different panels that allow for different
    kinds of visualizations.

    This class also provides central entry points to programmatically
    set certain aspects, like loading a network or input data,
    switching between different panels, etc.

    Attributes
    ----------

    **Class attributes:**

    PanelMeta: type
        A named tuple (id, label, cls, tooltip, addons). 'id' is a unique
        identifier for this panel and can be used to create or
        access that panel via the :py:meth:`panel` method.

    _panelMetas: list
        A list of PanelMeta entries supported by this MainWindow.

    **Instance attributes:**

    _application: QApplication
        The Qt application to which this window belongs
    _toolbox: Toolbox
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

    _timerId: int
        The ID of a :py:class:`QTimer` that is run to regularly update the
        interface, e.g., the status bar. The timer is started once
        the window gets visible and is stopped if the hidden Window
        becomes hidden. The resulting :py:class:`QTimerEvent` are handled
        by :py:meth:`timerEvent`.
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
        PanelMeta('segmentation', 'Segmentation',
                  '.panels.segmentation.SegmentationPanel',
                  'Show the segmentation panel', []),
        #  FIXME[todo]: if addons.use('lucid'):
        PanelMeta('lucid', 'Lucid',
                  '.panels.lucid.LucidPanel',
                  'Show the Lucid panel', ['lucid']),
        PanelMeta('styletransfer', 'Style transfer',
                  '.panels.styletransfer.StyletransferPanel',
                  'Show the style transfer panel', []),
        # FIXME[todo]: if addons.use('advexample'):
        PanelMeta('advexample', 'Adv. Examples',
                  '.panels.advexample.AdversarialExamplePanel',
                  'Show the adversarial example panel', ['advexample']),
        PanelMeta('occlusion', 'Occlusion',
                  '.panels.occlusion.OcclusionPanel',
                  'Show the occlusion panel', []),
        PanelMeta('resources', 'Resources',
                  '.panels.resources.ResourcesPanel',
                  'Show the resources panel', []),
        PanelMeta('face', 'Face detection',
                  '.panels.face.FacePanel',
                  'Show the Experiments panel', ['dlib', 'imutils']),
        PanelMeta('sound', 'Sound',
                  '.panels.sound.SoundPanel',
                  'Show the Ikkuna panel', ['ikkuna']),
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

    _toolbox: Toolbox = None
    _datasourceMenu = None

    #
    # Signals
    #

    # signals must be declared outside the constructor, for some weird reason

    # Signal emitted once the computation is done.
    # The signal is connected to :py:meth:`panel` which will be run
    # in the main thread by the Qt magic.
    panel_signal: pyqtSignal = pyqtSignal(str, bool, bool)

    # Signal is emitted once the import of a (Panel) module has
    # finished and the Panel is ready for instatiation.
    _panel_import_signal = pyqtSignal(str)  # panel_id

    # FIXME[problem]: currently this prints the following message
    # (when calling MainWindow.setWindowIcon('assets/logo.png')):
    # "libpng warning: iCCP: extra compressed data"
    # probably due to some problem with the file 'logo.png'
    # icon = 'assets/logo.png'
    def __init__(self, application: QApplication, toolbox: Toolbox,
                 title: str = 'Deep Learning ToolBox',
                 icon: str = 'assets/logo.png',
                 focus: bool = True) -> None:
        """Initialize the main window.

        Parameters
        ----------
        title: str
            Window title.
        icon: str
            (Filename of) the Window icon.
        focus: bool
            A flag indicating if the GUI (MainWindow) should "steal"
            the focus. This is the default behaviour, but may be
            undesirable if there is a shell continuing to accepting input.
        """
        super().__init__()
        self._app = application
        self._timerId = None
        self._timerIsRunning = False

        # FIXME[hack]: a dictionary mapping datasource keys (str) to
        # QActions selecting that datasource in the toolbox
        self._datasources = {}

        self._initActions()
        self.setToolbox(toolbox)
        self._runner = QtAsyncRunner()
        self._initUI(title, icon)

        if not focus:
            # Starting from Qt 4.4.0, the attribute
            # Qt.WA_ShowWithoutActivating forces the window not to
            # become activate (even if the Qt.WindowStaysOnTopHint
            # flag is set.
            self.setAttribute(Qt.WA_ShowWithoutActivating)

        # signals
        self.panel_signal.connect(self.panel)
        self._panel_import_signal.connect(self._onPanelImported)

    def run(self, **kwargs):
        """Run the apllication by starting its main event loop.  This function
        will only return once that event loop is finished (usually as
        a result of calling :py:meth:`stop`).

        """
        # FIXME[bug]: the following line will cause the statusBar-bug
        # (but only if the menu was installed via self._initMenu()).
        # However, it seems that no QStatusBar is involved (at least
        # self.statusBar() seems not to be invoked)
        self.show()

        # Initialize the Toolbox in the background, while the (event
        # loop of the) GUI is already started in the foreground.
        if self._toolbox is not None:
            self._runner.runTask(self._toolbox.setup, **kwargs)

        # This function will only return once the main event loop is
        # finished.
        result = self._app.exec_()

        # The following command is intended to avoid some QThread messages
        # in the shell on exit
        self._app.deleteLater()

        # Here we have a chance to cancel all running tasks
        # avoid QThread/QTimer error messages on exit
        # ...

        self.hide()
        return result

    def stop(self):
        """Quit the application.
        """
        self._saveState()

        # Stop the Qt main event loop of this application
        # QCoreApplication.quit()
        self._app.quit()

    @protect
    def showEvent(self, event: QShowEvent) -> None:
        LOG.info("DeepVisMainWindow will show")
        if self._timerId is None:
            self._timerId = self.startTimer(1000)

    @protect
    def hideEvent(self, event: QHideEvent) -> None:
        LOG.info("DeepVisMainWindow was hidden")
        if self._timerId is not None:
            self.killTimer(self._timerId)
            self._timerId = None

    @protect
    def timerEvent(self, event: QTimerEvent) -> None:
        self.showStatusResources()

    ##########################################################################
    #                          Public Interface                              #
    ##########################################################################

    def toolbox_changed(self, toolbox: Toolbox, info: Toolbox.Change) -> None:
        if info.datasources_changed and self._datasourceMenu is not None:
            self._updateDatasourceMenu()
        elif info.datasource_changed and self._datasourceMenu is not None:
            self._updateDatasourceMenu()
        elif info.server_changed:
            self.updataServerActions()
        elif info.shell_changed:
            self.updataShellActions()

    def _updateDatasourceMenu(self) -> None:
        """Update the "Datasource" menu.
        """
        # self._datasourceMenu is of type PyQt5.QtWidgets.QMenu'
        if self._toolbox is None:
            return  # FIXME[todo]: erase all Datasources from the menu

        def slot(id, datasource):
            def setDatasource(checked):
                LOG.info("qtgui.MainWindow.setDatasource(%s): %s [%s]",
                         checked, id, type(datasource))
                # We first set the Datasource in the toolbox
                # and then prepare it, in order to get a smooth
                # reaction in the user interface.
                self._toolbox.datasource = datasource
                # FIXME[bug]: Object is currently busy.
                # datasource.prepare()
            return protect(setDatasource)

        # Add all datasource to the menu
        # FIXME[bug]: only add datasources currently missing!
        # FIXME[todo]: provide a datasourceSelected signal,
        #  that provides the key for the datasource
        for datasource in self._toolbox.datasources:
            key = datasource.key
            label = str(datasource)
            if key in self._datasources:
                continue
            action = QAction(label, self)
            if True:  # datasource == self._toolbox.datasource:
                # FIXME[bug]: seems to have no effect
                action.font().setBold(True)
            action.triggered.connect(slot(id, datasource))
            self._datasourceMenu.addAction(action)
            self._datasources[key] = action

    def getRunner(self) -> Runner:
        """Get the runner set up by this MainWindow. This will
        be an instance of a special Qt Runner that can manage
        inter thread communication with Qt threads.

        Returns
        ------
        runner:
            The runner set up by this :py:class:`DeepVisMainWindow`.
        """
        return self._runner

    def panels(self) -> Iterable[str]:
        """An iterator of panel identifiers.
        """
        for panelMeta in self._panelMetas:
            yield panelMeta.id

    def panel(self, panel_id: str,
              create: bool = False, show: bool = False) -> Panel:
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
        KeyError:
            The given panel identifier is not known.
        ValueError:
            The panel requested has not yet been created and no creation
            was requested.
        """
        if not any(m.id == panel_id for m in self._panelMetas):
            raise KeyError(f"There is no panel with id '{panel_id}' "
                           "in DeepVisMainWindow.")

        if QThread.currentThread() != self.thread():
            # method was called from different thread
            # emit signal to run in main thread
            self.panel_signal.emit(panel_id, create, show)
            return

        meta = self._panelMeta(panel_id)
        panel = None
        if create:
            index = 0
            label = self._tabs.tabText(index)
            for m in self._panelMetas:
                if label == meta.label:
                    panel = self._tabs.widget(index)
                    break
                if m.id == meta.id:
                    self._newPanel(meta, index, m.label, show)
                    # FIXME[hack]: panel creation is asynchronous and
                    # hence does not really fit in this synchronous
                    # function! API should be changed accordingly!
                    panel = None
                    break
                if label == m.label:
                    index += 1
                    label = self._tabs.tabText(index)
        else:
            index = self._panelTabIndex(panel_id)
            panel = self._tabs.widget(index)
        if show and panel is not None:
            self._tabs.setCurrentIndex(index)
        return panel

    ###########################################################################
    #                           Initialization                                #
    ###########################################################################

    # FIXME[bug]: debug the statusBar-Bug (see above)
    def statusBar(self):
        LOG.debug("DeepVisMainWindow[debug]: statusBar() was invoked.")
        return super().statusBar()

    def _initUI(self, title: str, icon: str) -> None:
        """Initialize the graphical components of this user interface."""
        #
        # Initialize the Window
        #
        self.setWindowTitle(title)
        self.setWindowIcon(QIcon(icon))
        self.setAcceptDrops(True)

        desktop = self._app.desktop()  # of type QDesktopWidget
        screenGeometry = desktop.screenGeometry()  # of type QRect
        self.setMinimumSize(screenGeometry.width() // 2,
                            screenGeometry.height() // 2)

        #
        # Initialize the Tabs
        #
        self._tabs = QTabWidget(self)
        self._tabs.currentChanged.connect(self.onPanelSelected)
        self.setCentralWidget(self._tabs)

        #
        # Create the menu
        #
        # FIXME[bug]: with the following line the statusBar-bug (see above)
        # occurs when calling self.show()
        if not BUG:
            self._initMenu()

        #
        # Initialize the status bar
        #
        # FIXME[bug]: the following line directly causes the
        # statusBar-bug (see above)
        if not BUG:
            self._statusResources = QLabel()
            self.statusBar().addWidget(self._statusResources)
        else:
            self._statusResources = None

        #
        # Initialize the system tray
        #
        if not BUG:
            self._initSystemTray()

    def _initMenu(self):
        menubar = self.menuBar()

        #
        # Add exit menu
        #
        exitAction = QAction(QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcuts(['Ctrl+W', 'Ctrl+Q'])
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.onExitClicked)

        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(self.actionShell)
        fileMenu.addSeparator()
        fileMenu.addAction(self.actionServerStart)
        fileMenu.addAction(self.actionServerStop)
        fileMenu.addAction(self.actionServerOpen)
        fileMenu.addSeparator()
        fileMenu.addAction(exitAction)

        #
        # Network Menu
        #
        networkMenu = menubar.addMenu('&Network')
        networkMenu.addAction(self.actionAlexnet)

        #
        # Datasource Menu
        #
        self._datasourceMenu = menubar.addMenu('&Data')
        self._datasourceMenu.setTearOffEnabled(True)
        self._updateDatasourceMenu()

        #
        # Tools Menu
        #
        toolsMenu = menubar.addMenu('&Tools')
        toolsMenu.setTearOffEnabled(True)

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
        if isinstance(panel, Panel):
            # FIXME[hack]: during initialization the panel can be some
            # dummy widget ... maybe we can make the dummy panel a real panel
            panel.attention(False)

    def _saveState(self) -> None:
        """Callback for saving any application state inb4 quitting."""
        pass

    def showStatusMessage(self, message, timeout: int = 2000):
        """Hide the normal status indications and display the given message in
        the statusbar for the specified number of milli-seconds .

        """
        # If timeout is 0 (default), the message remains displayed
        # until the clearMessage() slot is called or until the
        # showMessage() slot is called again to change the message.
        # FIXME[bug]: commented out due to the statusBar-bug (see above)
        if not BUG:
            self.statusBar().showMessage(message, timeout)

    @protect
    def showStatusResources(self):
        if self._statusResources is None:
            return
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
        if self._runner is not None:
            message += (f", Tasks: {self._runner.active_workers}/"
                        f"{self._runner.max_workers}")
        self._statusResources.setText(message)

    ###########################################################################
    #                       Handler for Qt Events                             #
    ###########################################################################

    @protect
    def closeEvent(self, event: QCloseEvent) -> None:
        """Callback for [x] button click."""
        LOG.info("MainWindow: Window button"
                 " -> exiting the main window now ...")
        self._toolbox.quit()
        LOG.info("MainWindow: toolbox.quit() returned ...")

    @protect
    def onExitClicked(self, checked: bool) -> None:
        """Callback for clicking the exit button. This will save state and
        then terminate the Qt application.

        """
        LOG.info("MainWindow: Menu/exit -> exiting the main window now ...")
        if self._toolbox is None:
            self.stop()
        else:
            # FIXME[concept]: this branch seems unnecessary:
            # It should be enough to just stop the
            # Qt main event loop and let run() return to its caller,
            # which may then do remaining cleanup
            self._toolbox.quit()
        LOG.info("MainWindow: toolbox.quit() returned ...")

    ###########################################################################
    #                           Drag and Drop                                 #
    ###########################################################################

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
            LOG.info("Drag and dropped %d URLs, took first: %s",
                     len(mimeData.urls()), url)

            # 2. Convert the URL to a local filename
            filename = url.toLocalFile()
            description = f"Dropped in image ({filename})"
            LOG.info("Converted URL to local filename: '%s'", filename)

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
    #                             System Tray                                 #
    ###########################################################################

    # FIXME[experiment]: add a system tray icon
    # just for test, no real function yet,
    # FIXME[bug]: does not seem to fully work
    # only signal 3 (QSystemTrayIcon.Trigger) is received upon double click
    # and onSystemTrayMenuAboutToShow is received only on the
    # first click, but no menu is shown.

    def _initSystemTray(self, withMenu: bool = False):
        LOG.info("Initializing the system tray: available: %s, messages: %s",
                 QSystemTrayIcon.isSystemTrayAvailable(),
                 QSystemTrayIcon.supportsMessages())

        if not QSystemTrayIcon.isSystemTrayAvailable():
            return

        systemTrayIcon = QSystemTrayIcon(QIcon('assets/logo.png'), self._app)
        systemTrayIcon.setToolTip("The Deep Visualization Toolbox")
        systemTrayIcon.activated.connect(self.onSystemTrayActivated)

        if withMenu:
            contextMenu = QMenu()
            contextMenu.aboutToShow.connect(self.onSystemTrayMenuAboutToShow)

            actionTest = QAction('Test', contextMenu)
            actionTest.triggered.connect(self.onSystemTrayMenuActionTriggered)
            systemTrayIcon.setContextMenu(contextMenu)

        if QSystemTrayIcon.supportsMessages():
            systemTrayIcon.\
                messageClicked.connect(self.onSystemTrayMessageClicked)

        systemTrayIcon.show()

    # @pyqtSlot()
    def onSystemTrayActivated(self, signal: int):
        if signal == QSystemTrayIcon.Unknown:
            text = "Unknown reason"
        elif signal == QSystemTrayIcon.Context:
            text = "The context menu for the system tray entry was requested"
        elif signal == QSystemTrayIcon.DoubleClick:
            text = "The system tray entry was double clicked"
        elif signal == QSystemTrayIcon.Trigger:
            text = "The system tray entry was clicked"
        elif signal == QSystemTrayIcon.MiddleClick:
            text = ("The system tray entry was clicked "
                    "with the middle mouse button")
        else:
            text = "Illegal value"
        LOG.info("onSystemTrayActivated: %s (%s)", signal, text)

    # @pyqtSlot()
    def onSystemTrayMenuAboutToShow(self):
        LOG.info("onSystemTrayMenuAboutToShow")

    # @pyqtSlot()
    def onSystemTrayMenuActionTriggered(self, *args, **kwargs):
        LOG.info("onSystemTrayMenuActionTriggered: %r, %r", args, kwargs)

    # @pyqtSlot()
    def onSystemTrayMessageClicked(self):
        """React when the message displayed in the system tray using
        showMessage() was clicked by the user.
        """
        LOG.info("onSystemTrayMessageClicked.")

    ###########################################################################
    #                               Panels                                    #
    ###########################################################################

    def _panelMeta(self, panel_id: str) -> 'PanelMeta':
        """Get the :py:class:`PanelMeta` for a given panel id.

        Arguments
        ---------
        panel_id: str
            The panel identifier

        Raises
        ------
        KeyError:
            The panel_id given is not a valid panel identifier.
        """
        try:
            return next(filter(lambda m: m.id == panel_id, self._panelMetas))
        except StopIteration:
            raise KeyError(f"'{self}' does not know a panel identified "
                           f"by '{panel_id}'. Known panels are: " +
                           ", ".join([f"'{m.id}'"
                                      for m in self._panelMetas]))

    def _panelTabIndex(self, panel_id: str) -> int:
        """Get the index of a (created) panel in the panel tab widget.
        """
        meta = self._panelMeta(panel_id)
        for index in range(self._tabs.count()):
            if self._tabs.tabText(index) == meta.label:
                return index
        raise ValueError(f"Panel with id '{panel_id}' "
                         "has not been created yet.")

    def _panelModuleAndClass(self, panel_id: str) -> Tuple[str, str]:
        """Get module name and class name for the Panel with the given panel
        identifier.  These information will be obtained from the panel
        metadata register.
        """
        meta = self._panelMeta(panel_id)
        module_name, class_name = meta.cls.rsplit('.', 1)
        if module_name[0] == '.':  # relative import
            module_name = __name__.rsplit('.', 1)[0] + module_name
        return module_name, class_name

    def _newPanel(self, meta, index, label, show: bool = True):
        """Create a new panel from panel metadata.

        The creation will be split into multiple phases to allow for a
        smooth user experience:
        (0) Adding a new "dummy" tab  for new the Panel. This
            should happen quickly (and has to be done in the main thread)
            for responsive GUI behaviour. The dummy widget may display
            some progress notification to the user.
        (1) Import of the Panel's module. This may take some time,
            as it may imply the import of other (external) modules.
            This can be done in a background thread.
        (2) Instantiation of the new panel class. This has to be done
            in the main thread. After instantiation, the dummy tab
            can be replace by the actual panel.
        (3) Set observables of the panel.

        """
        LOG.info("Inserting new Panel '%s' at index %d.", meta.id, index)
        dummy = QBusyWidget()
        self._tabs.insertTab(index, dummy, label)
        if show:
            self._tabs.setCurrentIndex(index)

        LOG.debug("Running import %s for Panel '%s'.", meta.cls, meta.id)
        self._runner.runTask(self._importPanel, meta.id)

    def _importPanel(self, panel_id: str) -> None:
        module_name, class_name = self._panelModuleAndClass(panel_id)
        LOG.debug("Importing module '%s' for panel '%s'.",
                  module_name, panel_id)
        module = importlib.import_module(module_name)
        LOG.debug("Signaling sucessful imports for panel '%s'.", panel_id)
        self._panel_import_signal.emit(panel_id)

    @pyqtSlot(str)
    @protect
    def _onPanelImported(self, panel_id: str):
        module_name, class_name = self._panelModuleAndClass(panel_id)
        LOG.debug("Instantiating class %s from module %s for panel '%s'",
                  class_name, module_name, panel_id)
        module = sys.modules[module_name]
        cls = getattr(module, class_name)
        if hasattr(type(self), '_new' + class_name):
            panel = getattr(type(self), '_new' + class_name)(self, cls)
        else:
            panel = cls()
            if hasattr(type(self), '_init' + class_name):
                getattr(type(self), '_init' + class_name)(self, panel)

        # replace the dummy content by the actual pane
        LOG.debug("Setting new instance of %s as content for panel '%s'.",
                  class_name, panel_id)
        index = self._panelTabIndex(panel_id)
        meta = self._panelMeta(panel_id)
        dummy = self._tabs.widget(index)
        # self._tabs.replaceWidget(dummy, panel)
        currentIndex = self._tabs.currentIndex()
        self._tabs.removeTab(index)
        self._tabs.insertTab(index, panel, meta.label)
        self._tabs.setCurrentIndex(currentIndex)
        dummy.deleteLater()
        LOG.debug("Creation of panel '%s' (index=%d) finished.",
                  panel_id, index)

    def _newAutoencoderPanel(self, AutoencoderPanel: type) -> Panel:
        autoencoder = self._toolbox.autoencoder_controller
        training = self._toolbox.training_controller
        return AutoencoderPanel(toolbox=self._toolbox,
                                autoencoderController=autoencoder,
                                trainingController=training)

    def _newActivationsPanel(self, ActivationsPanel: type) -> Panel:
        # FIXME[old]
        # activation_tool = self._toolbox.add_tool('activation')
        # network = self._toolbox.autoencoder_controller
        network = None
        return ActivationsPanel(toolbox=self._toolbox, network=network,
                                datasource=self._toolbox.datasource)

    def _newSegmentationPanel(self, SegmentationPanel: type) -> Panel:
        return SegmentationPanel(toolbox=self._toolbox)

    def _initMaximizationPanel(self, maximization: Panel) -> None:
        """Initialise the activation maximization panel.
        """
        networkController = self._toolbox.autoencoder_controller  # FIXME[hack]
        maximizationController = self._toolbox.maximization_engine
        maximization.setToolbox(self._toolbox)
        maximization.setMaximizationController(maximizationController)
        maximization.setNetworkController(networkController)

    def _initLoggingPanel(self, loggingPanel: Panel) -> None:
        """Initialise the log panel.

        """
        loggingPanel.setToolbox(self._toolbox)
        loggingPanel.addLogger(logging.getLogger())

    def _initInternalsPanel(self, internalsPanel: Panel) -> None:
        """Initialise the internalspanel.
        """
        internalsPanel.setToolbox(self._toolbox)

    def _newResourcesPanel(self, ResourcesPanel: type) -> Panel:
        return ResourcesPanel(toolbox=self._toolbox)

    def _newFacePanel(self, FacePanel: type) -> Panel:
        return FacePanel(toolbox=self._toolbox)

    def _newAdversarialExamplePanel(self,
                                    AdversarialExamplePanel: type) -> Panel:
        return AdversarialExamplePanel(toolbox=self._toolbox)

    ##########################################################################
    #                             Actions                                    #
    ##########################################################################

    def _initActions(self):
        # triggered: This signal is emitted when an action is
        # activated by the user; for example, when the user clicks a
        # menu option, toolbar button, or presses an action's shortcut
        # key combination, or when trigger() was called. Notably, it
        # is not emitted when setChecked() or toggle() is called.

        # exitAction.setShortcuts(['Ctrl+S', 'Ctrl+Q'])
        action = QAction('Start Server', self)
        action.setStatusTip('Start a HTTP server')
        action.triggered.connect(self.onServerStartTriggered)
        self.actionServerStart = action

        action = QAction('Stop Server', self)
        action.setStatusTip('Stop the HTTP server')
        action.triggered.connect(self.onServerStopTriggered)
        self.actionServerStop = action

        action = QAction('Server open', self)
        action.setStatusTip('Open Server in a Web Browser')
        action.triggered.connect(self.onServerOpenTriggered)
        self.actionServerOpen = action

        action = QAction('Shell', self)
        action.setStatusTip('Open Toolbox shell in the Terminal')
        action.triggered.connect(self.onShellTriggered)
        self.actionShell = action

        action = QAction('AlexNet', self)
        action.setStatusTip('Load AlexNet')
        action.triggered.connect(self.onAlexnetTriggered)
        self.actionAlexnet = action

    @protect
    def onServerStartTriggered(self, checked: bool) -> None:
        if self._toolbox:
            self._toolbox.server = True

    @protect
    def onServerStopTriggered(self, checked: bool) -> None:
        if self._toolbox:
            self._toolbox.server = False

    @protect
    def onServerOpenTriggered(self, checked: bool) -> None:
        if self._toolbox:
            self._toolbox.server_open()

    def updataServerActions(self) -> None:
        enabled = self._toolbox is not None
        server_running = enabled and self._toolbox.server
        self.actionServerStart.setEnabled(enabled and not server_running)
        self.actionServerStop.setEnabled(server_running)
        self.actionServerOpen.setEnabled(server_running)

    @protect
    def onShellTriggered(self, checked: bool) -> None:
        if self._toolbox:
            self._toolbox.run_shell()

    def updataShellActions(self) -> None:
        self.actionShell.setEnabled(self._toolbox is not None and
                                    not self._toolbox.shell)

    @protect
    def onAlexnetTriggered(self, checked: bool) -> None:
        Network.instance_register['alexnet-tf'].initialize

    ##########################################################################
    #                           FIXME[old]                                   #
    ##########################################################################

    def setLucidEngine(self, engine: 'LucidEngine' = None) -> None:
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
        # self._toolbox.datasource = datasource

        activationsPanel = self.panel('activations')
        if activationsPanel is not None:
            activationsPanel.setController(self._toolbox.datasource,
                                           Datasource.Observer)

    # FIXME[old]: the following timer is thread safe in the sense that
    # it uses one-shot timers and it will start the next timer only
    # after the last operation finishes. This prevents overlapping
    # processing of timer events. Maybe this is a bit safer than our
    # simple implementation, but it is also a bit more complicated.
    # Maybe safe this somewhere for later use.
    def startTimer(self, timeout: int = 1000) -> None:
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
                LOG.info("GUI: QTimer was stopped.")

        self._timerIsRunning = True
        QTimer.singleShot(timeout, timer_event)

    def stopTimer(self):
        """
        """
        LOG.info("GUI: Stopping the QTimer ...")
        self._timerIsRunning = False
