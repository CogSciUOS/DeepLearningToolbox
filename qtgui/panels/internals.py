"""
File: internals.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
"""

# standard imports
from typing import Callable, Union, Any, Optional, Iterable
from types import ModuleType
import multiprocessing
import importlib
import sys
import os
import re

# Qt imports
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, pyqtSlot, pyqtSignal, QThread, QThreadPool
from PyQt5.QtGui import QFontDatabase, QPaintEvent
from PyQt5.QtWidgets import QWidget, QGroupBox, QLabel, QPushButton
from PyQt5.QtWidgets import QGridLayout, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QPlainTextEdit, QComboBox

# toolbox imports
from dltb.base.info import Info, InfoSource
from dltb.base.package import Package
from dltb.util.hardware import HardwareInfo, SystemInfo, ResourceInfo
from dltb.util.hardware import CudaInfo, CudaDeviceInfo, PythonInfo
from dltb.util.importer import Importer, ImportInterceptor, add_postimport_hook
from util import add_timer_callback
from toolbox import Toolbox

# GUI imports
from .panel import Panel
from ..utils import QObserver, QAttribute, protect, QBusyWidget, QPrepareButton


class InternalsPanel(Panel, QAttribute, qattributes={Toolbox: False}):
    """A Panel displaying system internals.
    May be of interest during development.


    The `InternalsPanel` makes use of the following classes defined
    below:
    * :py:class:`QInfoWidget`
    * :py:class:`QPackageInfo`
    * :py:class:`QPackageList`
    * :py:class:`QProcessInfo`
    * :py:class:`QNvmlInfo`
    
    Attributes
    ----------
    _packages: dict
        A mapping from package names to package information.
        This can be the acutal package (if already loaded),
        or a string describing the state of the package
        ("not loaddd" or "not found"). This information
        is initialized and updated by the method
        :py:meth:_updatePackages.

    _packageName: str = None

    Graphical elements
    ------------------
    _grid: QGridLayout
    _packageGrid: QGridLayout = None

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._packages = {}
        self._packageName = None

        self.initUI()

    def initUI(self):
        self._layout = QVBoxLayout()
        self._packageList = QPackageList()
        self._packageList.packageSelected.connect(self.showPackageInfo)

        self._grid = QGridLayout()
        self._grid.addWidget(self._packageList, 0, 0)
        self._grid.addLayout(self.systemInfo(), 0, 1)

        self._layout.addLayout(self._grid)
        self._info = QLabel("Info")
        self._layout.addWidget(self._info)


        self.setLayout(self._layout)

    #@pyqtSlot
    @protect
    def showPackageInfo(self, package: str) -> None:
        """Show a :py:class:`PackageInfo` widget in the info part
        of this `InternalsPanel`.

        """
        self.showInfo(QPackageInfo(package=Package[package]))

    def showInfo(self, info: QWidget):
        """Show a new info widget.

        Arguments
        ---------
        info: QWidget
           The widget to be displayed in the info region of the
           panel. This will replace the previously displayed
           info widget.
        """
        if self._layout.replaceWidget(self._info, info) is not None:
            self._info.deleteLater()
        self._info = info

    def systemInfo(self):

        #
        # Python
        #

        pythonBox = QGroupBox('Python')
        pythonInfoWidget = QInfoWidget(source=PythonInfo())
        boxLayout = QVBoxLayout()
        boxLayout.addWidget(pythonInfoWidget)
        boxLayout.addStretch()
        pythonBox.setLayout(boxLayout)

        #
        # Hardware
        #

        hardwareBox = QGroupBox('Hardware')
        boxLayout = QVBoxLayout()
        hardwareInfoWidget = QInfoWidget(source=HardwareInfo())
        boxLayout.addWidget(hardwareInfoWidget)
        boxLayout.addStretch()
        hardwareBox.setLayout(boxLayout)

        #
        # Platform
        #
        systemBox = QGroupBox('System')
        boxLayout = QVBoxLayout()
        systemInfoWidget = QInfoWidget(source=SystemInfo())
        boxLayout.addWidget(systemInfoWidget)
        boxLayout.addStretch()
        systemBox.setLayout(boxLayout)

        #
        # Resources
        #
        resourcesBox = QGroupBox('Resources')
        boxLayout = QVBoxLayout()
        resourceInfoWighet = QInfoWidget(source=ResourceInfo())
        boxLayout.addWidget(resourceInfoWighet)
        boxLayout.addStretch()
        resourcesBox.setLayout(boxLayout)

        #
        # CUDA
        #

        cudaBox = QGroupBox('CUDA')
        boxLayout = QVBoxLayout()
        cudaInfo = CudaInfo()
        cudaInfoWighet = QInfoWidget(source=cudaInfo)
        boxLayout.addWidget(cudaInfoWighet)
        deviceInfoWiget = QInfoWidget()
        if cudaInfo.number_of_devices:
            deviceInfoWiget.addInfosFromSource(cudaInfo[0])
        boxLayout.addWidget(deviceInfoWiget)
        boxLayout.addStretch()
        cudaBox.setLayout(boxLayout)

        button = QPushButton("Torch")
        #button.setFlat(True)
        @protect
        def slot(clicked: bool):
            self.showInfo(self.torchInfo())
        button.clicked.connect(slot)
        boxLayout.addWidget(button)

        button = QPushButton("Tensorflow")
        #button.setFlat(True)
        @protect
        def slot(clicked: bool):
            self.showInfo(self.tensorflowInfo())
        button.clicked.connect(slot)
        boxLayout.addWidget(button)

        #
        # CUDA (old)
        #

        cuda2Box = QGroupBox('CUDA2')
        boxLayout = QVBoxLayout()
        try:
            nvmlInfo = QNvmlInfo()
            boxLayout.addWidget(nvmlInfo)
            add_timer_callback(nvmlInfo.update)
        except ImportError as e:
            print(e, file=sys.stderr)
            label = QLabel("Python NVML module (py3nvml) not availabe")
            boxLayout.addWidget(label)
        boxLayout.addStretch()
        cuda2Box.setLayout(boxLayout)

        cuda3Box = QGroupBox('CUDA3')
        layout = self._cudaOld()
        cuda3Box.setLayout(layout)

        #
        # processes
        #
        
        self._processInfo = QProcessInfo()
        self.addAttributePropagation(Toolbox, self._processInfo)

        #
        # layout the boxes
        #
        layout = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(hardwareBox)
        row.addWidget(systemBox)
        row.addWidget(pythonBox)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(resourcesBox)
        row.addWidget(cudaBox)
        row.addWidget(cuda2Box)
        row.addWidget(cuda3Box)
        row.addStretch()
        layout.addLayout(row)

        layout.addWidget(self._processInfo)
        layout.addStretch()
        return layout

    def tensorflowInfo(self):
        tensorflowBox = QGroupBox('Tensorflow')
        layout = QHBoxLayout()

        tensorflowInfo = QInfoWidget()
        layout.addWidget(tensorflowInfo)
        tensorflowInfo.addInfosFromSource(Package['tensorflow'])

        layout.addStretch()
        tensorflowBox.setLayout(layout)
        return tensorflowBox

    def torchInfo(self):
        torchBox = QGroupBox('Torch')
        layout = QHBoxLayout()

        torchInfo = QInfoWidget()
        layout.addWidget(torchInfo)
        torchInfo.addInfosFromSource(Package['torch'])

        layout.addStretch()
        torchBox.setLayout(layout)
        return torchBox


    def _cudaOld(self):
        boxLayout = QVBoxLayout()
        boxLayout.addWidget(QLabel("Python CUDA module (pycuda) not availabe"))

        # boxLayout.addWidget(QLabel(f"NVIDIA Kernel driver: "
        #                            f"{cuda.driver_version}"))
        # boxLayout.addWidget(QLabel(f"CUDA Toolkit version: "
        #                            f"{cuda.toolkit_version}"))

        text = QPlainTextEdit()
        fixedFont = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        text.document().setDefaultFont(fixedFont)
        text.setReadOnly(True)
        # text.setPlainText(str(cuda.nvidia_smi))
        text.appendPlainText(str())
        boxLayout.addWidget(text)

        boxLayout.addStretch()
        return boxLayout


class QInfoWidget(QWidget):
    """A QInfoBox displays a collection of information.

    Information
    -----------
    
    Each piece of information is registered with an internal name, a
    (human readable) title that describes its meaning and a value. In
    addition it may specify a unit (like seconds, bytes, etc.).

    Dynamic Information
    -------------------

    Information can be dynamic, that is they change over time (like
    the current time or runtime of some model, the amount of memory
    allocated, the current GPU utilization and temperature, etc.).
    Such dynamic information can be registered by a function that
    returns the current value.

    If an `QInfoBox` contains dynamic information, that is the method
    :py:meth:`containsDynamicInformation` returns `True`, it has to be
    updated on a regular basis. There are different ways to achieve this:
    * call :py:class:`startDynamic()` to run a QTimer that calls
      :py:class:`update()` on a regular basis.
    * use some external mechanism to regularly call :py:class:`update()`.

    Layout
    ------

    The information can be arranged in different ways, referred to as
    layout.  Currently only one layout is implemented: a two-column
    grid in which the entries are displayed in the order in which they
    have been added to the `QInfoBox`.

    Wishlist (for the future)
    -------------------------
    
    For the future it may be nice to arrange information based on
    groups, allow to make better use of the available space, etc.

    It may also be useful to highlight certain
    important/critical/unusual values.

    """

    def __init__(self, dynamic: Optional[float] = None,
                 source: Optional[InfoSource] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        # dict mapping attribute names to a triple (info, qtitle, qvalue),
        # Dict[str, Tuple[Info, QLabel, QLabel]]
        self._infos = {}

        # a list of Info objects that should be added/updated 
        self._newInfos = []

        # a list of names of Info objects that should be deleted
        self._deleteInfoNames = []

        # a set of names of Info objects that have a dynamic value
        self._dynamic = set()
        
        self._initDynamic(dynamic)
        self._initUI()

        if source is not None:
            self.addInfosFromSource(source)

    def _initUI(self) -> None:
        """Initialize the user interface.
        """
        layout = QVBoxLayout()
        self._grid = QGridLayout()
        layout.addLayout(self._grid)
        layout.addStretch()

        # -- begin: debug
        row = QVBoxLayout()
        if self._dynamicTimer is not None:
            self._dynamicButton = QPushButton("Dynamic")
            self._dynamicButton.setCheckable(True) 
            self._dynamicButton.clicked.connect(self.toggleDynamic)
            row.addWidget(self._dynamicButton)

        self._repaintCounter = QLabel('0')
        row.addWidget(self._repaintCounter)
        row.addStretch()

        layout.addLayout(row)
        # -- end: debug

        self.setLayout(layout)
        self.setMinimumWidth(400)

    def addInfo(self, info: Info) -> None:
        self._newInfos.append(info)
        self.update()

    def addInfosFromSource(self, source: Info, clear: bool = False) -> None:
        if clear:
            self.clear_infos()
        for name in source.info_names():
            self._newInfos.append(source.get_info(name))
        self.update()

    def clearInfos(self) -> None:
        self._newInfos = []
        self._deleteInfoNames = list(self._widgets.keys)
        self.update()

    def _updateUI(self) -> None:
        """Update the user interface to match the information
        mentioned in `_attributes`.
        """
        oldInfos = self._infos      
        oldDynamic = bool(self._dynamic)
        self._widgets = {}

        for info in self._newInfos:
            oldEntry = oldInfos.pop(info.name, None)
            if oldEntry is None:  # new entry -> create widgets
                qtitle = QLabel(info.title)
                qvalue = QLabel(info.format_value())
                row = self._grid.rowCount()
                self._grid.addWidget(qtitle, row, 0)
                self._grid.addWidget(qvalue, row, 1)
            else:  # old entry -> update content
                _oldInfo, qtitle, qvalue = oldEntry
                qtitle.setText(info.title)
                qtitle.setText(info.format_value())

            self._infos[info.name] = (info, qtitle, qvalue)
            if info.is_dynamic:
                self._dynamic.add(info.name)
            else:
                self._dynamic.discard(info.name)
        self._newInfos = []

        # delete the widgets mentioned in _deleteInfoNames 
        for name in self._deleteInfoNames:
            oldEntry = oldInfos.pop(info.name, None)
            if oldEntry is None:  # bad name -> nothing delete
                continue
            _oldInfo, qtitle, qvalue = oldEntry
            self._grid.removeWidget(qtitle)
            self._grid.removeWidget(qvalue)
            qtitle.deleteLater()
            qvalue.deleteLater()
            self._dynamic.discard(info.name)
        self._deleteInfoNames = []

        # keep the remaining infos
        self._widgets.update(oldInfos)

        # start/stop the dynamic timer if the widget has become
        # dynamic/non-dynamic
        if self._dynamicTimer is not None:  # this widget provides a timer 
            dynamic = bool(self._dynamic)
            if not oldDynamic != dynamic:
                self.toggleDynamic(dynamic)

    @protect
    def paintEvent(self, event: QPaintEvent) -> None:
        self._repaintCounter.setText(str(int(self._repaintCounter.text())+1))
        if self._newInfos or self._deleteInfoNames:
            self._updateUI()
        if self.containsDynamicInformation():
            self._updateDynamicInformation()
        super().paintEvent(event)

    #
    # dynamic information
    #

    def _initDynamic(self, dynamic: Optional[float]) -> None:
        self._dynamicUdpateInterval = dynamic

        if dynamic is not None and bool(dynamic):
            self._dynamicTimer = QTimer()
            self._dynamicTimer.setInterval(int(dynamic*1000))
            self._dynamicTimer.timeout.connect(self.update)
        else:
            self._dynamicTimer = None
        print(self, self._dynamicTimer, dynamic)
        if self._dynamicTimer is not None:
            print(self._dynamicTimer.isActive())

    def containsDynamicInformation(self) -> bool:
        """Checks if this QInfoBox contains dynamic information.
        """
        return bool(self._dynamic)

    def dynamicUpdateInterval(self) -> float:
        """The desired update interval for dynamic information in seconds.
        This is a float value, allowing for intervals shorter than a
        second, e.g., `0.5` is 500ms.
        """
        return self._dynamicUdpateInterval

    def isDynamic(self) -> bool:
        """Checks if this `QInfoBox` is running a timer to update dynamic
        information.  This will only be the case if this `QInfoBox`
        actually contains dynamic information, otherwise it is False,
        even if this `QInfoBox` was marked as dynamic.
        """
        return self._dynamicTimer is not None and self._dynamicTimer.isActive()

    def _updateDynamicInformation(self) -> None:
        for name in self._dynamic:
            info, _qtext, qvalue = self._widgets[name]
            qvalue.setText(info.format_value())

    @protect
    def toggleDynamic(self, checked: bool) -> None:
        print("toggle dynamic:", checked, self._dynamicTimer)
        if checked == self.isDynamic():
            return  # nothing to do
        if checked:
            self._dynamicTimer.start()
        else:
            self._dynamicTimer.stop()

class QPackageInfo(QGroupBox, QObserver, qobservables={
        Package: Package.Change('state_changed')}):
    """A Widget providing information about a package.

    The Widget observes the :py:class:`Package` associated
    with the package it provides information about.  If the
    state of this Package changes (e.g. if the package is imported),
    the information will be updated.
    """
    _package: Package = None

    def __init__(self, package: Package = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._initUI()
        self.setPackage(package)
        add_postimport_hook(ImportInterceptor.THIRDPARTY_MODULES,
                            self.moduleImported)

    def __del__(self):
        self.setPackage(None)
        super().__del__()

    def _initUI(self):
        """Create a :py:class:`QGroupBox` showing package information.
        """
        layout = QVBoxLayout()

        buttonLayout = QHBoxLayout()
        self._importButton = QPrepareButton("Load")
        # Unprepare (=unloading module) is not implemented
        self._importButton.setUnpreparable(False)
        buttonLayout.addWidget(self._importButton)

        self._installButton = QPushButton("Install")
        self._installButton.clicked.connect(self._onInstallButtonClicked)
        buttonLayout.addWidget(self._installButton)
        self._busyWidget = QBusyWidget()
        buttonLayout.addWidget(self._busyWidget)
        buttonLayout.addStretch()

        self._nameLabel = QLabel()
        self._versionLabel = QLabel()
        self._libraryLabel = QLabel()
        self._descriptionLabel = QLabel()
        layout.addWidget(self._nameLabel)
        layout.addLayout(buttonLayout)
        layout.addWidget(self._versionLabel)
        layout.addWidget(self._libraryLabel)
        layout.addWidget(self._descriptionLabel)
        layout.addStretch()
        self.setLayout(layout)

    def setPackage(self, package: Package) -> None:
        self._updateUI(package)

    def _updateUI(self, package: Package) -> None:
        havePackage = package is not None and bool(package)

        self.setTitle(package.label if havePackage else "None")
        self._installButton.setVisible(havePackage and
                                       not package.available)
        #self._importButton.setVisible(havePackage and
        #                              package.available and
        #                              not package.prepared)
        self._importButton.setPreparable(package)

        if havePackage:
            self._nameLabel.setText(package.module_name)
            self._descriptionLabel.setText(package.description)
            if package.prepared:
                module = sys.modules[package.module_name]
                version = package.version or '?'
                directory = package.directory or '?'
                self._versionLabel.setText(f"Version: {version}")
                self._libraryLabel.setText(f"Library: {directory}")
            else:
                self._versionLabel.setText("")
                self._libraryLabel.setText("")
        else:
            self._nameLabel.setText("No module")
            self._versionLabel.setText("")
            self._libraryLabel.setText("")
            self._descriptionLabel.setText("")

        # if package is not self._package:
        #     interests = Package.Change('state_changed')
        #     self._exchangeView('_package', package, interests=interests)

    def preparable_changed(self, package: Package,
                           change: Package.Change) -> None:
        """Notification that a :py:class:`Package` has changed.
        """
        if change.state_changed:
            self._updateUI(package)

    @QtCore.pyqtSlot(bool)
    @protect
    def _onInstallButtonClicked(self, checked: bool = False) -> None:
        self._package.install()

    def moduleImported(self, module: ModuleType) -> None:
        """Called when a new module was imported.
        """
        name = module.__name__
        package = Package[name] if name in Package else None
        self.setPackage(package)


class QPackageList(QGroupBox):
    """A `QWidget` to display the :py:class:`Package` known to the
    Toolbox.  It will display the package name as well es the
    installation status of these packages including the version if
    the package is already loaded.

    The `QPackageList` emits the `packageSelected` signal with the
    :py:prop:`Package.key` of the package in case a package was
    selected.

    FIXME[todo]: The `QPackageList` should observe the `Package`
    regsiter to get informed if the state of a package changed.

    """
    packageSelected: pyqtSignal = pyqtSignal(str)

    def __init__(self, **kwargs) -> None:
        super().__init__("Packages", **kwargs)
        self._packages = {}
        self._packageName = None

        self._updateFlag = False
        self._initUI()

    def _initUI(self) -> None:
        """Create a QGridLayout with two columns displaying package
        information. The first column contains the package name, the
        second column version (if loaded) or availability.
        Packages are listed in the order given by :py:meth:packages.
        """
        self.setMinimumWidth(300)

        self._packageGrid = QGridLayout()
        self._packageGrid.addWidget(QLabel("<b>Package</b>", self), 0, 0)
        self._packageGrid.addWidget(QLabel("<b>Version</b>", self), 0, 1)
        self._updateFlag = True

        boxLayout = QVBoxLayout()
        boxLayout.addLayout(self._packageGrid)

        updateButton = QPushButton("Update")
        updateButton.clicked.connect(self._onUpdatePackages)
        boxLayout.addWidget(updateButton)
        boxLayout.addStretch()
        self.setLayout(boxLayout)

    def _updatePackageList(self) -> None:
        """To be called in the main task.
        """
        for idx in range(self._packageGrid.rowCount(), len(Package) + 1):
            button = QPushButton('')
            button.setFlat(True)
            button.clicked.connect(self._onPackageClicked)
            self._packageGrid.addWidget(button, idx, 0)
            self._packageGrid.addWidget(QLabel(''), idx, 1)

        for idx, package in enumerate(Package._register, start=1):
            button = self._packageGrid.itemAtPosition(idx, 0).widget()
            label = self._packageGrid.itemAtPosition(idx, 1).widget()
            if package.prepared:
                info = package.version
            elif package.available:
                info = "not loaded"
            else:
                info = "not found"
            button.ID = package.key
            button.setText(package.label)
            label.setText(info)

    @protect
    def paintEvent(self, event: QPaintEvent) -> None:
        if self._updateFlag:
            self._updatePackageList()
            self._updateFlage = False
        super().paintEvent(event)

    @protect
    @pyqtSlot(bool)
    def _onPackageClicked(self, checked: bool = False) -> None:
        sender = self.sender()
        print(sender, type(sender), sender.text(), sender.ID)
        self.packageSelected.emit(sender.ID)
        #resource = Resource._register[sender.ID]
        # resource = Package._register[sender.ID]
        # self.showInfo(QPackageInfo(resource=resource))

    @protect
    @pyqtSlot(bool)
    def _onUpdatePackages(self, checked: bool = False) -> None:
        self._updateFlage = True
        self.update()


class QProcessInfo(QWidget, QObserver, qobservables={
        Toolbox: {'processes_changed'}}):
    """A ``QWidget`` for displaying process information.  Process
    information is obtained from the Toolbox.

    The `QProcessInfo` observes the py:class:`Toolbox` to get informed
    when process information changes.

    Work in progress: this class is not finished yet 
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._updateFlag = False
        self._initUI()

    def _initUI(self) -> None:
        self._button = QPushButton("Test")
        self._button.clicked.connect(self._onButtonClicked)

        self._grid = QGridLayout()
        self._grid.addWidget(QLabel("<b>Process</b>", self), 0, 0)
        self._grid.addWidget(QLabel("<b>Name</b>", self), 0, 1)
        self._grid.addWidget(QLabel("<b>Dämon</b>", self), 0, 2)
        self._grid.addWidget(QLabel("<b>Alive</b>", self), 0, 3)
        self._updateFlag = True

        layout = QVBoxLayout()
        layout.addLayout(self._grid)
        layout.addWidget(self._button)
        self.setLayout(layout)

    def _updateProcessList(self) -> None:
        """To be called in the main task.
        """
        processes = [multiprocessing.current_process()] \
            + multiprocessing.active_children()
        for idx in range(max(self._grid.rowCount()-1, len(processes))):
            if idx >= len(processes):
                # remve a row of labels
                last_row = self._grid.rowCount() - 1
                for column in range(self._grid.columnCount()):
                    widget = \
                        self._grid.itemAtPosition(last_row, column).widget()
                    self._grid.removeWidget(widget)
                    widget.deleteLater()
                continue
            if idx+1 >= self._grid.rowCount():
                # add a row of labels
                for column in range(self._grid.columnCount()):
                    self._grid.addWidget(QLabel(''), idx+1, column)
            process = processes[idx]
            self._grid.itemAtPosition(idx+1, 0).widget().\
                         setText(f"{process.pid}")
            self._grid.itemAtPosition(idx+1, 1).widget().\
                         setText(f"{process.name}")
            self._grid.itemAtPosition(idx+1, 2).widget().\
                         setText(f"{process.daemon}")
            self._grid.itemAtPosition(idx+1, 3).widget().\
                         setText(f"{process.is_alive()}")

        # thread = QThread.currentThread()
        # pool = QThreadPool.globalInstance()

    @protect
    def paintEvent(self, event: QPaintEvent) -> None:
        if self._updateFlag:
            self._updateProcessList()
            self._updateFlage = False
        super().paintEvent(event)

    @QtCore.pyqtSlot()
    @protect
    def _onButtonClicked(self):
        self._toolbox.notify_process("Test")
        self.update()

    def toolbox_changed(self, toolbox: Toolbox, info: Toolbox.Change) -> None:
        self.update()

    def update(self) -> None:
        self._updateFlag = True
        super().update()


class QNvmlInfo(QWidget):
    """A QWidget for displaying information obtained from the
    NVIDIA Management Library.

    Graphical elements
    ------------------
    _devices: QComboBox
        A combo box to select the current device.
    _name: QLabel
    _driver_version: QLabel
    _temperature: QLabel
    _temperature_slowdown: QLabel
    _temperature_shutdown: QLabel
    """

    def __init__(self, parent=None):
        """Initialize this :py:class:`QNvmlInfo`. This includes importing the
        NVIDIA Management Library Python bindings (py3nvml) and
        initializing that module. If any of these operations fails, a
        dummy content will be created displaying a corresponding
        message.
        """
        super().__init__(parent)
        try:
            self.nvml = importlib.import_module('py3nvml.py3nvml')
            self.nvml.nvmlInit()
            self._deviceCount = self.nvml.nvmlDeviceGetCount()
            self._handle = None
            self._initUI()
        except ModuleNotFoundError:
            self.nvml = None
            layout = QVBoxLayout()
            layout.addWidget(QLabel("NVIDIA Management Library (py3nvml) "
                                    "not available."))
            self.setLayout(layout)

    def __del__(self):
        """Freeing resources. This includes shutting down the NVML module.
        """
        if self.nvml is not None:
            self.nvml.nvmlShutdown()
            self.nvml = None

    def _initUI(self):
        """Create an interface containing a QComboBox to select the current
        device and several QLabels to display (device) information,
        including driver version, as well as slowdown, shutdown, and
        current device temperature.
        """
        layout = QVBoxLayout()

        grid = QGridLayout()

        self._driver_version = QLabel(self.nvml.nvmlSystemGetDriverVersion())
        grid.addWidget(QLabel("Driver Version"), 0, 0)
        grid.addWidget(self._driver_version, 0, 1)

        layout.addLayout(grid)

        self._devices = QComboBox()
        for i in range(self._deviceCount):
            handle = self.nvml.nvmlDeviceGetHandleByIndex(i)
            self._devices.addItem(self.nvml.nvmlDeviceGetName(handle))
        @protect
        def slot(device: int) -> None:
            self.update()
        self._devices.activated.connect(slot)
        layout.addWidget(self._devices)

        grid = QGridLayout()

        self._name = QLabel()
        grid.addWidget(QLabel("Name"), 0, 0)
        grid.addWidget(self._name, 0, 1)

        self._temperature = QLabel()
        grid.addWidget(QLabel("Temperatur"), 1, 0)
        box = QHBoxLayout()
        box.addWidget(self._temperature)
        box.addWidget(QLabel(u'\N{DEGREE SIGN}C'))
        box.addStretch()
        grid.addLayout(box, 1, 1)

        self._temperature_slowdown = QLabel()
        grid.addWidget(QLabel("Slowdown Temperatur"), 2, 0)
        box = QHBoxLayout()
        box.addWidget(self._temperature_slowdown)
        box.addWidget(QLabel(u'\N{DEGREE SIGN}C'))
        box.addStretch()
        grid.addLayout(box, 2, 1)

        self._temperature_shutdown = QLabel()
        grid.addWidget(QLabel("Shutdown Temperatur"), 3, 0)
        box = QHBoxLayout()
        box.addWidget(self._temperature_shutdown)
        box.addWidget(QLabel(u'\N{DEGREE SIGN}C'))
        box.addStretch()
        grid.addLayout(box, 3, 1)

        layout.addLayout(grid)
        layout.addStretch()
        self.setLayout(layout)

    def update(self):
        """Update the widgets indicating slowdown, shutdown, and current
        temperature.
        """
        currentIndex = self._devices.currentIndex()
        if self._handle is None and currentIndex is not None:
            self._handle = self.nvml.nvmlDeviceGetHandleByIndex(currentIndex)
            self._name.setText(self.nvml.nvmlDeviceGetName(self._handle))

            slowdown = self.nvml.nvmlDeviceGetTemperatureThreshold(
                self._handle, self.nvml.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN)
            shutdown = self.nvml.nvmlDeviceGetTemperatureThreshold(
                self._handle, self.nvml.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN)
            self._temperature_slowdown.setText(str(slowdown))
            self._temperature_shutdown.setText(str(shutdown))

        if self._handle is not None:
            temperature = self.nvml.nvmlDeviceGetTemperature(
                self._handle, self.nvml.NVML_TEMPERATURE_GPU)
            self._temperature.setText(str(temperature))
