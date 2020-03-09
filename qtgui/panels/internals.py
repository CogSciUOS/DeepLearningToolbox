'''
File: internals.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
'''

import sys
from util.resources import (Resource, ModuleResource,
                            View as ResourceView,
                            Controller as ResourceController)
from ..utils import QObserver

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QWidget, QGroupBox, QLabel, QPushButton,
                             QHBoxLayout, QVBoxLayout)


class ModuleInfo(QGroupBox, QObserver, Resource.Observer):
    """A Widget providing information about a module.

    The Widget observes the :py:class:`ModuleResource` associated
    with the module it provides information about.  If the
    state of this resource changes (e.g. if the module is imported),
    the information will be updated.
    """
    _resource: ResourceController = None
    
    def __init__(self, resource: ResourceController=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._initGui()
        self.setResource(resource)

    def __del__(self):
        self.setResource(None)
        super().__del__()
        
    def setResource(self, resource: ResourceController) -> None:
        interests = Resource.Change('status_changed')
        self._exchangeView('_resource', resource, interests=interests)

    def resource_changed(self, resource: Resource,
                         change: Resource.Change) -> None:
        if change.status_changed:
            self.update()

    def _initGui(self):
        """Create a :py:class:`QGroupBox` showing module information.
        """
        layout = QVBoxLayout()

        buttonLayout = QHBoxLayout()
        self._importButton = QPushButton("Load")
        self._importButton.clicked.connect(self._onImportButtonClicked)
        buttonLayout.addWidget(self._importButton)

        self._installButton = QPushButton("Install")
        self._installButton.clicked.connect(self._onInstallButtonClicked)
        buttonLayout.addWidget(self._installButton)
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

    @QtCore.pyqtSlot()
    def _onImportButtonClicked(self):
        self._resource.prepare()

    @QtCore.pyqtSlot()
    def _onInstallButtonClicked(self):
        self._resource.install()
        
    def update(self):
        haveResource = self._resource is not None and bool(self._resource)
        self.setTitle(self._resource.label if haveResource else "None")
        self._installButton.setVisible(haveResource and
                                       not self._resource.available)
        self._importButton.setVisible(haveResource and
                                      self._resource.available and
                                      not self._resource.prepared)
            
        if not haveResource:
            self._nameLabel.setText("No module")
            self._versionLabel.setText("")
            self._libraryLabel.setText("")
            self._descriptionLabel.setText("")
        else:
            self._nameLabel.setText(self._resource.module)
            self._descriptionLabel.setText(self._resource.description)
            if self._resource.prepared:
                module = sys.modules[self._resource.module]
                self._versionLabel.setText("Version: " + self._resource.version)
                self._libraryLabel.setText("Library: " + module.__file__)
            else:
                self._versionLabel.setText("")
                self._libraryLabel.setText("")


from PyQt5 import QtCore
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QGroupBox,
                             QGridLayout, QHBoxLayout, QVBoxLayout,
                             QPlainTextEdit, QComboBox)
from PyQt5.QtGui import QFontDatabase

from .panel import Panel

import numpy as np
import importlib

import sys
import os
import re
from types import ModuleType
import util.resources
import util

from util.error import protect



class InternalsPanel(Panel):
    '''A Panel displaying system internals.
    May be of interest during development.

    Attributes
    ----------
    _modules: dict
        A mapping from module names to module information.
        This can be the acutal module (if already loaded),
        or a string describing the state of the module
        ("not loaddd" or "not found"). This information
        is initialized and updated by the method
        :py:meth:_updateModules.

    Graphical elements
    ------------------
    _grid: QGridLayout
    '''


    _grid: QGridLayout = None
    _moduleGrid: QGridLayout = None

    _moduleName: str = None

    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._modules = {}
        
        self.initUI()
    
    def initUI(self):
        self._layout = QVBoxLayout()
        self._grid = QGridLayout()
        self._grid.addWidget(self.modulesInfo(), 0,0)
        self._grid.addLayout(self.systemInfo(), 0,1)

        self._layout.addLayout(self._grid)
        self._info = QLabel("Info")
        self._layout.addWidget(self._info)
        self.setLayout(self._layout)

    @QtCore.pyqtSlot()
    @protect
    def _onInfo(self):
        sender = self.sender()
        resource = ResourceController(Resource[sender.ID])
        self.showInfo(ModuleInfo(resource=resource))

    @QtCore.pyqtSlot()
    @protect
    def _onUpdateModules(self):
        self._updateModules()

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

    def modulesInfo(self) -> QGroupBox:
        """Create a QGridLayout with two columns displaying module
        information. The first column contains the module name, the
        second column version (if loaded) or availability.
        Modules are listed in the order given by :py:meth:modules.

        Result
        ------
        box: QGroupBox
            A QWidget displaying the module information.
        """
        box = QGroupBox('Modules')
        box.setMinimumWidth(300)
        
        self._moduleGrid = QGridLayout()
        self._moduleGrid.addWidget(QLabel("<b>Package</b>", self), 0,0)
        self._moduleGrid.addWidget(QLabel("<b>Version</b>", self), 0,1)

        for i,m in enumerate(ModuleResource):
            button = QPushButton(m.label, self)
            button.ID = m._id  # FIXME[hack]
            button.setFlat(True)
            button.clicked.connect(self._onInfo)
            self._moduleGrid.addWidget(button, 1+i, 0)
            self._moduleGrid.addWidget(QLabel('', self), 1+i, 1)
        self._updateModules()

        boxLayout = QVBoxLayout()
        boxLayout.addLayout(self._moduleGrid)

        updateButton = QPushButton("Update")
        updateButton.clicked.connect(self._onUpdateModules)
        boxLayout.addWidget(updateButton)
        boxLayout.addStretch()
        box.setLayout(boxLayout)

        return box

    def _updateModules(self):
        """Update the module list.
        """
        for i, m in enumerate(ModuleResource):
            if m.prepared:
                info = m.version
            elif m.available:
                info = "not loaded"
            else:
                info = "not found"
            self._moduleGrid.itemAtPosition(1+i, 1).widget().setText(info)

    def systemInfo(self):

        pythonBox = QGroupBox('Python')
        boxLayout = QVBoxLayout()
        boxLayout.addWidget(QLabel(f"Python version: {sys.version}"))
        boxLayout.addWidget(QLabel(f"Platform: {sys.platform}"))
        boxLayout.addWidget(QLabel(f"Prefix: {sys.prefix}"))
        boxLayout.addWidget(QLabel(f"Executable: {sys.executable}"))
        boxLayout.addStretch()
        pythonBox.setLayout(boxLayout)

        hardwareBox = QGroupBox('Hardware')
        boxLayout = QVBoxLayout()
        for i, cpu in enumerate(util.resources.cpus):
            prefix = f"{i+1}. " if len(util.resources.cpus) > 1 else ""
            boxLayout.addWidget(QLabel(f"{prefix}CPU: {cpu.name}"))
        for i, gpu in enumerate(util.resources.gpus):
            prefix = f"{i+1}. " if len(util.resources.gpus) > 1 else ""
            boxLayout.addWidget(QLabel(f"{prefix}GPU: {gpu.name}"))
        # Memory (1)
        import os
        mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        # SC_PAGE_SIZE is often 4096.
        # SC_PAGESIZE and SC_PAGE_SIZE are equal.
        mem = mem_bytes>>20
        boxLayout.addWidget(QLabel("Total physical memory: {:,} MiB".
                                   format(mem)))
        boxLayout.addStretch()
        hardwareBox.setLayout(boxLayout)
        
        #
        # Platform
        #
        systemBox = QGroupBox('System')
        boxLayout = QVBoxLayout()

        import platform
        boxLayout.addWidget(QLabel(f"node: {platform.node()}"))
        # boxLayout.addWidget(QLabel(f"uname: {platform.uname()}"))
        boxLayout.addWidget(QLabel(f"system: {platform.system()}"))
        boxLayout.addWidget(QLabel(f"release: {platform.release()}"))
        # boxLayout.addWidget(QLabel(f"version: {platform.version()}"))
        boxLayout.addWidget(QLabel(f"machine/processor: "
                                   f"{platform.machine()}/"
                                   f"{platform.processor()}"))
        boxLayout.addStretch()
        systemBox.setLayout(boxLayout)
        

        resourcesBox = QGroupBox('Resources')
        boxLayout = QVBoxLayout()


        # Memory (2)
        # a useful solution that works for various operating systems,
        # including Linux, Windows 7, etc.:
        try:
            import psutil
            mem = psutil.virtual_memory()
            # mem.total: total physical memory available
            boxLayout.addWidget(QLabel("Total physical memory: "
                                       f"{mem.total}"))
            process = psutil.Process(os.getpid())
            boxLayout.addWidget(QLabel("Memory usage: "
                                       f"{process.memory_info().rss}"))
        except ModuleNotFoundError:
            pass

        # For Unixes (Linux, Mac OS X, Solaris) you could also use the
        # getrusage() function from the standard library module
        # resource. The resulting object has the attribute ru_maxrss,
        # which gives peak memory usage for the calling process.
        
        # resource is a standard library module.
        import resource
        
        # The Python docs aren't clear on what the units are exactly,
        # but the Mac OS X man page for getrusage(2) describes the
        # units as bytes. The Linux man page isn't clear, but it seems
        # to be equivalent to the information from /proc/self/status,
        # which is in kilobytes.
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        boxLayout.addWidget(QLabel("Peak Memory usage: {:,} kiB".
                                   format(rusage.ru_maxrss)))


        if util.resources.cuda is not None:
            button = QPushButton("CUDA")
            #button.setFlat(True)
            @protect
            def slot(clicked: bool):
                self.showInfo(self.cudaInfo())
            button.clicked.connect(slot)
            boxLayout.addWidget(button)

        resourcesBox.setLayout(boxLayout)



        #
        # layout the boxes
        #
        layout = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(hardwareBox)
        row.addWidget(pythonBox)
        row.addWidget(systemBox)        
        layout.addLayout(row)
        layout.addWidget(resourcesBox)
        layout.addStretch()
        return layout

    def cv2Info(self, cv2):
        layout = QVBoxLayout()
        
        info = cv2.getBuildInformation()
        
        text = QPlainTextEdit()
        fixedFont = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        text.document().setDefaultFont(fixedFont)
        text.setReadOnly(True)
        text.setPlainText(info)

        layout.addWidget(QLabel(f"Version: {cv2.__version__}"))
        layout.addWidget(QLabel(f"Library: {cv2.__file__}"))
        layout.addWidget(text)
        layout.addStretch()
        return layout

    def tensorflowInfo(self, tf):
        layout = QVBoxLayout()
        label = QLabel("<b>Tensorflow<b>\n"
                       f"Version = {tf.__version__}")
        layout.addWidget(label)

        layout.addWidget(QLabel(f"Tensorflow devices:"))
        #  There is an undocumented method called
        #  device_lib.list_local_devices() that enables you to list
        #  the devices available in the local process (As an
        #  undocumented method, this is subject to backwards
        #  incompatible changes.)
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        for dev in local_device_protos:
            layout.addWidget(QLabel(f"Device: {dev.name} ({dev.device_type})"))

        # Note that (at least up to TensorFlow 1.4), calling
        # device_lib.list_local_devices() will run some initialization
        # code that, by default, will allocate all of the GPU memory
        # on all of the devices (GitHub issue). To avoid this, first
        # create a session with an explicitly small
        # per_process_gpu_fraction, or allow_growth=True, to prevent
        # all of the memory being allocated. See this question for
        # more details

        # https://stackoverflow.com/questions/38009682/how-to-tell-if-tensorflow-is-using-gpu-acceleration-from-inside-python-shell
        layout.addStretch()
        return layout

    def kerasInfo(self, keras):
        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"Backend: {keras.backend.backend()}"))
        layout.addStretch()
        return layout

    def cudaInfo(self):
        cudaBox = QGroupBox('CUDA')
        boxLayout = QVBoxLayout()

        if os.path.exists('/proc/driver/nvidia/version'):
            with open('/proc/driver/nvidia/version') as f:
                driver_info = f.read()
            match = re.search('Kernel Module +([^ ]*)', driver_info)
            if match:
                boxLayout.addWidget(QLabel(f"Kernel module: {match.group(1)}"))
            match = re.search('gcc version +([^ ]*)', driver_info)
            if match:
                boxLayout.addWidget(QLabel(f"GCC version: {match.group(1)}"))


            boxLayout.addWidget(QLabel(f"NVIDIA Kernel driver: "
                                       f"{util.resources.cuda.driver_version}"))
            boxLayout.addWidget(QLabel(f"CUDA Toolkit version: "
                                       f"{util.resources.cuda.toolkit_version}"))

            text = QPlainTextEdit()
            fixedFont = QFontDatabase.systemFont(QFontDatabase.FixedFont)
            text.document().setDefaultFont(fixedFont)
            text.setReadOnly(True)
            text.setPlainText(str(util.resources.cuda.nvidia_smi))
            text.appendPlainText(str(util.resources.cuda.nvidia_smi_l))
            boxLayout.addWidget(text)

        # Now use the python module pycuda
        try:
            import pycuda.autoinit
            import pycuda.driver as cuda

            (free,total) = cuda.mem_get_info()
            boxLayout.addWidget(QLabel("<b>Global GPU Memory</b>"))
            boxLayout.addWidget(QLabel(f"Total: {total}"))
            boxLayout.addWidget(QLabel(f"Free: {free}"))
            boxLayout.addWidget(QLabel("Global memory occupancy: "
                                       f"{free*100/total:2.4}% free"))

            for devicenum in range(cuda.Device.count()):
                device=cuda.Device(devicenum)
                attrs=device.get_attributes()
            
                # Beyond this point is just pretty printing
                print("\n===Attributes for device %d"%devicenum)
                for (key,value) in attrs.items():
                    print("%s:%s"%(str(key),str(value)))
        except ImportError as e:
            print(e, file=sys.stderr)
            # ImportError: libcurand.so.8.0
            # The problem occurs with the current anaconda version
            # (2017.1, "conda install -c lukepfister pycuda").
            # The dynamic library "_driver.cpython-36m-x86_64-linux-gnu.so"
            # is linked against "libcurand.so.8.0". However, the cudatookit
            # installed by anaconda is verion 9.0.
            boxLayout.addWidget(QLabel("Python CUDA module (pycuda) not availabe"))

        try:
            nvmlInfo = QNvmlInfo()
            boxLayout.addWidget(nvmlInfo)
            util.add_timer_callback(nvmlInfo.update)
        except ImportError as e:
            print(e, file=sys.stderr)
            boxLayout.addWidget(QLabel("Python NVML module (py3nvml) not availabe"))

        cudaBox.setLayout(boxLayout)
        return cudaBox


class QNvmlInfo(QWidget):
    """A QWidget for displaying information obtained from the
    NVIDIA Management Library (Python bindings: py3nvml)

    Attributes
    ----------

    nvml: module
        A reference to the NVIDIA Management Library.
    _deviceCount: int
        The number of NVIDA devices.
    _handle:
        An NVML handle for the current device. None if no device
        is selected.
    
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
        except ModuleNotFoundError as e:
            self.nvml = None
            layout = QVBoxLayout()
            layout.add(QLabel("NVIDIA Management Library not available"))
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
        grid.addWidget(QLabel("Driver Version"), 0,0)
        grid.addWidget(self._driver_version, 0,1)

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
        grid.addWidget(QLabel("Name"), 0,0)
        grid.addWidget(self._name, 0,1)

        self._temperature = QLabel()
        grid.addWidget(QLabel("Temperatur"), 1,0)
        box = QHBoxLayout()
        box.addWidget(self._temperature)
        box.addWidget(QLabel(u'\N{DEGREE SIGN}C'))
        box.addStretch()
        grid.addLayout(box, 1,1)

        self._temperature_slowdown = QLabel()
        grid.addWidget(QLabel("Slowdown Temperatur"), 2,0)
        box = QHBoxLayout()
        box.addWidget(self._temperature_slowdown)
        box.addWidget(QLabel(u'\N{DEGREE SIGN}C'))
        box.addStretch()
        grid.addLayout(box, 2,1)

        self._temperature_shutdown = QLabel()
        grid.addWidget(QLabel("Shutdown Temperatur"), 3,0)
        box = QHBoxLayout()
        box.addWidget(self._temperature_shutdown)
        box.addWidget(QLabel(u'\N{DEGREE SIGN}C'))
        box.addStretch()
        grid.addLayout(box, 3,1)

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

            slowdown = self.nvml.nvmlDeviceGetTemperatureThreshold(self._handle,
                self.nvml.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN)
            shutdown = self.nvml.nvmlDeviceGetTemperatureThreshold(self._handle,
                self.nvml.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN)
            self._temperature_slowdown.setText(str(slowdown))
            self._temperature_shutdown.setText(str(shutdown))

        if self._handle is not None:
            temperature = self.nvml.nvmlDeviceGetTemperature(self._handle,
                self.nvml.NVML_TEMPERATURE_GPU)
            self._temperature.setText(str(temperature))
