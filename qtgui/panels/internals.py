'''
File: internals.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
'''

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QGroupBox,
                             QGridLayout, QHBoxLayout, QVBoxLayout,
                             QPlainTextEdit)
from PyQt5.QtGui import QFontDatabase

from .panel import Panel

import numpy as np
import importlib
#import tensorflow as tf
#import cv2
#import keras
import sys
import os
import re
from types import ModuleType
import util.resources

class InternalsPanel(Panel):
    '''A Panel displaying system internals.
    May be of interest during development.
    '''

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
        self._info = self._moduleInfo('cv2')
        self._layout.addWidget(self._info)
        self.setLayout(self._layout)

    @QtCore.pyqtSlot()
    def _onInfo(self):
        name = self.sender().text()
        self.showInfo(self._moduleInfo(name))

    def showInfo(self, info: QWidget):
        if self._layout.replaceWidget(self._info, info) is not None:
            self._info.deleteLater()
            self._info = info

    def modulesInfo(self):
        box = QGroupBox('Modules')
        box.setMinimumWidth(300)
        
        grid = QGridLayout()
        grid.addWidget(QLabel("<b>Package</b>", self), 0,0)
        grid.addWidget(QLabel("<b>Version</b>", self), 0,1)

        modules = ["numpy", "tensorflow", "keras", "appsdir", "matplotlib", "keras", "cv2", "caffe", "PyQt5", "pycuda"]
        for i,m in enumerate(modules):
            button = QPushButton(m, self)
            button.setFlat(True)
            button.clicked.connect(self._onInfo)
            grid.addWidget(button, 1+i, 0)
            if m in sys.modules:
                module = sys.modules[m]
                if hasattr(module, '__version__'):
                    info = str(module.__version__)
                elif m == "PyQt5":
                    info = QtCore.QT_VERSION_STR
                else:
                    info = "loaded, no version"
                self._modules[m] = sys.modules[m]
            else:
                spec = importlib.util.find_spec(m)
                if spec is None:
                    info = "not found"
                else:
                    # spec.name
                    # spec.origin -> file
                    info = "not loaded"
                self._modules[m] = info

            grid.addWidget(QLabel(info, self), 1+i, 1)

        boxLayout = QVBoxLayout()
        boxLayout.addLayout(grid)
        boxLayout.addStretch()
        box.setLayout(boxLayout)
        
        return box

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
            def slot(clicked: bool): self.showInfo(self.cudaInfo())
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

    def _moduleInfo(self, name):
        labels = {
            'cv2': 'OpenCV',
            'tensorflow': 'TensorFlow',
        }
        box = QGroupBox(labels.get(name, name))

        if name not in self._modules:
            boxLayout = QVBoxLayout()
            boxLayout.addWidget(QLabel("Not availabe"))
            boxLayout.addStretch()
        elif isinstance(self._modules[name], str):
            boxLayout = QVBoxLayout()
            boxLayout.addWidget(QLabel(self._modules[name]))
            boxLayout.addStretch()
        elif isinstance(self._modules[name], ModuleType):
            #boxLayout = contentProvider(self._modules[name])
            module = self._modules[name]
            infoMethod = name + "Info"
            op = getattr(self, infoMethod, None)
            if callable(op):
                boxLayout = op(module)
            else:
                boxLayout = QVBoxLayout()
                boxLayout = QVBoxLayout()
                if hasattr(module, '__version__'):
                    version = str(module.__version__)
                elif m == "PyQt5":
                    version = QtCore.QT_VERSION_STR
                else:
                    version = "no version info"
                boxLayout.addWidget(QLabel(f"Version = {version}"))
                boxLayout.addWidget(QLabel(f"Library = {module.__file__}"))
                boxLayout.addStretch()
                
        else:
            boxLayout = QVBoxLayout()
            boxLayout.addWidget(QLabel(f"'{name}' is of type "
                                       f"{type(self._modules[name])}"))
            boxLayout.addStretch()

        box.setLayout(boxLayout)
        return box

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
            boxLayout.addWidget(QLabel("<b>Global Memory</b>"))
            boxLayout.addWidget(QLabel(f"{total}"))
            boxLayout.addWidget(QLabel(f"{free}"))
            boxLayout.addWidget(QLabel("Global memory occupancy: "
                                       f"{free*100/total}%% free"))

            for devicenum in range(cuda.Device.count()):
                device=cuda.Device(devicenum)
                attrs=device.get_attributes()
            
                # Beyond this point is just pretty printing
                print("\n===Attributes for device %d"%devicenum)
                for (key,value) in attrs.iteritems():
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

        cudaBox.setLayout(boxLayout)
        return cudaBox
