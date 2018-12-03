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

from .panel import Panel

import numpy as np
import importlib
#import tensorflow as tf
#import cv2
#import keras
import sys
from types import ModuleType

class InternalsPanel(Panel):
    '''A Panel displaying system internals.
    May be of interest during development.
    '''

    def __init__(self, parent=None):
        super().__init__(parent)
        self._modules = {}
        
        self.initUI()
    
    def initUI(self):
        self._grid = QGridLayout()
        self._grid.addWidget(self.modulesInfo(), 0,0)
        self._grid.addWidget(self.cudaInfo(), 0,1)
        self._info = self._moduleInfo('cv2')
        self._grid.addWidget(self._info, 1,0,1,2)
        # layout.addWidget(self.tensorflowInfo(), 1,1)
        self.setLayout(self._grid)

    @QtCore.pyqtSlot()
    def _onInfo(self):
        print(self.sender().text())
        print(type(self))
        name = self.sender().text()
        info = self._info = self._moduleInfo(name)
        print(f"_info={self._info}")
        old_info = self._grid.replaceWidget(self._info, info)
        #old_info = self.getLayout().replaceWidget(self._info, info)
        print(f"old_info={old_info}")
        if old_info is not None:
            self._info = info
        self.update()

    def modulesInfo(self):
        box = QGroupBox('Modules')
        
        grid = QGridLayout()
        grid.addWidget(QLabel("<b>Package</b>", self), 0,0)
        grid.addWidget(QLabel("<b>Version</b>", self), 0,1)

        modules = ["numpy", "tensorflow", "keras", "appsdir", "matplotlib", "keras", "cv2", "caffe"]
        for i,m in enumerate(modules):
            button = QPushButton(m, self)
            button.clicked.connect(self._onInfo)
            grid.addWidget(button, 1+i, 0)
            if m in sys.modules:
                module = sys.modules[m]
                if hasattr(module, '__version__'):
                    info = str(module.__version__)
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

    def cudaInfo(self):
        box = QGroupBox('CUDA')

        layout = QVBoxLayout()
        #
        # Memory usage
        #
        
        # a useful solution that works for various operating systems,
        # including Linux, Windows 7, etc.:
        import os
        import psutil
        process = psutil.Process(os.getpid())
        layout.addWidget(QLabel("Memory usage: {process.memory_info().rss}"))

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
        layout.addWidget(QLabel("Memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}"))

        grid = QGridLayout()

        try:
            import pycuda.autoinit
            # ImportError: libcurand.so.8.0: cannot open shared object file: No such file or directory
            import pycuda.driver as cuda

            (free,total)=cuda.mem_get_info()
            grid.addWidget(QLabel("<b>Global Memory</b>", self), 0,0)
            grid.addWidget(QLabel(f"{total}", self), 0,1)
            grid.addWidget(QLabel(f"{free}", self), 0,2)

            print("Global memory occupancy:%f%% free"%(free*100/total))

            for devicenum in range(cuda.Device.count()):
                device=cuda.Device(devicenum)
                attrs=device.get_attributes()
            
                # Beyond this point is just pretty printing
                print("\n===Attributes for device %d"%devicenum)
                for (key,value) in attrs.iteritems():
                    print("%s:%s"%(str(key),str(value)))
        except ImportError as e:
            print(e, file=sys.stderr)
            grid.addWidget(QLabel("<b>CUDA module not installed</b>", self),
                           0,0)
        layout.addLayout(grid)

        import subprocess
        #nvidia_smi = str(subprocess.check_output(["nvidia-smi", "-L"]))
        #n = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
        nvidia_smi = subprocess.check_output(["nvidia-smi"])

        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(nvidia_smi)
        layout.addWidget(text)

        layout.addStretch()
        box.setLayout(layout)
        return box

    def _moduleInfo(self, name):
        print(f"HELLO module {name}")

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
                boxLayout.addWidget(QLabel(f"Version = {module.__version__}"))
                boxLayout.addStretch()
                
        else:
            boxLayout = QVBoxLayout()
            boxLayout.addWidget(QLabel(f"'{name}' is of type "
                                       f"{type(self._modules[name])}"))
            boxLayout.addStretch()

        box.setLayout(boxLayout)
        return box

    def cv2Info(self, cv2):
        print("HELLO CV2")
        layout = QVBoxLayout()
        
        info = cv2.getBuildInformation()
        
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(info)

        layout.addWidget(QLabel(f"Version: {cv2.__version__}"))
        layout.addWidget(QLabel(f"Version: {cv2.__file__}"))
        layout.addWidget(text)
        layout.addStretch()
        return layout

    def tensorflowInfo(self, tf):
        # from tensorflow.python.client import device_lib

        #  There is an undocumented method called
        #  device_lib.list_local_devices() that enables you to list
        #  the devices available in the local process (As an
        #  undocumented method, this is subject to backwards
        #  incompatible changes.)

        # def get_available_gpus():
        #     local_device_protos = device_lib.list_local_devices()
        #     return [x.name for x in local_device_protos if x.device_type == 'GPU']

        # Note that (at least up to TensorFlow 1.4), calling
        # device_lib.list_local_devices() will run some initialization
        # code that, by default, will allocate all of the GPU memory
        # on all of the devices (GitHub issue). To avoid this, first
        # create a session with an explicitly small
        # per_process_gpu_fraction, or allow_growth=True, to prevent
        # all of the memory being allocated. See this question for
        # more details

        # https://stackoverflow.com/questions/38009682/how-to-tell-if-tensorflow-is-using-gpu-acceleration-from-inside-python-shell
        label = QLabel("<b>Tensorflow<b>\n"
                       f"Version = {tf.__version__}")
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addStretch()
        return layout
