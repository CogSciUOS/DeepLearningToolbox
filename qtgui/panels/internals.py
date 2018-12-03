'''
File: internals.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
'''

from PyQt5.QtWidgets import QLabel, QGridLayout

from .panel import Panel

import numpy as np
#import tensorflow as tf
#import keras
import sys

class InternalsPanel(Panel):
    '''A Panel displaying system internals.
    May be of interest during development.
    '''

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
    
    def initUI(self):
        layout = QGridLayout()
        layout.addLayout(self.modulesInfo(), 0,0)
        layout.addLayout(self.cudaInfo(), 0,1)
        self.setLayout(layout)
        
    def modulesInfo(self):
        layout = QGridLayout()
        layout.addWidget(QLabel("<b>Package</b>", self), 0,0)
        layout.addWidget(QLabel("<b>Version</b>", self), 0,1)

        modules = ["numpy", "tensorflow", "keras", "appsdir"]
        for i,m in enumerate(modules):
            layout.addWidget(QLabel(m, self), 1+i, 0)
            if m not in sys.modules:
                info = "not loaded"
            else:
                module = sys.modules[m]
                if hasattr(module, '__version__'):
                    info = str(module.__version__)
                else:
                    info = "loaded, no version"
            layout.addWidget(QLabel(info, self), 1+i, 1)
        return layout

    def tensorFlowInfo(self):
        pass
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

    

    def cudaInfo(self):

        layout = QGridLayout()

        try:
            import pycuda.autoinit
            # ImportError: libcurand.so.8.0: cannot open shared object file: No such file or directory
            import pycuda.driver as cuda

            (free,total)=cuda.mem_get_info()
            layout.addWidget(QLabel("<b>Global Memory</b>", self), 0,0)
            layout.addWidget(QLabel(f"{total}", self), 0,1)
            layout.addWidget(QLabel(f"{free}", self), 0,2)

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
            layout.addWidget(QLabel("<b>CUDA module not installed</b>", self),
                             0,0)

        # import subprocess
        # n = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
        return layout
