"""SoundPanel (experimental!)

Author: Ulf Krumnack
"""

# Generic imports
import logging
import os

# third party imports
import numpy as np

# Qt imports
from PyQt5.QtWidgets import QVBoxLayout

# toolbox imports
from dltb.base.sound import Sound, SoundDisplay
from dltb.base.sound import SoundReader, SoundPlayer, SoundRecorder
from dltb.util.error import print_exception
#from dltb.thirdparty.soundfile import SoundReader as SoundfileReader
#from dltb.thirdparty.sounddevice import (SoundPlayer as SoundDevicePlayer,
#                                         SoundRecorder as SoundDeviceRecorder)
from toolbox import Toolbox

# GUI imports
from .panel import Panel
from ..utils import QObserver
from ..widgets.sound import QSoundControl

# logging
LOG = logging.getLogger(__name__)


class SoundPanel(Panel, QObserver, qobservables={
        Toolbox: {'datasource_changed', 'input_changed'}}):

    def __init__(self, toolbox: Toolbox = None,
                 **kwargs) -> None:
        """Initialization of the ActivationsPael.

        Parameters
        ----------
        toolbox: Toolbox
        """
        super().__init__(**kwargs)

        frequency = 440
        duration = 2

        print("Creating sound")
        sound = Sound()
        print(str(sound))

        print("Appending sound")
        sound += np.sin(np.arange(0, sound.samplerate * duration) *
                        2 * np.pi * frequency / sound.samplerate)
        print(str(sound))

        print("Reading sound2")
        reader = SoundReader()

        soundfile = None
        for directory in (os.environ.get('HOME', False),
                          os.environ.get('NET', False)):
            if not directory:
                continue
            soundfile = os.path.join(directory, 'projects', 'examples',
                                     'mime', 'audio', 'wav', 'le_tigre.wav')
            if os.path.isfile(soundfile):
                break
            soundfile = None
        if soundfile is None:
            print("error: no soundfile provided")
            sys.exit(1)
        self._sound = reader.read(soundfile)
            
        print("Creating player")
        self._player = SoundPlayer()

        print("Creating recorder")
        self._recorder = SoundRecorder()

        self._initUI()
        self._layoutUI()

    def _initUI(self) -> None:
        print("Creating QSoundControl")
        self._soundControl = QSoundControl(self._sound,
                                           player=self._player,
                                           recorder=self._recorder)

    def _layoutUI(self) -> None:
        layout = QVBoxLayout()
        layout.addWidget(self._soundControl)
        self.setLayout(layout)
