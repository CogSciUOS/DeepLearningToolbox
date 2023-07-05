"""Utilization of the GStreamer service [1]. This service allows to
stream sounds.


[1] http://www.portaudio.com/
"""


# standard imports
from typing import Union
import logging
import threading

# third party imports
import numpy as np
import gi
# gi.require_version('Gst', '1.0')
from gi.repository import Gst


# toolbox imports
from ..base.sound import SoundPlayer as SoundPlayerBase

# logging
LOG = logging.getLogger(__name__)


gi.require_version('Gst', '1.0')
Gst.init(None)

class SoundPlayer(SoundPlayerBase):
    """An implementation of a :py:class:`SoundPlayerBase` based on
    GStreamer.
    """
