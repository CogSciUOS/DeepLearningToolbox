"""Access to librosa.


from dltb.thirdparty.matplotlib.sound import MplSoundDisplay
from dltb.thirdparty.librosa import SpectrogramPlotter
from dltb.base.sound import Sound

display = MplSoundDisplay(plotter=SpectrogramPlotter(), player=True)
display.show(sound='examples/win_xp_shutdown.wav')

"""
# standard imports
from typing import Optional

# third party imports
import numpy as np
import librosa
import librosa.display
from matplotlib.collections import QuadMesh

# toolbox imports
from dltb.base.sound import Sound, Soundlike
from dltb.base.sound import SoundView, SoundReader as SoundReaderBase
from .matplotlib.sound import SoundPlotter


class SoundReader(SoundReaderBase):
    """A :py:class:`SoundReader` based on `librosa`.
    Notice that librosa does not actually implement sound loading,
    but rather uses other libraries, if available, like
    `PySoundFile`, or `audioread`.
    """

    def __str__(self) -> str:
        return "Librosa based SoundReader"

    def read(self, filename: str, channels: Optional[int] = None,
             samplerate: Optional[float] = None,
             endian: Optional[str] = None) -> Sound:

        # data is in the (frames, channels) format
        array, samplerate = librosa.load(filename)
        return Sound(array=array, samplerate=samplerate)


class SpectrogramPlotter(SoundPlotter):
    _mpl_spec: QuadMesh = None
    
    def _set_sound(self, sound: Optional[Sound]) -> None:
        """Set a new :py:class:`Sound` for this `MplSoundPlotter`.
        The new sound is guaranteed to be different from the current
        sound.
        """
        super()._set_sound(sound)
        self._update_sound()

    def _update_sound(self) -> None:
        axes, sound = self.axes, self.sound
        if axes is None:
            return  # cannot display sound without an Axes object

        axes.clear()
        self._mpl_position = axes.axvline(0, color='red')

        if sound is not None:
            wave_array = sound.data[:, 0]
            time_step = 1 / sound.samplerate
            duration = len(wave_array) * time_step
            time_array = np.r_[0:duration:time_step]
            
            D = librosa.stft(wave_array)  # STFT of wave_array
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            self._mpl_spec = \
                librosa.display.specshow(S_db, axes=self.axes,
                                         x_axis='time', y_axis='log')
            # print("Spectrogam: ", axes.get_xlim(), axes.get_ylim())
            #axes.set_xlim(0, duration)
            # extent=(left, right, bottom, top)
            # self._mpl_spec.set_extent((0, duration, axes.get_ylim()[0],
            #                            axes.get_ylim()[1]))

        super()._update_sound()
