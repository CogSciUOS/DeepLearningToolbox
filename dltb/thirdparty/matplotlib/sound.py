"""Matplotlib implementation of sound related functionality.


Examples:

>>> from dltb.thirdparty.matplotlib.sound import MplSoundDisplay
>>> display = MplSoundDisplay(sound="examples/win_xp_shutdown.wav", player=True)
>>> display.show()

"""
# FIXME[todo]:
# - adaptive display (like in librosa, use subsampling)
# - create a Librosa Soundplotter
# - spectrogram display
# - freeze display during playback and only update position marker
#    https://matplotlib.org/stable/tutorials/advanced/blitting.html

# standard imports
from typing import Optional

# thirdparty imports
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.backend_bases import KeyEvent, MouseEvent

# toolbox imports
from dltb.base.sound import Sound, SoundView, SoundDisplay, SoundPlayer
from . import MplPlotter, MplSimpleDisplay, LOG


class SoundPlotter(SoundView, MplPlotter, SoundPlayer.Observer):
    """A matplotlib implementation of a :py:class:`SoundView`.
    """
    _mpl_position: Line2D = None

    _mpl_key_press_id = None
    _mpl_button_press_id = None
    _mpl_timer = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _init_axes(self) -> None:
        super()._init_axes()
        axes = self.axes
        # figure.set_figwidth(16)
        # figure.set_figheight(8)
        #plt.style.use(['dark_background', 'bmh'])
        #plt.rc('axes', facecolor='k')
        #plt.rc('figure', facecolor='k')
        #plt.rc('figure', figsize=figsize, dpi=dpi)
        # cmap = 'gray' if canvas.ndim == 2 else None
        #self._mpl_wave, = axes.plot(time_array, wave_array)
        # with plt.style.context('fivethirtyeight'):
        self._update_sound()
        self._mpl_position = axes.axvline(0, color='red')

        canvas = self.axes.figure.canvas
        self._mpl_key_press_id = \
            canvas.mpl_connect('key_press_event', self.on_key_pressed)
        self._mpl_button_press_id = \
            canvas.mpl_connect('button_press_event', self.on_button_pressed)

        # Create timer with 50ms (default=1000ms) - do not start it yet ...
        self._mpl_timer = canvas.new_timer(interval=50)
        self._mpl_timer.add_callback(self.update_position)

    def _release_axes(self) -> None:
        canvas = self.axes.figure.canvas
        canvas.mpl_disconnect(self._key_press_id)
        canvas.mpl_disconnect(self._button_press_id)
        self._mpl_timer.stop()
        super()._release_axes()

    def _set_sound(self, sound: Optional[Sound]) -> None:
        """Set a new :py:class:`Sound` for this `MplSoundPlotter`.
        The new sound is guaranteed to be different from the current
        sound.
        """
        super()._set_sound(sound)
        self._update_sound()

    def _update_sound(self) -> None:
        axes = self.axes
        if axes is not None:
            canvas = axes.figure.canvas 
            canvas.draw()
            canvas.flush_events()

    def on_key_pressed(self, event: KeyEvent) -> None:
        """Implementation of a Matplotlib key event handler.
        """
        # event properties:
        #   'canvas'
        #   'guiEvent'
        #   'inaxes'
        #   key: str
        #       'a','b', ... 'up', 'down', 'left', 'right',
        #       'delete', 'backspace', ...
        #   'lastevent'
        #   'name'
        #   'x'
        #   'xdata'
        #   'y'
        #   'ydata'
        key = event.key
        LOG.info("Detected key press event in matplotlib figure: '%s'.", key)
        if key == 'p':  # p = play
            if self.player is not None:
                if self.player.playing:
                    self.player.stop()
                else:
                    self.player.play(self.sound, blocking=False)
            else:
                LOG.warning("No player for SoundDisplay "
                            "- will not play sound.")
        elif key == 'u':  # u = update
            axes = self.axes
            if axes is not None:
                canvas = axes.figure.canvas 
                canvas.draw()
                canvas.flush_events()

    def on_button_pressed(self, event: MouseEvent) -> None:
        """Implementation of a Matplotlib mouse press event handler.
        """
        # event Properties:
        #   'button'
        #   'canvas'
        #   'dblclick'
        #   'guiEvent'
        #   'inaxes'
        #   'key'
        #   'lastevent'
        #   'name'
        #   'step'
        #   'x'
        #   'xdata'
        #   'y'
        #   'ydata'
        x, y = event.x, event.y
        xdata, ydata = event.xdata, event.ydata
        button = event.button
        LOG.info("Detected mouse press event in matplotlib figure (%d, %d), "
                 "data = (%.2f,%.2f), button=%s.", x, y, xdata, ydata, button)
        if self.player is not None:
            self.player.position = xdata
            self.update_position()

    def update_position(self) -> None:
        player = self.player
        if player is None:
            return  # No current position
        position = player.position
        if position is not None:
            self._mpl_position.set_xdata(position)
        else:
            self._mpl_position.set_xdata(0)
        if not player.playing:
            self._mpl_timer.stop()

    def _set_player(self, player: Optional[SoundPlayer]) -> None:
        if self._player is not None:
            self.unobserve(self._player)
        super()._set_player(player)
        if self._player is not None:
            self.observe(self._player, {'state_changed'})

    def player_changed(self, player: SoundPlayer,
                       info: SoundPlayer.Change) -> None:
        if info.state_changed:
            if self._mpl_timer is not None:
                if player.playing:
                    self._mpl_timer.start()
                # else:
                #     self._mpl_timer.stop()
                # QObject::killTimer: Timers cannot be stopped from
                # another thread


class WavePlotter(SoundPlotter):
    """
    """
    _mpl_wave = None

    def _init_axes(self) -> None:
        axes = self.axes
        axes.set_xlabel('time [s]')
        axes.set_ylabel('amplitude [/]')
        axes.set_title(r'$x(t)$', size=20)
        self._mpl_wave, = axes.plot([], [])
        super()._init_axes()

    def _release_axes(self) -> None:
        self._mpl_wave = None
        super()._release_axes()

    def _update_sound(self) -> None:
        sound, axes, mpl_wave = self.sound, self.axes, self._mpl_wave
        if axes is None or mpl_wave is None:
            return  # cannot display sound without an Axes object
        if sound is not None:
            wave_array = sound.data[:, 0]
            time_step = 1 / sound.samplerate
            duration = len(wave_array) * time_step
            time_array = np.r_[0:duration:time_step]
            axes.set_xlim(0, duration)
            axes.set_ylim(-1, 1)
            self._mpl_wave.set_data(time_array, wave_array)
        else:
            mpl_wave.set_data([], [])

        super()._update_sound()


class MplSoundDisplay(SoundDisplay, MplSimpleDisplay):
    """The `SoundDisplay` uses a :py:class:`SoundPlotter` to plot a
    :py:class:`Sound` object.  It may also incorporate controls
    for playback and recording of sound.

    Playback and Recording
    ----------------------
    """

    def __init__(self, plotter: Optional[SoundPlotter] = None,
                 **kwargs) -> None:
        if plotter is None:
            plotter = WavePlotter()
        super().__init__(plotter=plotter, **kwargs)
