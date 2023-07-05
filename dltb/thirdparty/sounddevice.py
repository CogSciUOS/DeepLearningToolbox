"""Access to the sounddevice library [1]. This library allows to
use sound devices for recording and playback. The library
is based on the PortAudio library [2].

Example 1: simple playback

from dltb.thirdparty.sounddevice import SoundPlayer
player = SoundPlayer()
player.play('examples/win_xp_shutdown.wav')


Example 2: backgrounded playback

from dltb.thirdparty.sounddevice import SoundPlayer
player = SoundPlayer(backgrounded=True)
player.play('examples/win_xp_shutdown.wav')

FIXME[experiments]: 
player.sound = '/home/ulf/projects/examples/mime/audio/wav/le_tigre.wav'


Remarks
-------

Sounddevice seems to automatically set ALSA as default, which can
result in problems, if other programs also use ALSA for output.


References
----------
[1] https://python-sounddevice.readthedocs.io
[2] http://www.portaudio.com

"""
# FIXME[bug]: I am experiences frequent crashes on my office computer
# (Ubuntu 16.04):
#    src/hostapi/alsa/pa_linux_alsa.c:3636:
#        PaAlsaStreamComponent_BeginPolling:
#            Assertion `ret == self->nfds' failed.
#
#
#  cat /proc/asound/version
#  Advanced Linux Sound Architecture Driver Version k4.4.0-179-generic.
#
#  aplay --version
#  aplay: version 1.1.0 by Jaroslav Kysela <perex@perex.cz>
#
#  pulseaudio --version
#  pulseaudio 8.0
#
#  python -c "import sounddevice; print(sounddevice.__version__)"
#  0.4.0


# standard imports
from typing import Union, Optional
from types import ModuleType
import ctypes
import sys
import time
import logging
import threading
import importlib
import multiprocessing as mp

# third party imports
import numpy as np
# Avoid importing sounddevice in the main process to allow for backgrounding
# import sounddevice as sd

# toolbox imports
from ..base.sound import Sound, SoundReader
from ..base.sound import SoundPlayer as SoundPlayerBase
from ..base.sound import SoundRecorder as SoundRecorderBase
from ..base.background import Backgroundable
from ..util.error import handle_exception
from .. import config


# logging
LOG = logging.getLogger(__name__)


class SoundPlayer(Backgroundable, SoundPlayerBase):
    """An implementation of a :py:class:`SoundPlayerBase` based on
    the `sounddevice` library.

    Configuring the OutputStream
    ----------------------------
    Although it is possible to configure the OutputStream upon
    construction, this configuration may be constrained by the
    underlying sound device.  For example, under Linux with ALSA,
    the PCM output device is often configured to use a samplerate
    of 44,100 Hz, and trying to create a stream that (significantly)
    deviates from that value, may result in a failure.
    Hence we don't try to change the samplerate and instead
    adapt the sound data during playback if necessary.

    Starting and stopping
    ---------------------
    Playback is done in a background thread initiated by the
    `OutputStream.start()` method.  Official ways to stop
    playback are:
    (1) reaching the end of the sound,
    (2) invoking the stop method.
    In both cases, the `_stopped` event is set.  Idally, also
    the `stream.stopped` flag should be set (by calling `stream.stop()`
    or `stream.abort()`) but this is not allowed in the background
    thread (and hence from the stream callbacks).  Hence, when reaching
    the reaching the end of sound, the stream may end up in state where
    neither `stream.active` nor `stream.stopped` is set.  Such a stream
    can not be started, however, it may be put back in a sane state
    by calling `stream.stop()`.

    Multiprocessing (not working!)
    ---------------
    In some situations, e.g., when running with an active graphical
    user interface, one may experience problems when the playback buffer
    is filled too slowly (buffer underflow).  To avoid such problems,
    one may run the `sounddevice` player in its own process.
    Multiprocessing can be initiated by passing the `multiprocessing`
    flag upon creation of the :py:class:`SoundPlayer`, or by setting
    the :py:prop:`multiprocessing` property.  Trying to change
    `multiprocessing` during playback may result in a `RuntimeError`.

    When running in multiprocessing mode, some points have to be taken
    into account:
    * the `sounddevice` Stream has to be created in the main process
      (where the `sounddevice` module is imported). Trying to create it in
      another process will result in an error.
    * once started, the subprocess will not reflect any changes of variables
      in the main process.  Hence any change in state has to be announced
      to the subprocess via some form of inter process communication.

    Problem: starting the stream in the new process does not show any effect
    """
    _stream = None  # Optional[sd.OutputStream]
    _process: Optional[mp.Process] = None

    # sd: the 'sounddevice' module. Avoid importing sounddevice in the
    # main process to allow for backgrounded.
    sd: Optional[ModuleType] = None

    _sound_array: Optional[np.ndarray] = None
    _sound_samplerate: Optional[float] = None

    def __init__(self, samplerate: Optional[float] = None,
                 channels: Optional[int] = 2,
                 **kwargs) -> None:
        self._lock = mp.RLock()

        self._started = mp.Event()
        self._started.clear()
        
        self._stopped = mp.Event()
        self._stopped.set()

        self._position_value = mp.Value('d', -1)

        super().__init__(**kwargs)
        # Create stream only when needed, to allow for switch to
        # background process
        # self._create_stream(samplerate, channels)

    def _attributes_str(self) -> str:
        stream = self._stream
        if stream is None:
            stream_info = "None"
        else:
            stream_info = (f"Stream(active={stream.active}, "
                           f"stopped={stream.stopped}, "
                           f"closed={stream.closed})")
        return "sounddevice: "+ super()._attributes_str() + \
            (f", samplerate={self.samplerate}, channels={self.channels}, "
             f"stream={stream_info}, "
             f"blocking={self._blocking}, "
             f"stopped={self._stopped.is_set()}")

    def __repr__(self) -> str:
        result = ("Sounddevice SoundPlayer["
                  f"started={self._started.is_set()}, "
                  f"stopped={self._stopped.is_set()}, "
                  f"position={self.position}]")
        stream = self._stream
        if stream is None:
            result += " without output stream"
        else:
            result += (f"(active={stream.active}, stopped={stream.stopped}, "
                       f"closed={stream.closed})")
        return result

    #
    # overwriting methods from the superclass
    #

    @property
    def position(self) -> Optional[float]:
        position = self._position_value.value
        return None if position == -1 else position

    @position.setter
    def position(self, position: Optional[float]) -> None:
        # as we may set the position from within the playback loop,
        # we lock the operation to avoid interferences.
        with self._lock:
            changed = position != self.position
            if changed:
                self._position_value.value = \
                    -1 if position is None else position
        if changed:
            self.change('position_changed')

    def _set_sound(self, sound: Optional[Sound]) -> None:
        """Actual implementation of the sound setter, to be overwritten by
        subclasses.
        """
        if self.backgrounded:
            if sound is None:
                self.set_background_attribute('_sound_array', None)
                self.set_background_attribute('_sound_samplerate', None)
            else:
                self.set_background_attribute('_sound_array', sound.data)
                self.set_background_attribute('_sound_samplerate',
                                              sound.samplerate)
        super()._set_sound(sound)

    def _background_update_sound(self) -> None:
        if self._sound_array is None:
            if self._sound is not None:
                LOG.info("background sound update: None")
                self._sound = None
            return  # done - no sound
        if self._sound is not None and self._sound.data is self._sound_array:
            LOG.debug("background sound update: up to date.")
            return  # already up to date
        LOG.debug("background sound update: array=%s, samplerate=%s",
                  self._sound_array.shape, self._sound_samplerate)
        self._sound = \
            Sound(array=self._sound_array, samplerate=self._sound_samplerate)
        LOG.info("background sound update: %s (same=%s)", self._sound,
                 self._sound.data is self._sound_array)

    #
    # additional sounddevice methods
    #
    
    @property
    def samplerate(self) -> float:
        """Samplerate to be used for playback.
        """
        return None if self._stream is None else self._stream.samplerate

    @property
    def channels(self) -> int:
        """Number of channels to be used for playback.
        """
        return None if self._stream is None else self._stream._channels

    #
    # playback commands
    #

    def _pause(self) -> None:
        LOG.debug("SoundDevicePlayer: pausing playback loop")
        self._stopped.set()
        if self.backgrounded or self.in_background:
            # stream has to be stopped in the background process
            self.trigger_background_action()
        self._stop_stream()

    def _playing(self) -> bool:
        """Determine whether the player is currently playing.
        """
        # Remark: The sounddevice stream.active flag reflects the
        # actual playback Thread, which may lag a bit behind the
        # invocation of the start and stop/abort methods of the
        # SoundPlayer. To get a consistent behaviour with the
        # `state_changed` notifications, which are triggered upon
        # start and stop/abort call, we use the _started flag instead
        # of the stream.active flag.
        return self._started.is_set() and not self._stopped.is_set()

    def _play(self, blocking: bool) -> None:
        """Run the `sounddevice` stream loop. This will start a background
        thread, periodically invoking :py:meth:`_play_block`.

        Arguments
        ---------
        blocking:
            If `True`, the method will block and wait until the playback
            has finished.
        """
        LOG.info("SoundDevicePlayer: starting playback (blocking=%s,"
                 "backgrounded=%s/%s)", blocking,
                 self.backgrounded, self.in_background)

        if self.backgrounded:
            # we are operating in backgrounded mode and this method is
            # called in the main process - signal the background
            # process to start playing.  This will in turn invoke
            # _play() in the background process.
            LOG.info("SoundDevicePlayer: starting playback loop "
                     "at position %s (blocking=%s, backgrounded=%s)",
                     self.position, blocking, self.backgrounded)
            self._stopped.clear()
            #self._started.set()
            self.trigger_background_action()

            self._started.wait()
            self._player_started()

            if blocking:
                try:
                    self._stopped.wait()
                except KeyboardInterrupt:
                    LOG.warning("KeyboardInterrupt (SoundDevicePlayer._play() "
                                "in process '%s') - stopping playback.",
                                mp.current_process().name)
                    self._stop()
            return

        LOG.info("SoundDevicePlayer: starting playback loop (blocking=%s)",
                 blocking)

        if self._stream is None:
            self._create_stream(samplerate=None, channels=2)

        if self._stream.active:
            # stream was already started
            LOG.info("SoundDevicePlayer: stream was already started, will "
                     "simply let it run ... (blocking=%s)", blocking)
        else: 
            if not self._stream.stopped:
                # stream is in an unsane state (neither active nor stopped)
                # and will not start - cure it by making setting it into
                # the stopped state
                self._stream.stop()

            LOG.info("SoundDevicePlayer: Playing sound in process '%s': %s",
                     mp.current_process().name, self.sound.data.shape)

            self._stopped.clear()
            self._block_count = 0
            self._next_debug_position = 0

            # start the actual playback
            # problem[multiprocessing]: starting the stream from
            # subprocess will hang (but not throw any exception)
            self._stream.start()

            # inform observers that playback was started
            self._started.wait()
            self._player_started()

        if blocking:
            try:
                LOG.info("SoundDevicePlayer: Running in blocking mode - "
                         "waiting for 'stopped' event")
                self._stopped.wait()
            except KeyboardInterrupt:
                # Playback/recording may have been stopped with
                # a `KeyboardInterrupt` - make sure the stream
                # is closed
                LOG.warning("KeyboardInterrupt during SoundDevicePlayer "
                            "playback: stopping playback")
                raise
            finally:
                self._stop()
                self._player_stopped()

        LOG.info("SoundDevicePlayer: Leaving play loop (blocking=%s)",
                 blocking)

    def _play_block(self, outdata: np.ndarray, frames: int,
                    sd_time, status) -> None:
        # status: sd.CallbackFlags
        """Callback to be called by the output stream to play a
        block of frames.

        Arguments
        ---------
        outdata: np.ndarray
            An array of shape (frames, channels) and dtype float32.
            This is a buffer provided by the OutputStream in which
            the next block of output data should be stored.
        frames: int
            The number of frames to be stored. This should be the
            sames as len(outdata)
        sd_time:
            A CFFI structure with timestamps indicating the
            ADC capture time of the first sample in the input buffer
            (`time.inputBufferAdcTime`),
            the DAC output time of the first sample in the output buffer
            (`time.outputBufferDacTime`) and
            the time the callback was invoked (`time.currentTime`).
            These time values are expressed in seconds and are
            synchronised with the time base used by `time` for the
            associated stream.
        status:
            The status indicates potential problems. It has the following
            boolean attributes: `input_overflow`, `input_underflow`,
            `output_underflow`, `output_underflow`, and `priming_output`.
        """
        start_time = time.time()

        if status:
            LOG.warning("SoundDevicePlayer: play_block: "
                        "status = %s (at time %.8fs)", status,
                        sd_time.currentTime - sd_time.outputBufferDacTime)
            # The time difference is always 0 (both times are identical).
            # Trying to get the self._stream.time for reference results in
            # a PortAudioError('Error getting stream time')
            # -> "With the exception of cpu_load it is not permissible to
            #    call PortAudio API functions from within the stream callback."

        # warn_only_few_frames = 0
        # if frames < warn_only_few_frames:
        #     LOG.warning("SoundDevicePlayer: play_block: stream requests "
        #                 "only %d frames.", frames)

        self._block_count += 1
        position = self.position
        reverse = self.reverse
        samplerate = self.samplerate
        duration = frames / samplerate
        
        if position is None:
            LOG.debug("play block: no position - will not play")
            wave_frames, valid_frames = 0, 0
            outdata[wave_frames:, :].fill(0)
        elif False:  # new Sound.sample method - may be more efficient (no intermediate array) and less buggy ;-) - but still incomplete, see below ...
            start = position
            if start >= self.end and self.loop:
                start = self.start
            end = min(position+duration, self.end)

            # FIXME[todo]: reverse sampling
            (_, wave_frames), _ = \
                self._sound.sample(start=position, output=outdata,
                                   samplerate=samplerate)

            valid_frames = min(wave_frames, frames)
            if wave_frames != frames:
                LOG.warning("SoundDevicePlayer: play_block: invalid frames: "
                            "wave/frames=%d/%d", wave_frames, frames)
        else:
            # obtain the relevant sound data
            if not reverse:
                start = position
                end = min(position+duration, self.end)
            else:
                start = max(self.start, position-duration)
                end = position

            wave = self._sound[start:end:samplerate]
            wave_frames = len(wave)
            if wave_frames != frames:
                LOG.warning("SoundDevicePlayer: play_block: invalid frames: "
                            "wave/frames=%d/%d", wave_frames, frames)

            # provide the wave to the OutputStream via the outdata array.
            valid_frames = min(wave_frames, frames)
            if not reverse:
                outdata[:valid_frames, :] = wave[:valid_frames]
            else:
                outdata[:valid_frames, :] = wave[valid_frames-1::-1]

            # pad missing data with zeros
            if wave_frames < frames:
                outdata[wave_frames:, :].fill(0)

        # If we have not obtained any data (wave_frames == 0) we will stop
        # playback here.
        if not reverse:
            new_position = end if wave_frames > 0 else None
            if new_position is not None and new_position >= self.end:
                new_position = self.start if self.loop else None
            if new_position is None:
                print(f"play_block: new_position={new_position} ({position}), "
                      f"wave_frames/frames={wave_frames}/{frames}, "
                      f"reverse={reverse}, end={end}, "
                      f"start/end={self.start}/{self.end}")
        else:
            new_position = start if wave_frames > 0 else None
            if new_position is not None and new_position <= self.start:
                new_position = self.end if self.loop else None
        # We have to avoid overwriting a change of position
        # that may have occured in the meantime (by some other thread)
        with self._lock:
            # only update position, if position was not externally updated
            # in the meantime ...
            if self.position == position:
                self.position = new_position

        if new_position is None:
            # We cannot call _stream.stop() (or _stream.abort()) from
            # within the sub-thread (also not from finished_callback)
            # this will cause some error in the underlying C library).
            # The official way to stop the thread from within is to
            # raise an exception:
            raise self.sd.CallbackStop()

        if new_position > self._next_debug_position:
            processing_time = time.time() - start_time
            LOG.debug("output block[%d/%d frames/%.5fs]: "
                      "processing time=%.5f (%d%%), new_position=%.3f",
                      valid_frames, frames, duration,
                      processing_time, int((processing_time/duration) * 100),
                      float('nan') if new_position is None else new_position)
            self._next_debug_position += 1  # debug every second

        # after having played the first block, hopefully the stream is
        # running smoothly - inform waiting threads that playback has
        # started so that they can continue their work ...
        if not self._started.is_set() and not self._stopped.is_set():
            self._started.set()

    def _finished(self) -> None:
        """The finished_callback is called once the playback thread
        finishes (either due to an exception in the inner loop or by
        an explicit call to stream.stop() from the outside).

        Note: the method is called in the same process and the same thread
        in which the stream loop was running.
        """
        LOG.info("SoundDevicePlayer: finished callback was invoked.")
        # inform main thread that we finished
        self._stopped.set()
        self._player_stopped()

    #
    # sounddevice Stream related
    #

    def _create_stream(self, samplerate: Optional[float],
                       channels: Optional[int]) -> None:
        """Create a new output stream.
        """
        LOG.info("Creating Stream in process '%s'", mp.current_process().name)
        cls = type(self)
        if cls.sd is None:
            LOG.info("Importing sounddevice in process '%s'",
                     mp.current_process().name)
            cls.sd = importlib.import_module('sounddevice')
        # Agruments
        # latency: float or {‘low’, ‘high’} or pair thereof, optional)
        #    The desired latency in seconds. The special values 'low' and
        #    'high' (latter being the default) select the device’s default
        #    low and high latency, respectively (see query_devices()).
        #    'high' is typically more robust (i.e. buffer under-/overflows
        #    are less likely), but the latency may be too large for
        #    interactive applications.
        #
        if self._stream is not None:
            LOG.warning("SoundDevicePlayer: trying to recreate output stream. "
                        "Operation skipped, continue using existing stream.")
            return  # skip operation 

        LOG.info("SoundDevicePlayer: creating an output stream"
                 " (samplerate=%s, channels=%s)", samplerate, channels)
        try:
            self._stream = \
                self.sd.OutputStream(samplerate=samplerate,
                                     channels=channels,
                                     callback=self._play_block,
                                     finished_callback=self._finished)
        except self.sd.PortAudioError as error:
            raise RuntimeError("Failed to create OutputStream(samplerate="
                               f"{samplerate}, channels={channels})") \
                               from error

    def _stop_stream(self) -> None:
        # Here we could either call stream.stop() or stream.abort().
        # The first would stop acquiring new data, but finish processing
        # buffered data, while the second would abort immediately.
        # For the sake of a responsive interface, we choose abort here.
        if self._stream is None:
            return  # no stream
        if self._stream.active:
            self._stream.abort(ignore_errors=True)
        elif not self._stream.stopped:
            self._stream.stop(ignore_errors=True)

    def _close_stream(self) -> None:
        """Close the output stream.
        """
        if self._stream is not None:
            LOG.info("SoundDevicePlayer: closing output stream")
            self._stream.close(ignore_errors=True)
            del self._stream


    #
    # Backgroundable
    #

    @property
    def backgroundable(self) -> bool:
        return 'sounddevice' not in sys.modules and \
            super().backgroundable

    def _background_action(self) -> None:
        """This method may be called in a background thread.
        """
        super()._background_action()
        self._background_update_sound()
        if self._started.is_set():
            # We will run in non-blocking mode so that the background
            # process immediatly returns to its main loop.
            # This has the effect, that playback cannot be stopped
            # by only setting the '_stopped' event - in addition one has
            # to also set the '_loop_event'. 
            # The stop() method will handle this correctly - hence always
            # call that method instead of manually setting the event.
            self._play(blocking=False)

    def _stop_background_action(self) -> None:
        # stop current playback (if running), as this may be in blocking
        # event processing
        self._started.clear()
        self._stop_stream()
        super()._stop_background_action()

    #
    # debugging
    #

    @staticmethod
    def debug() -> None:
        from ..util.logging import TerminalFormatter
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(TerminalFormatter())
        LOG.addHandler(handler)
        LOG.setLevel(logging.DEBUG)
        LOG.debug("Outputting debug messages")


class SoundRecorder(SoundRecorderBase):
    """A :py:class:`SoundRecorder` based on the Python sounddevice
    library.
    """

    # sd: the 'sounddevice' module. Avoid importing sounddevice in the
    # main process to allow for backgrounded.
    sd: ModuleType = None

    def __init__(self, channels: int = None, samplerate: float = None,
                 device: Union[int, str] = None, **kwargs):
        super().__init__(**kwargs)

        if channels is None:
            channels = 2 if self._sound is None else self._sound.channels
        if samplerate is None:
            samplerate = (44100 if self._sound is None else
                          self._sound.samplerate)
        # device: input device (numeric ID or substring)
        # device_info = sd.query_devices(device, 'input')
        # samplerate = device_info['default_samplerate']
        cls = type(self)
        if cls.sd is None:
            cls.sd = importlib.import_module('sounddevice')
        
        self._stream = self.sd.InputStream(device=device, channels=channels,
                                           samplerate=samplerate,
                                           callback=self._record_block,
                                           finished_callback=self._finished)

    def __str__(self) -> str:
        return ("sounddevice-based SoundRecorder"
                f"(samplerate={self.samplerate}, channels={self.channels})")

    @property
    def samplerate(self) -> float:
        """Samplerate used for recording.
        """
        return self._stream.samplerate

    @property
    def channels(self) -> int:
        """Number of channels to be recorded.
        """
        return self._stream.channels

    @property
    def recording(self) -> bool:
        return self._stream.active

    def _record(self) -> None:
        """
        """
        LOG.info("Recorder: samplerate=%f", self.samplerate)
        LOG.info("Recorder: sound=%s", self.sound)

        LOG.info("Recorder: starting stream")
        self._stream.start()
        LOG.info("Recorder: stream started")

    def _FIXME_old_record(self) -> None:
        # This implementation assumes a plotter (like the
        # MatplotlibSoundPlotter), that has to start its own Thread
        # (as the matplotlib.animation.FuncAnimation class does).
        # The context manager (with self._stream) will start
        # the sounddevice.InputStream in its own Thread, and then
        # execute the inner block.
        #
        # # the context manager will start the stream task
        # with self._stream:
        #     # this will start the plotter and block until the
        #     # plotter has finished - hence we have to explicitly
        #     # stop the plotter, once the stream has finished.
        #     self._plotter.start_plot()

        # stream = sd.InputStream(device=device, channels=channels,
        #    samplerate=samplerate, callback=audio_callback)
        # ani = FuncAnimation(fig, update_plot, interval=interval, blit=True)
        # with stream:
        #    plt.show()
        pass

    def _record_block(self, indata, _frames, _time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            LOG.debug("SoundDeviceRecorder: %s", status)

        # append new data to the sound object
        self._sound += indata

    def _finished(self) -> None:
        LOG.info("SoundDeviceRecorder: finished")

    def _stop(self) -> None:
        """Stop ongoing sound recording.
        """
        # Here we could either call stream.stop() or stream.abort().
        # The first would stop acquiring new data, but finish processing
        # buffered data, while the second would abort immediately.
        # In order to not loose any data, we choose stop here.
        LOG.info("SoundDeviceRecorder: aborting stream")
        # self._stream.abort()
        if self._stream.active:
            self._stream.stop()
        LOG.info("SoundDeviceRecorder: stream aborted")


def main() -> None:
    """Main program to quickly test the sounddevice play functionality.
    """
    sd = importlib.import_module('sounddevice')
    print(sd.query_hostapis())
    print(sd.query_devices())
    reader = SoundReader()
    sound = reader.read(config.sound_example)
    array, samplerate = sound._array, sound.samplerate
    print(f"Playing sound example '{config.sound_example}' {array.shape}, "
          f"samplerate={sound.samplerate} with sounddevice.play() ...")
    sd.play(array, samplerate, blocking=True)  # device='default'


if __name__ == '__main__':
    main()
