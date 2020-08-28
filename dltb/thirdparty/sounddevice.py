"""Access to the sounddevice library. This library allows to
use sound devices for recording and playback. The library
is based on the PortAudio library [1].


[1] http://www.portaudio.com/
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
from typing import Union
import logging
import threading

# third party imports
import numpy as np
import sounddevice as sd

# toolbox imports
from ..base.sound import (SoundPlayer as SoundPlayerBase,
                          SoundRecorder as SoundRecorderBase)
from ..base import get_default_run

# logging
LOG = logging.getLogger(__name__)


class SoundPlayer(SoundPlayerBase):
    """An implementation of a :py:class:`SoundPlayerBase` based on
    the `sounddevice` library.
    """

    def __init__(self, samplerate: float = None, channels: int = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._event = threading.Event()

        if channels is None:
            channels = 2 if self._sound is None else self._sound.channels
        if samplerate is None:
            samplerate = (44100 if self._sound is None else
                          self._sound.samplerate)

        # _finishing: this is a hack - we need it to mark a stream that
        # finishes, but that has not yet been stopped (see method _finished).
        self._blocking = False
        self._finishing = False
        self._stream = None
        self._check_stream(samplerate=samplerate, channels=channels)

    def _check_stream(self, samplerate: float = None,
                      channels: int = None) -> None:
        """This function is a hack to fix a problem with an sounddevice
        streams in an unsane state: these streams have both, `active`
        and `stopped` flag (and also the `closed` flag) set to `False`.
        Such a state seems to occur when the stream is stopped
        (or aborted) from within the stream Thread (while stopping
        or aborting from another Thread seems to be ok).
        Such unsane streams can not be restarted by calling stream.start(),
        they seem to be dead (at least I did not find a way to revive
        them). As a workaround, we simply create a new stream here to
        replace the original one.
        """
        # Check the state of the current stream
        if self._stream is not None and not self._stream.closed:
            if self._stream.active or self._stream.stopped:
                return  # Stream seems to be ok

            LOG.warning("SoundDevicePlayer: "
                        "discovered unsane stream - creating a new one ...")
            # Stream seems to be dead - copy stream parameters
            samplerate = samplerate or self._stream.samplerate
            channels = channels or self._stream.channels
            self._stream.close()

        # create a new stream
        self._stream = sd.OutputStream(samplerate=samplerate,
                                       channels=channels,
                                       callback=self._play_block,
                                       finished_callback=self._finished)

    def _set_position(self, position: float) -> None:
        """Set the current playback position.
        """
        # as we set the position from within the playback loop,
        # we lock the operation to avoid interferences.
        with self._lock:
            super()._set_position(position)

    @property
    def playing(self) -> bool:
        return self._stream.active and not self._finishing

    @property
    def samplerate(self) -> float:
        """Samplerate to be used for playback.
        """
        return self._stream.samplerate

    @property
    def channels(self) -> int:
        """Number of channels to be used for playback.
        """
        return self._stream.channels

    def play(self, *args, run: bool = None, **kwargs):
        # we have to overwrite the super method to care for the 'run'
        # parameter (which would usually be done by the @run decorator):
        # as the stream playback is done in its own thread (and there
        # is no way to prevent this from happening), we will realize
        # a blocking call (run=False), explicitly waiting for the
        # playback to finish.
        self._blocking = not get_default_run(run)
        super().play(self, *args, run=False, **kwargs)

    def _play(self) -> None:
        """Start the actual playback in a background thread.
        """
        self._check_stream()
        # another hack:
        self._finishing = False
        self._event.clear()

        # this will start the background thread, periodically invoking
        # _play_block
        self._stream.start()

        print("Soundplayer: blocking:", self._blocking)
        if self._blocking:
            try:
                self._event.wait()
            finally:
                # Playback/recording may have been stopped with
                # a `KeyboardInterrupt` - make sure the stream
                # is closed
                self._stream.close(ignore_errors=True)

    def _play_block(self, outdata: np.ndarray, frames: int,
                    time, status: sd.CallbackFlags) -> None:
        """Callback to be called by the output stream.

        Arguments
        ---------
        outdata: np.ndarray
            An array of shape (frames, channels) and dtype float32.
            This is a buffer provided by the OutputStream in which
            the next block of output data should be stored.
        frames: int
            The number of frames to be stored. This should be the
            sames as len(outdata)
        """
        if status:
            LOG.debug("SoundDevicePlayer: status = %s", status)

        position = self._position
        reverse = self.reverse

        if position is None:
            LOG.debug("play block: no position")
            wave_frames = 0
        else:
            # obtain the relevant sound data
            samplerate = self.samplerate
            duration = frames / samplerate
            if not reverse:
                start = position
                end = min(position+duration, self.end)
            else:
                start = max(self.start, position-duration)
                end = position
            wave = self._sound[start:end:samplerate]
            wave_frames = len(wave)

            # provide the wave to the OutputStream via the outdata array.
            valid_frames = min(wave_frames, frames)
            if not reverse:
                outdata[:valid_frames, :] = wave[:valid_frames]
            else:
                outdata[:valid_frames, :] = wave[valid_frames-1::-1]
            LOG.debug("block, position=%f:.2, reverse=%s; "
                      "start=%f:.2, end=%f:.2, duration=%f:.4/%f:.4, "
                      "frames=%d/%d", position, reverse,
                      start, end, duration, end-start,
                      wave_frames, valid_frames)

        # pad missing data with zeros
        if wave_frames < frames:
            outdata[wave_frames:, :].fill(0)

        # If we have not obtained any data (wave_frames == 0) we will stop
        # playback here.
        if not reverse:
            new_position = end if wave_frames > 0 else None
            if new_position is not None and new_position >= self.end:
                new_position = self.start if self.loop else None
        else:
            new_position = start if wave_frames > 0 else None
            if new_position is not None and new_position <= self.start:
                new_position = self.end if self.loop else None
        # We have to avoid overwriting a change of position
        # that may have occured in the meantime (by some other thread)
        with self._lock:
            if self._position == position:
                super()._set_position(new_position)

        if new_position is None:
            # We cannot call _stream.stop() (or _stream.abort()) from
            # within the sub-thread (also not from finished_callback)
            # this will cause some error in the underlying C library).
            # The official way to stop the thread from within is to
            # raise an exception:
            raise sd.CallbackStop()

    def _finished(self) -> None:
        """The finished_callback is called once the playback thread
        finishes (either due to an exception in the inner loop or by
        an explicit call to stream.stop() from the outside).
        """
        # When the finihed_callback is called, the stream may not have
        # stopped yet - so when informing the observers, the playing
        # property may still report playing - to avoid this, we have
        # introduced the _finishing flag, that indicates that playback
        # has finished.
        self._event.set()
        if self.playing:
            self._finishing = True
            self.change('state_changed')

    def _stop(self) -> None:
        """Stop an ungoing playback.
        """
        # Here we could either call stream.stop() or stream.abort().
        # The first would stop acquiring new data, but finish processing
        # buffered data, while the second would abort immediately.
        # For the sake of a responsive interface, we choose abort here.
        if self._stream.active:
            self._stream.abort(ignore_errors=True)


class SoundRecorder(SoundRecorderBase):
    """A :py:class:`SoundRecorder` based on the Python sounddevice
    library.
    """

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

        self._stream = sd.InputStream(device=device, channels=channels,
                                      samplerate=samplerate,
                                      callback=self._record_block,
                                      finished_callback=self._finished)

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
