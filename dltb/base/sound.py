"""Definition of an abstract sound interface.

Examples:

>>> from dltb.base.sound import Sound, SoundPlayer, SoundDisplay
>>> sound = Sound(sound="examples/win_xp_shutdown.wav")
>>> player = SoundPlayer()
>>> player.play(sound)
>>> display = SoundDisplay()

"""
# FIXME[todo]: some conceptual work has to be done on
# synchronous/asynchronous use for reading and writing,
# but especially for recording and playback!
#
# FIXME[todo]: allow time indexing of sound (module util.timing)
# FIXME[todo]: support frames and subsampling


# standard imports
from typing import Union, Optional
from abc import ABC
import logging
from pathlib import Path

# third party imports
import numpy as np

# toolbox imports
from .observer import Observable
from .implementation import Implementable
from .data import Data
from .gui import SimpleDisplay

# logging
LOG = logging.getLogger(__name__)


# Soundlike is intended to be everything that can be used as
# a sound.
#
# np.ndarray:
#    The raw sound data
# str:
#    A URL.
Soundlike = Union[np.ndarray, str, Path]


class Sound(Data):
    """A :py:class:`Sound` sound represents a piece of sound.
    It has a duration (time in seconds) and a sampling rate
    (in Hertz).


    Attributes
    ----------
    data: np.ndarray
        The actual sound wave data. The array has dtype np.float32
        and shape (frames, channels). We have choosen the "channels last"
        layout (that is frames on axis 0 and channels on axis 1)
        as this seems to be the most common layout. It allows to
        easily access subwaves by indexing data[start:end]. However
        extracting a channel would require data[:, channel].

    channels: int
        The number of channels of this :py:class:`Sound`.

    samplerate: float
        The samplerate of this :py:class:`Sound`. Typical rates
        are 44,100.0 Hz (e.g. used for Audio-CDs).

    duration: float
        The duration of this :py:class:`Sound` in seconds.

    frames: int
        The length of this :py:class:`Sound` in frames, that is
        in discrete time points (this is duration times samplerate).

    """

    @staticmethod
    def as_sound(sound: Optional[Soundlike]) -> Optional['Sound']:
        """Create a `Sound` from a `Soundlike` object.  Will
        return `None` for `None` argument.
        """
        if sound is None or isinstance(sound, Sound):
            return sound
        return Sound(sound=sound)

    def __new__(cls, sound: Optional[Soundlike] = None,
                array: Optional[np.ndarray] = None, copy: bool = False,
                samplerate: int = 44100, channels: Optional[int] = None,
                **kwargs) -> None:
        if isinstance(sound, Sound) and not copy:
            return sound  # just reuse the given Sound instance

        if isinstance(sound, Path):
            sound = str(sound)
        if isinstance(sound, str):
            LOG.debug("Loading sound '%s'.", sound)
            return SoundReader.read_with_default_reader(filename=sound,
                                                        samplerate=samplerate,
                                                        channels=channels)

        return super().__new__(cls, sound, array, copy, **kwargs)

    def __init__(self, sound: Optional[Soundlike] = None,
                 array: Optional[np.ndarray] = None, copy: bool = False,
                 samplerate: int = 44100, channels: Optional[int] = None,
                 **kwargs) -> None:
        if isinstance(sound, Sound) and not copy:
            return  # just reuse the given Sound instance (returned by __new__)
        if isinstance(sound, (Path, str)):
            return  # created Sound object with SoundReader in __new__

        super().__init__(**kwargs)

        # if data is given, infer information from that data
        if array is None and isinstance(sound, np.ndarray):
            array = sound

        if array is not None:
            # check array dimensions and channels
            if array.ndim == 1:
                array = array[:, np.newaxis]
            elif array.ndim == 2:
                if array.shape[0] < array.shape[1]:
                    array = array.T
            else: 
                raise ValueError("Invaled array shape for Sound: "
                                 f"{array.shape}")

            if channels is None:
                channels = array.shape[1]
            elif channels != array.shape[1]:
                raise ValueError("Inconsistent numbers of channels: "
                                 f"{channels} vs. {array.shape[1]}")

        channels = channels or 1
        self._samplerate = samplerate

        if array is not None and not copy:
            self._array = array
            self._data = array
        else:
            # create an initial array with space for 1 second of sound
            self._array = np.empty(shape=(self.samplerate, channels),
                                   dtype=np.float32)
            self._data = self._array[:0]

            if array is not None:
                self += array

    @property
    def frames(self) -> int:
        """The duration of the :py:class:`Sound` in seconds.
        """
        return self._data.shape[0]

    @property
    def channels(self) -> int:
        """The number of channels of this :py:class:`Sound`.
        """
        return self._data.shape[1]

    @property
    def duration(self) -> float:
        """The duration of the :py:class:`Sound` in seconds.
        """
        return (self.frames / self._samplerate
                if self._samplerate else float('nan'))

    @property
    def samplerate(self) -> int:
        """The samplerate of this :py:class:`Sound`.
        """
        return self._samplerate

    @property
    def data(self) -> np.ndarray:
        """The wave form of the :py:class:`Sound`.
        This is a numpy array of shape (frames, channels) of dtype
        np.float32 with sound values in range between -1.0 and 1.0.
        """
        return self._data

    def __getitem__(self, index: slice) -> np.ndarray:
        """Get sound wave for a given interval.

        Arguments
        ---------
        index: slice
            A (float based) slice providing start and end time
            (in seconds) and optionally a samplerate. If the samplerate
            is different than the original samplerate of this
            :py:class:`Sound`, the data will be resampled.

        Result
        ------
        data: np.ndarray
            An array of shape (frames, channels) containing the
            sound for the requested interval.
        """
        if isinstance(index, slice):
            if not 0. <= index.start <= self.duration:
                raise ValueError("Invalid start position ({start}) "
                                 "for indexing Sound object {self}.")
            if not 0. <= index.stop <= self.duration:
                raise ValueError("Invalid end position ({stop}) "
                                 "for indexing Sound object {self}.")

            index_start = round(index.start * self.samplerate)
            index_end = round(index.stop * self.samplerate)
            samplerate = index.step or self.samplerate

            # we will compute indices of sample points
            step = self.samplerate / samplerate
            round_step = round(step)

            if abs(round_step - step) < 0.001:
                # target samplerate is approximate divider of
                # source samplerate: we will use numpy slicing
                # (should be more efficient)
                return self._data[index_start:index_end:round_step]

            samples = round((index.stop - index.start) * samplerate)
            return self._data[np.linspace(index_start, index_end, samples,
                                          endpoint=False, dtype=np.int)]

        raise TypeError(f"Index ({index}) to Sound object has invalid type: "
                        f"{type(index)}")

    def __setitem__(self, index: slice, data: np.ndarray) -> None:
        """Set sound data for a given time interval.

        Arguments
        ---------
        index: slice
            A (float based) slice providing start and end time
            (in seconds) and optionally a samplerate for the data
            provided. If the data samplerate is different from the
            samplerate of this :py:class:`Sound`, the data will
            be resampled.
        data: np.ndarray
            The data of shape (frames, channels) to be inserted at
            the given interval.
        """
        if isinstance(index, slice):
            if not 0. <= index.start <= self.duration:
                raise ValueError("Invalid start position ({start}) "
                                 "for indexing Sound object {self}.")
            if not 0. <= index.stop <= self.duration:
                raise ValueError("Invalid end position ({stop}) "
                                 "for indexing Sound object {self}.")

            index_start = index.start * self.samplerate
            index_end = index.stop * self.samplerate

            # samplerate of the data to be inserted
            samplerate = index.step or self.samplerate

            # the length of the inserted data in this Sound
            length = len(data)
            if samplerate != self.samplerate:
                length *= (self.samplerate / samplerate)

            # we will compute indices of sample points
            step = self.samplerate / samplerate

            if abs(round(step) - step) < 0.001:
                # target samplerate is approximate divider of
                # source samplerate: we will use numpy slicing
                # (should be more efficient)
                self._insert(data[::step], start=index_start, end=index_end)
            else:
                points = int((index_end - index_start) / step)
                self._insert(data[np.linspace(index_start, index_end, points,
                                              dtype=np.int)],
                             start=index_start, end=index_end)

        raise TypeError(f"Index ({index}) to Sound object has invalid type: "
                        f"{type(index)}")

    def sample(self, samplerate: Optional[int] = None,
               start: Optional[float] = 0,
               output: Optional[np.ndarray] = None,
               samples: Optional[int] = None) -> None:
        """Sample from this sound.
        """
        if output is None:
            if samples is None:
                raise ValueError("Neither output nor samples provided.")
            output = np.asarray((samples, self.channels),
                                dtype=self._data.dtype)
        else:
            samples = len(output)

        if samplerate is None:
            samplerate = self.samplerate
            step = 1
        else:
            step = self.samplerate/samplerate
        round_step = round(step)
        index_start = round(start * self.samplerate)

        if abs(round_step - step) < 0.001:
            # target samplerate is approximate divider of
            # source samplerate: we will use numpy slicing
            # (should be more efficient)
            length = samples * round_step
            if index_start + length <= len(self._data):
                valid_samples = samples
            else:
                length = len(self._data) - index_start + round_step - 1
                valid_samples = length // round_step
            output[:valid_samples] = \
                self._data[index_start:index_start+length:round_step]
        else:
            length = round(samples * step)
            if index_start + length <= len(self._data):
                valid_samples = samples
            else:
                length = len(self._data) - index_start
                valid_samples = round(length / step)
            output[:valid_samples] = \
                self._data[np.linspace(index_start, index_start + length,
                                       valid_samples,
                                       endpoint=False, dtype=np.int)]
        if valid_samples < samples:
            output[valid_samples:].fill(0)
        return (0, valid_samples), output

    def level(self, blocks: int,
              start: Optional[float] = None,
              end: Optional[float] = None) -> np.ndarray:
        """Compute the block wise signal level.

        FIXME[todo]: one may provide a blocksize argument instead of
        the blocks argument. One may also consider supporting a block
        overlap argument.
        """
        index_start = 0 if start is None else int(start * self.samplerate)
        index_end = self.frames if end is None else int(end * self.samplerate)
        data = self._data[index_start:index_end]
        length = len(data)
        block_size = length // blocks
        squares = ((data[:blocks*block_size]**2).sum(axis=1) /
                   self.channels)
        blocks = squares.reshape((blocks, -1)).mean(axis=1)
        return np.sqrt(blocks)

    def __iadd__(self, other: Soundlike) -> None:
        """Add (append) another sound to this :py:class:`Sound`. The
        other sound should be compatible with this sound
        (with respect to sampling rate and number of channels).
        """
        if isinstance(other, Sound):
            if other.samplerate != self.samplerate:
                raise ValueError("Trying to add Sounds with different "
                                 "samplerates: "
                                 f"{self.samplerate} vs. {other.samplerate}")
            if other.channels != self.channels:
                raise ValueError("Trying to add Sounds with different numbers "
                                 "of channels: "
                                 f"{self.channels} vs. {other.channels}")
            new_data = other._data
        elif isinstance(other, np.ndarray):
            if other.ndim == 1:
                new_data = np.tile(other, reps=(self.channels, 1)).T
            elif other.shape[0] == self.channels:
                new_data = other.T
            elif other.shape[1] == self.channels:
                new_data = other
            elif other.shape[0] == 1:
                new_data = other.T.repeat(self.channels, axis=0)
            elif other.shape[1] == 1:
                new_data = other.repeat(self.channels, axis=1)
            else:
                raise ValueError("Trying to add Sounds with different numbers "
                                 "of channels: "
                                 f"{self.channels} vs. {other.shape[1]}")
        else:
            raise TypeError(f"Trying to add a non-Sound (type {type(other)})")

        self._append(new_data)
        return self

    def _append(self, data: np.ndarray) -> None:
        self._insert(data, start=self.frames)

    def _insert(self, data: np.ndarray, start: Optional[int] = None,
                end: Optional[int] = None) -> None:
        """Insert data into this Sound, potentially overwriting
        some part of this sound.
        """
        old_array = self._array
        old_frames = self.frames
        index_start = start or 0
        index_end = end or old_frames

        data_length = 0 if data is None else len(data)
        new_frames = old_frames + data_length - (index_end - index_start)
        if new_frames > len(self._array):
            new_length = max(new_frames, len(self._array)*2)
            new_array = np.empty((new_length, self.channels))
            if index_start > 0:
                new_array[0:index_start] = old_array[0:index_start]
        else:
            new_array = self._array
        if index_end < old_frames:
            new_array[index_start + data_length:new_frames] = \
                old_array[index_end:old_frames]
        if data_length > 0:
            new_array[index_start:index_start + data_length] = data
        self._array = new_array
        self._data = self._array[:new_frames]

    def __str__(self) -> str:
        return (f"Sound({self.channels} channels, "
                f"{self.duration:.2f} seconds at rate {self.samplerate})")


class SoundReader(Implementable):
    """A :py:class:`SoundReader` can be used to read :py:class:`Sound` objects.
    """

    _default_reader: Optional['SoundReader'] = None

    @classmethod
    def get_default_reader(cls) -> 'SoundReader':
        """Get the default `SoundReader`.  If none has been assigned yet,
        a new one will be created.
        """
        if cls._default_reader is None:
            cls._default_reader = SoundReader()
        return cls._default_reader

    @classmethod
    def read_with_default_reader(cls, *args, **kwargs) -> Sound:
        """Read a :py:class:`Sound` with the default `SoundReader`.
        """
        reader = cls.get_default_reader()
        return reader.read(*args, **kwargs)

    def read(self, filename: str, channels: Optional[int] = None,
             samplerate: Optional[float] = None,
             endian: Optional[str] = None) -> Sound:
        """Read a sound file and return it as :py:class:`Sound` object.

        Arguments
        ---------
        filename:
            The name of the file containing the sound data.
        channels:
            The number of channels the sound object should posses.
            If `None`, the channels of the sound file will be used.
        samplerate:
            The samplerate of the resulting :py:class:`Sound` object.
            If `None`, the samplerate of the sound file will be used.
        endian: str
            Either 'LITTLE' or 'BIG'. If no value is provided,
            the platform default will be used.

        Result
        ------
        sound:
            The :py:class:`Sound` object that was read from `filename`.
        """

    class Async(Observable):
        """Asynchronous sound reader.
        """

        def __init__(self, reader: 'SoundReader', **kwargs) -> None:
            super().__init__(**kwargs)
            self._reader = reader
            self._sound = None

        @property
        def sound(self) -> Sound:
            """The sound read by this sound reader.
            """
            return self._sound

        # @run
        def read(self, **kwargs) -> None:
            """Read sound asynchronously and store it in the
            :py:attr:`sound` property.
            """
            self._sound = self._reader.read(**kwargs)

    def async_reader(self) -> 'Async':
        """Obtain an asynchronous reader for this :py:class:`SoundReader`.
        """
        return self.Async(self)


class SoundWriter(Implementable):
    """A :py:class:`SoundWriter` can be used to write :py:class:`Sound`
    objects.
    """

    def write(self, sound: Soundlike, filename: str) -> None:
        """Write the given :py:class:`Sound` object to a file.
        The file format is inferred from the file suffix.
        """

    class Async(Observable):
        """Asynchronous sound writer.
        """

        def __init__(self, writer: 'SoundWriter', **kwargs) -> None:
            super().__init__(**kwargs)
            self._writer = writer

        # @run
        def write(self, **kwargs) -> None:
            """Asynchronously write some sound data.
            """
            self._writer.write(**kwargs)

    def async_writer(self) -> 'Async':
        """Obtain an asynchronous reader for this :py:class:`SoundReader`.
        """
        return self.Async(self)


class SoundPlayer(Observable, Implementable, method='player_changed', changes={
        'state_changed', 'position_changed', 'sound_changed'}):
    """A :py:class:`SoundPlayer` can play sounds on a suitable audio
    device.

    Playback
    --------
    The `SoundPlayer` is controlled by to main commands:
    :py:meth:`play` and :py:meth:`stop`.  `play` starts the playback
    and `stop` will end it.

    Playback can be done in blocking and non-blocking mode.  In blocking
    mode, the `play` method will only return once playback has finished
    (either by reaching the end of the sound or by due to interruption,
    i.e., some call to the :py:meth:`stop` method, e.g. from another
    thread or by some interrupt handler).

    The current state of the player can be inspected by the property
    :py:prop:`playing`.  The `SoundPlayer` will notify interested
    observers whenever this flag changes its value.

    Loop mode
    ---------
    A player can be set in loop mode, indicated by the property
    :py:prop:`loop`.  In loop mode, when reaching the end of the sound,
    the player will continue playing, starting again from the beginning.

    Replay interval
    ---------------
    A `SoundPlayer` can be assiged a replay interval by the :py:prop:`start`
    and :py:prop:`end` properties.  If set, the playback is restricted
    to this interval, including looping and reverting.

    Changes
    -------
    A :py:class:`SoundPlayer` is observable and can notify observers
    on the following events:

    state_changed:
        The state changed, with possible states being 'playing' and
        not playing (stopped).
    position_changed:
        The current position of this player has changed.
    sound_changed:
        The sound object to be played by this :py:class:`SoundPlayer`
        was exchanged.

    Arguments
    ---------
    sound:
        The sound to be played.
    blocking:
        A flag indicating if playback should be done blocking (synchronous)
        or non-blocking (asynchronous). This default behaviour can be
        overwritten when invoking the :py:meth:`play` method.
    loop:
        A flag indicating if the player should operate in loop mode.
    reverse:
        A flag indicating if the player should operate in reverse mode.
    """

    _position: float = None
    _start: float = None
    _end: float = None
    _blocking: bool = None
    _sound: Sound = None

    def __init__(self, sound: Optional[Soundlike] = None,
                 blocking: bool = True, loop: bool = False,
                 reverse: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sound = sound
        self.blocking = blocking
        self.loop = loop
        self.reverse = reverse

    def __del__(self) -> None:
        if self.playing:
            LOG.info("SoundPlayer: stop playback to allow for smooth deletion")
            self.stop()
        super().__del__()

    def __str__(self) -> str:
        return f"SoundPlayer({self._attributes_str()}"

    def _attributes_str(self) -> str:
        return (f"sound={self.sound}, "
                f"{'playing' if self.playing else 'stopped'}, "
                f"position={self.position} in {self.start} to {self.end}")

    @property
    def position(self) -> Optional[float]:
        """The current position of this :py:class:`SoundPlayer` in seconds.
        A value of `None` means that no position has been assigned.
        When playing, the position is increased in small steps
        (e.g., for each block put on the audio device).
        """
        return self._position

    @position.setter
    def position(self, position: Optional[float]) -> None:
        """Set the position. If set during playing, this will instruct the
        :py:class:`SoundPlayer` to continue playback at that position.
        """
        if position != self.position:
            self._set_position(position)

    def _set_position(self, position: float) -> None:
        """The actual assignment of position. This method can
        be overwritten in subclasses.
        """
        self._position = position
        self.change('position_changed')

    @property
    def start(self) -> float:
        """Start time of playing (in seconds). This will be used
        to initialize the no current position is set (if :py:attr:`position`
        is `None`) and for looping.
        """
        return self._start or 0

    @start.setter
    def start(self, start: float) -> None:
        """Set the start time of playing. If invoked during playing,
        it will only have an effect for the next play or when playing
        in loop mode.
        """
        self._start = start

    @property
    def end(self) -> float:
        """End time of playing (in seconds). None means no
        fixed end time, that is the :py:class:`Sound` is played
        until the end.
        """
        return self._end or (self._sound and self._sound.duration)

    @end.setter
    def end(self, end: float) -> None:
        """End time of playing (in seconds). None means no
        fixed end time, that is the :py:class:`Sound` is played
        until the end. When set during playing, the new setting
        will have an immediate effect.
        """
        self._end = end

    @property
    def blocking(self) -> bool:
        """A flag indicating if the player will operate in blocking
        mode (synchronous) of non-blocking mode (asynchronous).
        """
        return self._blocking

    @blocking.setter
    def blocking(self, blocking: bool) -> None:
        self._blocking = blocking

    @property
    def sound(self) -> Sound:
        """The sound assigned to this :py:class:`SoundPlayer`.
        """
        return self._sound

    @sound.setter
    def sound(self, sound: Optional[Soundlike]) -> None:
        """Assign a new sound to this :py:class:`SoundPlayer`.
        """
        if sound is not self._sound:
            self._set_sound(Sound.as_sound(sound))

    def _set_sound(self, sound: Optional[Sound]) -> None:
        """Actual implementation of the sound setter, to be overwritten by
        subclasses.
        """
        self._sound = sound
        self.change('sound_changed')

    @property
    def playing(self) -> bool:
        """A flag indicating if the player is currently playing.
        """
        return self._playing()

    def _playing(self) -> bool:
        """Report if the player is currently playing.
        """
        return False  # to be implemented by sublcassses

    def play(self, sound: Optional[Soundlike] = None,
             start: Optional[float] = None, end: Optional[float] = None,
             duration: Optional[float] = None, loop: Optional[bool] = None,
             reverse: Optional[bool] = None,
             # run: Optional[bool] = None,
             blocking: Optional[bool] = None) -> None:
        # pylint: disable=too-many-arguments
        """Play the given :py:class:`Sound`.

        This will initiate an asynchronous (non-blocking) playback of
        the sound, that is the actual playback will happen in a background
        thread.

        Arguments
        ---------
        sound: Sound
            The :py:class:`Sound` to play.
        start: float
            The start position in seconds.
        end: float
            The end position in seconds.
        duration: float
            The duration in seconds.
        loop: bool
            A flag indicating if playback should be looped.
        reverse: bool
            A flag indicating if playback should be backwards.
        """
        if self.playing:
            # raise RuntimeError("SoundPlayer is already playing.")
            LOG.warning("Trying to start a SoundPlayer that is "
                        "already playing.")
            return

        LOG.info("Playing sound: %s, start=%s, end=%s", sound, start, end)
        if sound is not None:
            self.sound = sound
        if self.sound is None:
            raise ValueError("No Sound provided for playing.")

        if start is not None:
            self._start = start
            self.position = start
        elif self.position is None:
            self.position = self._start or 0

        if end is not None and duration is not None:
            raise ValueError("Specification of end and duration.")
        if end is not None:
            self._end = end
        elif duration is not None:
            self._end = self._start + duration
        else:
            self._end = self.sound.duration

        if loop is not None:
            self.loop = loop
        if reverse is not None:
            self.reverse = reverse

        self._play(self._blocking if blocking is None else blocking)

    def _play(self, blocking: bool) -> None:
        """The actual implementation of starting the player
        (to be implemented by subclasses).
        """

    def stop(self) -> None:
        """Stop ongoing sound playback.
        """
        if not self.playing:
            # raise RuntimeError("SoundPlayer is not playing.")
            LOG.warning("Trying to stop a SoundPlayer that is not playing.")
            return
        self._stop()

        # After calling _stop, playing should have stopped. Make
        # sure that this is the case and inform the observers.
        if self.playing:
            LOG.warning("SoundPlayer did not stop playing.")

    def _stop(self) -> None:
        """The actual implementation of stopping the player
        (to be implemented by subclasses).
        """
        LOG.debug("SoundPlayer: stopping playback")
        self._pause()
        self.position = None

    def pause(self) -> None:
        """Pause playback. The audio playback is interrupted but the
        current position is kept so that playback can be continued
        later by calling :py:meth:`play`.
        """
        LOG.info("SoundPlayer: pausing playback")
        self._pause()

    def _pause(self) -> None:
        LOG.debug("SoundPlayer: pausing playback")

    def _player_started(self) -> None:
        """This method should be invoked once the player has started
        playing.  It will inform observers.
        """
        if not self.playing:
            LOG.warning("SoundPlayer was started but is not playing: %s",
                        self)
        LOG.info("SoundPlayer sending state_changed notification (started)")
        self.change('state_changed')

    def _player_stopped(self) -> None:
        """This method should be invoked once the player has stopped
        playing.  It will inform observers.
        """
        if self.playing:
            LOG.warning("SoundPlayer was stopped but is still playing: %s",
                        self)
        LOG.info("SoundPlayer sending state_changed notification (stopped)")
        self.change('state_changed')


class SoundRecorder(Observable, Implementable, changes={
        'state_changed', 'time_changed'}, method='recorder_changed'):
    """A :py:class:`SoundRecorder` provides functions for sound
    recording.
    """

    def __init__(self, sound: Optional[Sound] = None,
                 samplerate: int = 44100, channels: int = 2):
        super().__init__()
        self._sound = sound
        self._samlerate = samplerate
        self._channels = channels
        self._start = 0
        self._end = None
        self._position = None

    @property
    def recording(self) -> bool:
        """A flag indicating if this :py:class:`SoundRecorder`
        is currently recording.
        """
        return False  # to be implemented by sublcassses

    @property
    def sound(self) -> Sound:
        """The :py:class:`Sound` object to which this
        :py:class:`SoundRecorder` is writing its sound data.
        """
        return self._sound

    @sound.setter
    def sound(self, sound: Optional[Sound]) -> None:
        self._sound = sound

    def record(self, sound: Optional[Sound] = None,
               start: Optional[float] = None, end: Optional[float] = None,
               duration: Optional[float] = None) -> None:
        """Start recording a sound into a :py:class:`Sound` object.

        Arguments
        ---------
        sound: Sound
            The :py:class:`Sound` to play.
        start: float
            The start position in seconds. If None the recorded
            sound will be appended.
        end: float
            The end time of recording in seconds. If None, recording
            will not stop automatically. The default is to stop at the
            end of the given sound in case of inserting and to run
            to not stop in case of appending.
        duration: float
            The duration in seconds.
        loop: bool
            A flag indicating if playback should be looped.

        Examples
        --------

        Append to the current sound
        >>> record()

        Append to the current sound a new sound of length duration
        >>> record(duration=duration)

        Overwrite the interval [start, stop] with a new sound of same length
        >>> record(start=start, end=end)

        Overwrite the interval [start, start+duration] with a new sound
        of same length
        >>> record(start=start, duration=duration)

        Overwrite the interval [start, stop] with a new sound of
        length duration
        >>> record(start=start, end=end, duration)
        """
        # FIXME[todo]: appending works, but insert logic not fully
        # implemented yet
        if sound is not None:
            self._sound = sound
        elif self._sound is None:
            self._sound = Sound()
        self._start = start

        if end is not None and duration is not None:
            raise ValueError("You cannot specify both: end and duration!")
        if end is not None:
            self._end = end
        elif duration is not None:
            self._end = self._start + duration
        else:
            self._end = self._sound.duration

        if start is not None:
            self._position = start
        elif self._position is None:
            self._position = 0

        self._record()
        self.change('state_changed')

    def _record(self) -> None:
        """The actual implementation of starting the recorder
        (to be implemented by subclasses).
        """

    def stop(self) -> None:
        """Stop ongoing sound recording.
        """
        self._stop()
        self.change('state_changed')

    def _stop(self) -> None:
        """The actual implementation of stopping the player
        (to be implemented by subclasses).
        """


class SoundView(Implementable):
    """A graphical component capable of displaying a :py:class:`Sound`.
    """

    _sound: Optional[Sound] = None
    _player: Optional[SoundPlayer] = None
    _recorder: Optional[SoundRecorder] = None

    def __init__(self, sound: Optional[Soundlike] = None,
                 player: Optional[SoundPlayer] = None,
                 recorder: Optional[SoundRecorder] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sound = sound
        self.player = player
        self.recorder = recorder

    @property
    def sound(self) -> Optional[Sound]:
        """The :py:class:`Sound` to be displayed in this
        :py:class:`SoundView`.  If `None` the display will
        be cleaned.
        """
        return self._sound

    @sound.setter
    def sound(self, sound: Optional[Soundlike]) -> None:
        if sound is not self._sound:
            self._set_sound(Sound.as_sound(sound=sound))

    def _set_sound(self, sound: Optional[Sound]) -> None:
        # if self._sound is not None:
        #     self.unobserve(self._sound)
        self._sound = sound
        # if self._sound is not None:
        #     self.observe(self._sound)

        player = self.player
        if player is not None:
            player.sound = sound

        recorder = self.recorder
        if recorder is not None:
            recorder.sound = sound

    @property
    def player(self) -> SoundPlayer:
        """A :py:class:`SoundPlayer` observed by this
        :py:class:`SoundView`. Activities of the player may be
        reflected by this view and the view may contain
        graphical elements to control the player. The player
        may be `None` in which case such controls will be disabled.
        """
        return self._player

    @player.setter
    def player(self, player: Optional[SoundPlayer]) -> None:
        if player is not self._player:
            self._set_player(player)

    def _set_player(self, player: Optional[SoundPlayer]) -> None:
        self._player = player
        if player is not None:
            player.sound = self.sound

    @property
    def recorder(self) -> SoundRecorder:
        """A :py:class:`SoundRecorder` observed by this
        :py:class:`SoundView`. Activities of the recorder may be
        reflected by this view and the view may also contain
        graphical elements to control the recorder.
        """
        return self._recorder

    @recorder.setter
    def recorder(self, recorder: Optional[SoundRecorder]) -> None:
        if recorder is not self._recorder:
            self._set_recorder(recorder)

    def _set_recorder(self, recorder: Optional[SoundRecorder]) -> None:
        self._recorder = recorder
        if recorder is not None:
            recorder.sound = self.sound

    def sound_changed(self, sound: Sound, change: Sound.Change) -> None:
        """React to a change of the sound.
        """
        del sound, change  # unused arguments
        self.update()

    def player_changed(self, player: SoundPlayer,
                       change: SoundPlayer.Change) -> None:
        """React to a change of the player.
        """
        del player, change  # unused arguments
        self.update()

    def recorder_changed(self, recorder: SoundRecorder,
                         change: SoundRecorder.Change) -> None:
        """React to a change of the recorder.
        """
        del recorder, change  # unused arguments
        self.update()

    def update(self) -> None:
        """Update this `View`.
        """


class SoundDisplay(SimpleDisplay, Implementable, ABC):
    """A graphical element to plot sound waves.

    A sound viewer may be coupled with a
    :py:class:`SoundPlayer` and/or a :py:class:`SoundRecorder`.

    Arguments
    ---------
    sound:
        A sound to display in the newly created `SoundDisplay`.
    player:
        An :py:class:`SoundPlayer`, whose activities should
        be indicated by the `SoundDisplay`.  The support of displaying
        player activities is optional and be missing in some
        `SoundDisplay` implementations.
    """

    def __init__(self, sound: Optional[Soundlike] = None,
                 player: Optional[Union[SoundPlayer, bool]] = None,
                 recorder: Optional[Union[SoundRecorder, bool]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.sound = sound

        if isinstance(player, bool) and player:
            player = SoundPlayer()
        self.player = player

        if isinstance(recorder, bool) and recorder:
            recorder = SoundRecorder()
        self.recorder = recorder

    @property
    def sound(self) -> Optional[Sound]:
        """The :py:class:`Sound` object currently displayed in this
        `SoundDisplay`.  ``None`` if no sound is displayed.
        """
        return self.view.sound

    @sound.setter
    def sound(self, sound: Optional[Soundlike]) -> None:
        self.view.sound = sound

    @property
    def player(self) -> Optional[SoundPlayer]:
        """The :py:class:`SoundPlayer` associated with this `SoundDisplay`.
        `None` if there is no sound player.
        """
        return self.view.player

    @player.setter
    def player(self, player: Optional[SoundPlayer]) -> None:
        self.view.player = player

    @property
    def recorder(self) -> Optional[SoundRecorder]:
        """The :py:class:`SoundRecorder` associated with this `SoundDisplay`.
        `None` if there is no sound recorder.
        """
        return self.view.recorder

    @recorder.setter
    def recorder(self, recorder: Optional[SoundRecorder]) -> None:
        self.view.recorder = recorder

    def show(self, sound: Optional[Soundlike] = None, **kwargs) -> None:
        """Display a :py:class:`Sound` object.
        """
        if sound is not None:
            self.sound = sound
        super().show(**kwargs)
