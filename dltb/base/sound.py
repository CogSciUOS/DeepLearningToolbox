"""Definition of an abstract sound interface.

"""

# standard imports
import logging

# third party imports
import numpy as np

# toolbox imports
from base import Observable

# logging
LOG = logging.getLogger(__name__)


class Sound:
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

    def __init__(self, samplerate: int = 44100, channels: int = None,
                 data: np.ndarray = None) -> None:

        # if data is give, infer information from that data
        if data is not None:
            data_channels = 1 if data.ndim == 1 else min(data.shape)
            if channels is None:
                channels = data_channels
            elif channels != data_channels:
                raise ValueError("Inconsistent numbers of channels: "
                                 f"{channels} vs. {data_channels}")

        channels = channels or 1
        self._samplerate = samplerate

        self._array = np.empty(shape=(self.samplerate, channels),
                               dtype=np.float32)
        self._data = self._array[:0]

        if data is not None:
            self += data

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
        return self.frames / self._samplerate

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

            index_start = int(index.start * self.samplerate)
            index_end = int(index.stop * self.samplerate)
            samplerate = index.step or self.samplerate

            # we will compute indices of sample points
            step = self.samplerate / samplerate

            if abs(round(step) - step) < 0.001:
                # target samplerate is approximate divider of
                # source samplerate: we will use numpy slicing
                # (should be more efficient)
                return self._data[index_start:index_end:int(step)]

            points = int((index_end - index_start) / step)
            return self._data[np.linspace(index_start, index_end, points,
                                          dtype=np.int)]

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
                self._insert(data[::step], start=index_start, end=index_end)
            else:
                points = int((index_end - index_start) / step)
                self._insert(data[np.linspace(index_start, index_end, points,
                                              dtype=np.int)],
                             start=index_start, end=index_end)

        raise TypeError(f"Index ({index}) to Sound object has invalid type: "
                        f"{type(index)}")

    def level(self, blocks: int,
              start: float = None, end: float = None) -> np.ndarray:
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

    def __iadd__(self, other: 'Sound') -> None:
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

    def _insert(self, data: np.ndarray,
                start: int = None, end: int = None) -> None:
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


class SoundReader:
    """A :py:class:`SoundReader` can be used to read :py:class:`Sound` objects.
    """

    def read(self, filename: str, channels: int = None,
             samplerate: float = None, endian: str = None) -> Sound:
        """
        endian: str
            Either 'LITTLE' or 'BIG'. If no value is provided,
            the platform default will be used.
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


class SoundWriter:
    """A :py:class:`SoundWriter` can be used to write :py:class:`Sound`
    objects.
    """

    def write(self, sound: Sound, filename: str) -> None:
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


class SoundPlayer(Observable, method='player_changed', changes=[
        'state_changed', 'position_changed', 'sound_changed']):
    """A :py:class:`SoundPlayer` can play sounds on a suitable audio
    device.

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
    """

    def __init__(self, sound: Sound = None,
                 loop: bool = False, reverse: bool = False) -> None:
        super().__init__()
        self._sound = sound
        self._start = None
        self._end = None
        self._position = None
        self.loop = loop
        self.reverse = reverse

    @property
    def position(self) -> float:
        """The current position of this :py:class:`SoundPlayer` in secondes.
        A value of None means that no position has been assigned.
        When playing, the position is increased in small steps
        (e.g., for each block put on the audio device).
        """
        return self._position

    @position.setter
    def position(self, position: float) -> None:
        """Set the position. If set during playing, this will instruct the
        :py:class:`SoundPlayer` to continue playback at that position.
        """
        self._set_position(position)

    def _set_position(self, position: float) -> None:
        """The actual assignment of position. This method cancel
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
        return self._end or self._sound.duration

    @end.setter
    def end(self, end: float) -> None:
        """End time of playing (in seconds). None means no
        fixed end time, that is the :py:class:`Sound` is played
        until the end. When set during playing, the new setting
        will have an immediate effect.
        """
        self._end = end

    @property
    def sound(self) -> Sound:
        """The sound assigned to this :py:class:`SoundPlayer`.
        """
        return self._sound

    @sound.setter
    def sound(self, sound: Sound) -> None:
        """Assign a new sound to this :py:class:`SoundPlayer`.
        """
        self._set_sound(sound)

    def _set_sound(self, sound: Sound) -> None:
        """Actual implementation of the sound setter, to be overwritten by
        subclasses.
        """
        self._sound = sound
        self.change('sound_changed')

    @property
    def playing(self) -> bool:
        """A flag indicating if the player is currently playing.
        """
        return False  # to be implemented by sublcassses

    def play(self, sound: Sound = None, start: float = None, end: float = None,
             duration: float = None,
             loop: bool = None, reverse: bool = None) -> None:
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
            raise RuntimeError("SoundPlayer is already playing.")

        LOG.info("Playing sound: %s, start=%s, end=%s", sound, start, end)
        if sound is not None:
            self.sound = sound
        if self.sound is None:
            raise ValueError("No Sound provided for playing.")

        if start is not None:
            self._start = start
            self._position = start
        elif self._position is None:
            self._position = self._start or 0

        if end is not None and duration is not None:
            raise ValueError("Specification of end and duration.")
        if end is not None:
            self._end = end
        elif duration is not None:
            self._end = self._start + duration
        else:
            self._end = sound.duration

        if loop is not None:
            self.loop = loop
        if reverse is not None:
            self.reverse = reverse

        self._play()
        self.change('state_changed')

    def _play(self) -> None:
        """The actual implementation of starting the player
        (to be implemented by subclasses).
        """

    def stop(self) -> None:
        """Stop ongoing sound playback.
        """
        if not self.playing:
            raise RuntimeError("SoundPlayer is not playing.")
        self._stop()
        self.change('state_changed')

    def _stop(self) -> None:
        """The actual implementation of stopping the player
        (to be implemented by subclasses).
        """


class SoundRecorder(Observable, changes=['state_changed', 'time_changed'],
                    method='recorder_changed'):
    """A :py:class:`SoundRecorder` provides functions for sound
    recording.
    """

    def __init__(self, sound: Sound = None,
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

    def record(self, sound: Sound = None,
               start: float = None, end: float = None,
               duration: float = None) -> None:
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


class SoundDisplay:
    """A graphical element to plot sound waves.

    A sound viewer may be coupled with a
    :py:class:`SoundPlayer` and/or a :py:class:`SoundRecorder`.
    """

    def __init__(self, sound: Sound, **kwargs) -> None:
        super().__init__(**kwargs)
        self._sound = sound

    def show(self, sound: Sound) -> None:
        """Display a :py:class:`Sound` object.
        """
        raise NotImplementedError(f"{self.__class__.__name__} claims to "
                                  "be an SoundDisplay, but does not implement "
                                  f"the show method.")

    def sound_changed(self, sound: Sound, change) -> None:
        """React to a change of the sound.
        """
        if change:
            self.show(sound)
