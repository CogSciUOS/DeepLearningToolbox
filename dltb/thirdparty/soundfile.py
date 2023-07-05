"""Access to the soundfile library

"""
# standard imports
from typing import Optional

# third party imports
import soundfile as sf

# toolbox imports
from ..base.sound import (Sound,
                          SoundReader as SoundReaderBase,
                          SoundWriter as SoundWriterBase)


class SoundReader(SoundReaderBase):
    """A :py:class:`SoundReader` based on the `soundfile` library.
    This python module is based on libsndfile, which provides
    access to files in many audio formats, including 'wav' and
    'mp3'.
    """

    def __str__(self) -> str:
        return "soundfile-based SoundReader"

    def read(self, filename: str, channels: Optional[int] = None,
             samplerate: Optional[float] = None,
             endian: Optional[str] = None) -> Sound:

        # FIXME[todo]: soundfile.read() can also read from file like
        # objects, e.g., open files:
        #
        #   from urllib.request import urlopen
        #   url = "http://tinyurl.com/shepard-risset"
        #   data, samplerate = sf.read(io.BytesIO(urlopen(url).read()))

        # FIXME[todo]: soundfile.read() can auto-detect file type of most
        # soundfiles and obtain the correct metadata (channels and samplerate)
        # However, for raw audio files this is not possible, one has
        # to provide thos values explicitly:
        #
        #   data, samplerate = sf.read('myfile.raw',
        #                              channels=1, samplerate=44100,
        #                              subtype='FLOAT')

        # data is in the (frames, channels) format
        array, samplerate = sf.read(filename)
        return Sound(array=array, samplerate=samplerate)


class SoundWriter(SoundWriterBase):

    def __str__(self) -> str:
        return "soundfile-based SoundWriter"

    def write(self, sound: Sound, filename: str) -> None:
        """
        """
        sf.write(filename, data=sound.data, samplerate=sound.samplerate)
