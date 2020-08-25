"""Access to the soundfile library

"""

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

    def read(self, filename: str) -> Sound:

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
        data, samplerate = sf.read(filename)
        return Sound(samplerate=samplerate, data=data)


class SoundWriter(SoundWriterBase):

    def write(self, sound: Sound, filename: str) -> None:
        """
        """
        sf.write(filename, data=sound.data, samplerate=sound.samplerate)
