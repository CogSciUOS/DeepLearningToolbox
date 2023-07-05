"""Implementation of a `YoutubeDownloader` using the `pytube` library.
"""
# standard imports
from typing import Optional, Union
from pathlib import Path
from io import BytesIO
import errno

# third-party imports
from pytube import YouTube

# toolbox imports
from dltb.types import Pathlike, as_path
from dltb.util.download import YoutubeDownloader


class PyTube(YoutubeDownloader):
    """A :py:class:`YoutubeDownloader` utilizing the third-party
    `pytube` library.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.youtube = YouTube(self.url)
        self.stream = self.youtube.streams\
            .filter(progressive=True, file_extension='mp4')\
            .order_by('resolution')\
            .first()

    def to_buffer(self, buffer: Optional[BytesIO] = None) -> BytesIO:
        if buffer is None:
            buffer = BytesIO()
        self.stream.stream_to_buffer(buffer)
        return buffer

    def to_file(self, filename: Optional[Pathlike] = None,
                overwrite: bool = False, skip_if_exists: bool = False) -> Path:
        """Download a video file from youtube.

        Arguments
        ---------
        video_file:
            Filename for storing the video.
        """
        if filename is None:
            filename = Path(self.stream.get_file_path())
        else:
            filename = as_path(filename).resolve()

        if filename.is_file() and not overwrite:
            if skip_if_exists:
                return filename  # nothing to do ...
            raise FileExistsError(errno.EEXIST, "video file already exists",
                                  filename)

        self.stream.download(output_path=filename.parent,
                             filename=filename.name)
        return filename
