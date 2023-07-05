"""Interface to support downloading files.
"""

# FIXME[hack]: get rid of third-party imports
# FIXME[hack]: allow to pass progress bar argument
# FIXME[hack]: rename 'skip_if_exists' argument by 'overwrite' (with opposite semantics)
# FIXME[hack]: do logging and error processing

# standard imports
from typing import Optional, Union
from io import BytesIO
from pathlib import Path
import logging

# third-party imports
import requests

# conda install -c conda-forge tqdm
# For Jupyter notebooks: from tqdm.notebook import tqdm
from tqdm import tqdm

# toolbox imports
from ..types import Pathlike, as_path
from ..base.busy import BusyObservable
from ..base.implementation import Implementable

# logging
LOG = logging.getLogger(__name__)


# Helper function to download (windows compatibility)
def download_request(url: str, filename: Pathlike, skip_if_exists=True):
    """
    """
    filename = as_path(filename)
    if skip_if_exists and filename.is_file():
        LOG.info("Download skipped - file '%s' already exists.", filename)
        return

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()

    if total_size_in_bytes not in (0, progress_bar.n):
        LOG.error("Something went wrong while downloading '%s' to '%s'. "
                  "Please manually delete any residual download files",
                  url, filename)


download = download_request

# -- end: download --


# FIXME[old]: there is also dltb.util.resource.Downloadable
# (both classes seem not to be used yet/anymore?)
class Downloadable(BusyObservable):
    """An interface to be implemented by resources that are
    downloadable.


    Attributes
    ----------

    _download_url: str
        The URL from where the resource can be downloaded.

    _download_target: str
        The target location on the local system where the resource
        is to be stored.

    _download_bytes: int
        (Only during download): the number of bytes already downloaded.

    _download_size: int
        (Only during download): the size of the currently downloaded
        file in bytes.
    """

    _download_url: str = None
    _download_target: str = None

    _downloaded_bytes: int = None
    _download_size: int = None

    @property
    def download_url(self) -> str:
        """The URL from where we the resource can be downloaded.
        """
        return self._download_url

    @property
    def download_target(self):
        """The target (on the local system) where the resource should be
        stored when downloaded.
        """
        return self._download_target

    @property
    def downloaded(self) -> bool:
        """Check if the resource was downloaded.
        """
        if not self._download_target:
            raise RuntimeError("No target location for was specified "
                               "for storing the downloaded file.")

    @property
    def downloading(self) -> bool:
        """Check if the resource is currently downloaded.
        """
        return self._downloaded_bytes is not None

    @property
    def download_progress(self) -> (int, int):
        """Check the current progress in downloading.
        """
        return (self._downloaded_bytes, self._download_size)

    # FIXME[todo]: may be run in the background ...
    # FIXME[todo]: may notify observers ...
    # FIXME[todo]: resume download:
    #              resume_header = {'Range': 'bytes=%d-' % resume_byte_pos}
    #           self._downloading = True
    # check the file size with from pathlib import Path;
    #  path = Path(..); # -> path.stat().st_size
    def download(self, force: bool = False, chunk_size: int = 8192) -> None:
        """Download the resource and store it at the target location.

        Arguments
        ---------
        force: bool
            Force download even if the resource is already available.

        Raises
        ------
        """
        if self.downloaded and not force:
            return

        if not self._download_url:
            raise RuntimeError("No download URL was specified.")

        if not self._download_url:
            raise RuntimeError("No download URL was specified.")

        self._downloaded_bytes = 0

        # file_name = url.split('/')[-1]
        file_name = self._download_target
        request = requests.get(self._download_url, stream=True)

        self._download_size = request.headers.get('Content-Length')
        print(f"Downloading: {file_name} Bytes: {self._download_size}")

        with open(file_name, 'wb') as output_file:
            for chunk in request.iter_content(chunk_size=chunk_size):
                self._downloaded_bytes += len(chunk)
                output_file.write(chunk)
                status = "{self._downloaded_bytes:10d}  " \
                    "[{self._downloaded_bytes*100./self._download_size:3.2f}%]"
                status = status + chr(8)*(len(status)+1)
                print(status)

        delattr(self, '_downloaded_bytes')
        delattr(self, '_download_size')
        delattr(self, '_downloading')



URL = str
URLlike = Union[URL, str]


class YoutubeDownloader(Implementable):
    """A downloader for the Youtube platform. The downlader supports
    downloading video files als well as metadata.

    Example
    -------
    from dltb.util.download import YoutubeDownloader
    path = YoutubeDownloader.download_to_file('dQw4w9WgXcQ')

    url:
        Youtube URL for the video.
    
    """

    @staticmethod
    def download_to_file(url: URLlike,
                         filename: Optional[Pathlike] = None,
                         overwrite: bool = False,
                         skip_if_exists: bool = False) -> Path:
        """Download a given Youtube video to a file.
        """
        downloader = YoutubeDownloader(url=url)
        return downloader.to_file(filename, overwrite=overwrite,
                                  skip_if_exists=skip_if_exists)

    def __init__(self, url: Optional[URLlike] = None,
                 youtube: Optional[str] = None) -> None:
        if (url is None) == (youtube is None):
            raise ValueError("Provide either URL or youtube ID to "
                             "initialized a YoutubeDownloader.")

        if url is not None:
            youtube = url.split('/')[-1]
            if '=' in youtube:
                youtube = youtube.split('=')[-1]

        self.url = f'https://youtu.be/{youtube}'

    def to_file(self, filename: Optional[Pathlike] = None,
                overwrite: bool = False, skip_if_exists: bool = False) -> Path:
        """Download the video into a file.
        """

    def to_buffer(self, buffer: Optional[BytesIO] = None) -> BytesIO:
        """Download the video into a buffer.
        """
