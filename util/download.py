from base import BusyObservable, change

import requests


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
            raise RuntimeError("No target location for download was specified.")

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
    # FIXME[todo]: resume download: resume_header = {'Range': 'bytes=%d-' % resume_byte_pos}
    #           self._downloading = True
    # check the file size with from pathlib import Path;
    #  path = Path(..); # -> path.stat().st_size
    def download(self, force: bool=False, chunk_size: int=8192) -> None:
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

        #file_name = url.split('/')[-1]
        file_name = self._download_target
        request = requests.get(self._download_url, stream=True)

        self._download_size = request.headers.get('Content-Length')
        print("Downloading: %s Bytes: %s" % (file_name, self._download_size))

        with open(file_name, 'wb') as output_file:
            for chunk in request.iter_content(chunk_size=chunk_size):
                self._downloaded_bytes += len(chunk)
                output_file.write(chunk)
                status = r"%10d  [%3.2f%%]" % (self._downloaded_bytes,
                                               self._downloaded_bytes * 100. / self._download_size)
                status = status + chr(8)*(len(status)+1)
                print(status)

        delattr(self, '_downloaded_bytes')
        delattr(self, '_download_size')
        delattr(self, '_downloading')
