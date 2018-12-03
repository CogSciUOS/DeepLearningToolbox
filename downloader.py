#!/usr/bin/env python3
'''Script to download models and datasets.

FIXME[hack]: Just an ad hoc solution (quick & dirty), to be integrated
into the GUI later on, to allow for interactive loading of models and
data

.. moduleauthor:: Ulf Krumnack

'''

import sys
import os
import os.path
import ctypes
import platform
import argparse
import urllib.request
import urllib.parse

import datasources

datasets = datasources.datasources
models = ['alexnet', 'resnet']


# https://stackoverflow.com/questions/51658/cross-platform-space-remaining-on-volume-using-python
def get_free_space_mb(dirname):
    """Return folder/drive free space (in megabytes)."""
    if platform.system() == 'Windows':
        free_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(dirname),
                                                   None, None,
                                                   ctypes.pointer(free_bytes))
        return free_bytes.value / 1024 / 1024
    else:
        st = os.statvfs(dirname)
        return st.f_bavail * st.f_frsize / 1024 / 1024


def download_to_file(url, file):
    if os.path.isfile(file):
        print(f"file {file} already exists. skip downloading ...",
              file=sys.stderr)
        return

    # Silent version (no progress report ...)

    # print(f"downloading {url} as {file} ...",
    #      file=sys.stderr, end='', flush=True)
    # urllib.request.urlretrieve(url,file)
    # print(" done",
    #      file=sys.stderr)

    with open(file, 'wb') as f:
        with urllib.request.urlopen(url) as u:
            file_size = int(u.getheader('Content-Length'))
            print(f"Downloading: file Bytes: file_size",
                  file=sys.stderr)

            file_size_dl = 0
            block_sz = 8192
            while True:
                buffer = u.read(block_sz)
                if not buffer:
                    break

                file_size_dl += len(buffer)
                f.write(buffer)
                status = r"%10d  [%3.2f%%]" % (file_size_dl,
                                               file_size_dl * 100. / file_size)
                status = status + chr(8)*(len(status)+1)
                print(status, file=sys.stderr)


def main():
    '''Command line interface to the download frunctions.'''

    parser = argparse.ArgumentParser(
        description='Download models and datasets.')
    parser.add_argument('--model', help='name of a model',
                        choices=models)
    parser.add_argument('--dataset', help='name of a dataset',
                        choices=datasets)
    parser.add_argument('--directory', help='directory containing models/data')
    parser.add_argument('--framework', help='The framework to use.',
                        choices=['keras-tensorflow', 'keras-theano', 'torch'])
    args = parser.parse_args()

    if args.framework:
        print("error: framework selection is currently not supported. sorry!",
              file=sys.stderr)
        sys.exit(1)

    if args.directory:
        print("error: directory selection is currently not supported. sorry!",
              file=sys.stderr)
        sys.exit(1)

    if args.model and args.dataset:
        print("error: cannot simultanously download model and dataset. sorry!",
              file=sys.stderr)
        sys.exit(1)
    if not args.model and not args.dataset:
        print("error: you either have to specify a model or a dataset"
              "to download!", file=sys.stderr)
        sys.exit(1)

    if args.dataset == 'mnist':
        print("Downloading mnist")
        urls = [
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
        dir = os.environ['HOME']
        print("info: free space on drive: {}MB".format(get_free_space_mb(dir)),
              file=sys.stderr)
        for url in urls:
            a = urllib.parse.urlparse(url)
            file = os.path.join(dir, os.path.basename(a.path))
            download_to_file(url, file)
    else:
        print("error: operation not supported yet. sorry!",
              file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
