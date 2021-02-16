# FIXME[hack]: get rid of third-party imports
# FIXME[hack]: allow to pass progress bar argument
# FIXME[hack]: rename 'skip_if_exists' argument by 'overwrite' (with opposite semantics)
# FIXME[hack]: do logging and error processing

import os
import sys


# standard imports
# third-party imports
import requests

# conda install -c conda-forge tqdm
# For Jupyter notebooks: from tqdm.notebook import tqdm
from tqdm import tqdm


# Helper function to download (windows compatibility)
def download(url, filename, skip_if_exists=True):
    """
    """
    if skip_if_exists and os.path.isfile(filename):
        print(f"Download skipped - file '{filename}' already exists.",
              file=sys.stderr)
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

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong. "
              "Please manually delete any residual download files",
              file=sys.stderr)

# -- end: download --
