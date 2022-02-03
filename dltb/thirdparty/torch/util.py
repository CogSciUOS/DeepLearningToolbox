"""Some utilities for torch.
"""

# standard imports
from typing import Union

# thirdparty imports
import numpy as np
import torch


class Debug:
    # pylint: disable=too-few-public-methods
    """An torch preprocessing image debugging class.
    This class is intended to be placed in some composed
    transformation to report the intermediate state of
    an object sent through this transformation pipeline.
    """

    def __init__(self):
        pass

    def __call__(self, image: Union[np.array, torch.Tensor]):
        print(type(image))
        print(image.dtype)
        print(torch.is_tensor(image))
        print(image.ndimension())
        return image
