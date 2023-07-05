"""Some old code which may either be properly integrated into the Deep
Learning Toolbox or deleted.

"""

# standard imports
from typing import Iterable

# thirdparty imports
import numpy as np
import skimage


# there is also torch.transforms.Resize - so what is the point of
# providing this class?
class Resize:
    # pylint: disable=too-few-public-methods
    """An image resizing operator.
    """

    def __init__(self, size):
        assert isinstance(size, int) or \
            (isinstance(size, Iterable) and len(size) == 2)
        if isinstance(size, int):
            self._size = (size, size)
        else:
            self._size = size

    def __call__(self, img: np.ndarray):
        print(img.dtype)
        resize_image = skimage.transform.resize(img, self._size)
        # , preserve_range=True
        print(resize_image.dtype)
        # the resize will return a float64 array
        # return skimage.util.img_as_ubyte(resize_image)
        return resize_image.astype(np.float32)
