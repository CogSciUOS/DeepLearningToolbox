"""Functions for supporting image operations on a canvas (currently
only numpy arrays are supported).

"""

# standard imports
from typing import Optional

# thirdparty imports
import numpy as np

# toolbox imports
from ..base.image import Image, Imagelike
from ..base.image import Size, Sizelike


def canvas_create(image: Optional[Imagelike] = None, copy: bool = True,
                  size: Optional[Sizelike] = None,
                  channels: int = 3) -> np.ndarray:
    """Create a new canvas.


    Arguments
    ---------
    image:
        An image to be used for initializing the canvas.
    copy:
        If `False` the image will not be copied (if possible), meaning that
        canvas operations will be performed directly on that image.
    size:
        The size of the canvas.
    channels:
        The number of channels.
    """
    if size is not None:
        size = Size(size)

    if image is not None:
        image = Image.as_array(image, copy=copy)

    if size is None and image is None:
        raise ValueError("Neither image nor size for new canvas "
                         "was specified.")

    if size is not None and (image is None or
                             ((size.height, size.width) != image.shape[:2])):
        canvas = np.zeros((size.height, size.width, channels), dtype=np.uint8)
        if image is not None:
            canvas_add_image(canvas, image)
    else:
        canvas = image

    return canvas


def _adapt_slice(the_slice: slice, diff: int) -> slice:
    diff1 = diff // 2
    diff2 = diff - diff1
    return slice(the_slice.start + diff1, the_slice.stop - diff2,
                 the_slice.step)


def canvas_add_image(canvas: np.ndarray, image: Imagelike,
                     rows: int = 1, columns: int = 1, index: int = 1) -> None:
    """Add an image to the canvas.

    Arguments
    ---------
    canvas:
        The canvas to which to add the image.
    image:
        The image to add to the canvas.
    rows:
        Rows of the canvas grid.
    columns:
        Columns of the canvas grid.
    index:
        Index of the cell in the canvas grid, starting with 1 (like
        matplotlib subplots).
    """
    row, column = (index-1) // columns, (index-1) % columns
    height, width = canvas.shape[0] // rows, canvas.shape[1] // columns
    canvas_x = slice(column * width, (column+1) * width)
    canvas_y = slice(row * height, (row+1) * height)

    image = Image.as_array(image)
    image_x = slice(0, image.shape[1])
    image_y = slice(0, image.shape[0])

    diff = image_x.stop - width
    if diff > 0:
        image_x = _adapt_slice(image_x, diff)
    elif diff < 0:
        canvas_x = _adapt_slice(canvas_x, -diff)

    diff = image_y.stop - height
    if diff > 0:
        image_y = _adapt_slice(image_y, diff)
    elif diff < 0:
        canvas_y = _adapt_slice(canvas_y, -diff)

    canvas[canvas_y, canvas_x] = image[image_y, image_x]
