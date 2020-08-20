# standard imports

# third party imports
import matplotlib.pyplot as plt
import numpy as np

# toolbox imports
from ..base.image import ImageReader, ImageWriter, ImageDisplay


class ImageIO(ImageReader, ImageWriter, ImageDisplay):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._figure = plt.figure()

    def read(self, filename: str, **kwargs) -> np.ndarray:
        return plt.imread(filename)

    def write(self, image: np.ndarray, filename: str, **kwargs) -> None:
        # vmin, vmax: scalar
        #     vmin and vmax set the color scaling for the image by
        #     fixing the values that map to the colormap color
        #     limits. If either vmin or vmax is None, that limit is
        #     determined from the arr min/max value.
        #
        # cmap: str or Colormap, optional
        #     A Colormap instance or registered colormap name. The
        #     colormap maps scalar data to colors. It is ignored for
        #     RGB(A) data. Defaults to rcParams["image.cmap"] =
        #     'viridis' ('viridis').
        #
        # format: str
        #    The file format, e.g. 'png', 'pdf', 'svg', ...
        #
        # origin: {'upper', 'lower'}
        #    Indicates whether the (0, 0) index of the array is in the
        #    upper left or lower left corner of the axes. Defaults to
        #    rcParams["image.origin"] = 'upper' ('upper').
        #
        # dpi: int
        #    The DPI to store in the metadata of the file. This does
        #    not affect the resolution of the output image.
        #
        plt.imsave(filename, image)

    def show(self, image: np.ndarray, title: str = 'Image',
             **kwargs) -> None:
        plt.title(title)
        plt.imshow(image)
        plt.show()  # FIXME[todo]: this will block until the window is closed
