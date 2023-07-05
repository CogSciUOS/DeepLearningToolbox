"""Image related Matplotlib code.

Examples:

>>> from dltb.thirdparty.matplotlib.image import MplImageDisplay
>>> display = MplImageDisplay(image="examples/reservoir-dogs.jpg")
>>> display.show()

"""

# standard imports
from typing import Optional

# third party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backend_bases import KeyEvent, MouseEvent

# toolbox imports
from dltb.base.image import Image, Imagelike
from dltb.base.image import ImageReader, ImageWriter, ImageDisplay, ImageView
from . import MplSimpleDisplay, MplPlotter, LOG


class ImageIO(ImageReader, ImageWriter):
    """Implementation of the image I/O functionality based on
    Matplotlib.

    Matplotlib Image I/O is not the recommended for Image I/O as
    there is now dedicated library (`imageio`), supporting more
    formats.  If, however, matplotlib is used anyhow,  it may be
    a quick alternative to loading additional libraries.
    """

    def read(self, filename: str, **kwargs) -> np.ndarray:
        """Read image from the given `filename`.

        Arguments
        ---------
        filename:
            Matplotlib accepts as filename: a filename, a URL, or a
            file-like object opened in read-binary mode.

        Note: Matplotlib seems to read images as np.float64 with
        a value range from 0.0 to 255.0.

        Note: Matplotlib can only read PNGs natively. Further image
        formats are supported via the optional dependency on
        Pillow. Note, URL strings are not compatible with Pillow.
        """
        return plt.imread(filename)  # .astype(np.uint8)

    def write(self, filename: str, image: Imagelike, **kwargs) -> None:
        """Write an `image` to the given `filename`.
        """
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
        plt.imsave(filename, Image.as_array(image, dtype=np.uint8))


class MplImagePlotter(ImageView, MplPlotter):
    """A plotter that can display an image on a matplotlib axes.
    """
    _mpl_image = None

    def _init_axes(self) -> None:
        super()._init_axes()
        image_array = (np.zeros((100, 100, 3), dtype=np.uint8)
                       if self.image is None else self.image.array)
        self._mpl_image = self._axes.imshow(image_array)

    def _release_axes(self) -> None:
        self._mpl_image = None
        super()._release_axes()

    def _set_image(self, image: Optional[Image]) -> None:
        """Set a new :py:class:`Image` for this `MplImagePlotter`.
        The new image is guaranteed to be different from the current
        image.
        """
        super()._set_image(image)
        axes, mpl_image = self.axes, self._mpl_image
        if axes is None or mpl_image is None:
            return  # cannot display image without an Axes object

        if image is not None:
            image_array = image.array
            axes.set_xlim(0, image_array.shape[1])
            axes.set_ylim(image_array.shape[0], 0)
            mpl_image.set_extent((-0.5, image.shape[1]-.5,
                                  image.shape[0]-.5, -0.5))
            mpl_image.set_data(image_array)
        else:
            mpl_image.set_data(None)

    def on_key_pressed(self, event: KeyEvent) -> None:
        """Implementation of a Matplotlib key event handler.
        """
        # event properties:
        #   'canvas'
        #   'guiEvent'
        #   'inaxes'
        #   key: str
        #       'a','b', ... 'up', 'down', 'left', 'right',
        #       'delete', 'backspace', ...
        #   'lastevent'
        #   'name'
        #   'x'
        #   'xdata'
        #   'y'
        #   'ydata'
        LOG.info("Detected key press event in matplotlib figure.")
        
    def on_button_pressed(self, event: MouseEvent) -> None:
        """Implementation of a Matplotlib mouse press event handler.
        """
        # event Properties:
        #   'button'
        #   'canvas'
        #   'dblclick'
        #   'guiEvent'
        #   'inaxes'
        #   'key'
        #   'lastevent'
        #   'name'
        #   'step'
        #   'x'
        #   'xdata'
        #   'y'
        #   'ydata'
        x, y = event.x, event.y
        LOG.info("Detected mouse press event in matplotlib figure.")


class MplImageDisplay(ImageDisplay, MplSimpleDisplay):
    """Matplotlib implementation of an :py:class:`ImageDisplay`.
    """
    _key_press_id = None
    _button_press_event = None

    def __init__(self, plotter: Optional[MplImagePlotter] = None,
                 **kwargs) -> None:
        if plotter is None:
            plotter = MplImagePlotter()
        super().__init__(plotter=plotter, **kwargs)

    def _set_figure(self, figure: Figure) -> None:
        if self._figure is not None:
            self._figure.canvas.mpl_disconnect(self._key_press_id)
            self._figure.canvas.mpl_disconnect(self._button_press_id)
        super()._set_figure(figure)
        if self._figure is not None:
            self._key_press_id = \
                self._figure.canvas.mpl_connect('key_press_event',
                                                self.view.on_key_pressed)
            self._button_press_id = \
                self._figure.canvas.mpl_connect('button_press_event',
                                                self.view.on_button_pressed)
