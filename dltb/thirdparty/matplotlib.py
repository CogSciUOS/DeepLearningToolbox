# standard imports

# third party imports
import matplotlib.pyplot as plt
import numpy as np

# toolbox imports
from ..base.image import Image, Imagelike
from ..base.image import ImageReader, ImageWriter, ImageDisplay


class ImageIO(ImageReader, ImageWriter, ImageDisplay):

    def read(self, filename: str, **kwargs) -> np.ndarray:
        """
        """
        # Matplotlib accepts as filename: a filename, a URL, or a
        # file-like object opened in read-binary mode.
        #
        # Note: Matplotlib seems to read images as np.float64 with
        # a value range from 0.0 to 255.0.
        #
        # Note: Matplotlib can only read PNGs natively. Further image
        # formats are supported via the optional dependency on
        # Pillow. Note, URL strings are not compatible with Pillow.
        return plt.imread(filename)  # .astype(np.uint8)

    def write(self, image: Imagelike, filename: str, **kwargs) -> None:
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


class Display(ImageDisplay):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # self._figure, self._ax = plt.subplots(1, 1)
        self._figure, self._ax = None, None

    def _show(self, image: np.ndarray, title: str = 'Image',
              **kwargs) -> None:
        # self._ax.set_xlim(0, image.shape[1])
        # self._ax.set_ylim(image.shape[0], 0)
        self._imshow.set_extent((-0.5, image.shape[1]-.5,
                                 image.shape[0]-.5, -0.5))
        self._imshow.set_data(image)
        self._figure.canvas.set_window_title(title)
        self._figure.canvas.draw()
        self._figure.suptitle("MatPlotLib")

    def _open(self) -> None:
        if self._figure is not None:
            # self._figure.canvas.draw()
            self._figure.show()
            return
        if self._blocking is not True:
            plt.ion()  # enable interactive mode
        self._figure, self._ax = plt.subplots(1, 1)
        self._figure.canvas.mpl_connect('close_event', self._handle_close)
        self._imshow = self._ax.imshow(np.zeros((100, 100, 3), dtype=np.uint8))
        #if self._blocking is not True:
        #    self._figure.show(block=False)  # matplotlib 1.1

    def _close(self) -> None:
        plt.close(self._figure)

    def _process_events(self) -> None:
        self._figure.canvas.draw()

    def _run_blocking_event_loop(self, timeout: float = None) -> None:
        # show() will block until the window is closed
        self._figure.show()

    def _run_nonblocking_event_loop(self) -> None:
        # show() will block until the window is closed
        # self._figure.show()
        pass

    def _handle_close(self, evt):
        print('Closed Figure!')
        self.close()
