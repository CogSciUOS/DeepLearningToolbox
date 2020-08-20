# standard imports
import sys
import importlib

# third party imports
from PyQt5.QtWidgets import QApplication
import numpy as np

# toolbox imports
from qtgui.widgets.image import QImageView
from ..base.image import ImageDisplay


class ImageDisplay(ImageDisplay):
    """An image display that uses Qt Widgets to display an image.
    """

    def __init__(self, view: QImageView = None,
                 loop: bool = False, **kwargs) -> None:
        """
        """
        super().__init__(**kwargs)
        if loop:
            print("Creating QApplication")
            self._application = QApplication(sys.argv)  # FIXME[hack]
        self._view = view or QImageView()
        self._loop = loop
        if loop:
            self._view.show()

    def show(self, image: np.ndarray, **kwargs) -> None:
        self._view.setImage(image)
        if not self._loop:
            self._view.repaint()
            self._view.show()
