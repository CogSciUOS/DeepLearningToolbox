'''
File: ikkuna.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
'''

from PyQt5 import QtCore
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton)

from .panel import Panel


class IkkunaPanel(Panel):
    """A Panel to work with the Ikkuna library for live inspection during
    training.

    """

    def __init__(self, parent=None):
        """Initialization of the AdversarialExamplePanel.

        Parameters
        ----------
        parent: QWidget
            The parent argument is sent to the QWidget constructor.
        """
        super().__init__(parent)

        self._initUI()
        self._layoutUI()


    def _initUI(self):
        pass

    def _layoutUI(self):
        pass
