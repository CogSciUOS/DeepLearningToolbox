"""
File: logging.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QLabel, QListWidget, QRadioButton,
                             QVBoxLayout, QHBoxLayout, QGroupBox,
                             QSplitter)

from qtgui.utils import QLogHandler
from qtgui.widgets.maximization import (QMaximizationConfig,
                                        QMaximizationControls,
                                        QMaximizationDisplay)
from .panel import Panel



class LoggingPanel(Panel):
    """A panel containing elements to log messages.

    Attributes
    ----------
    _log_handler: QLogHandler
        A widget to display log messages

    """

    def __init__(self, parent=None):
        """Initialization of the LoggingPael.

        Parameters
        ----------
        parent  :   QWidget
                    The parent argument is sent to the QWidget constructor.
        """
        super().__init__(parent)
        self._initUI()

    def _initUI(self):
        """Add the UI elements

            * The ``QLogHandler`` showing the log messages

        """
        #
        # Controls
        #
        self._log_handler = QLogHandler()
        self._total = QLabel()

        self._modules = QListWidget()

        self._radio_fatal = QRadioButton("Fatal")
        self._radio_error = QRadioButton("Error")
        self._radio_warning = QRadioButton("Warning")
        self._radio_info = QRadioButton("Info")
        self._radio_debug = QRadioButton("Debug")

        self._log_handler._message_signal.connect(self._new_message)
        self._layoutComponents()

        

    def _layoutComponents(self):
        """Layout the UI elements.

            * The ``QLogHandler`` displaying the log messages

        """

        layout = QVBoxLayout()
        layout.addWidget(self._log_handler)
        

        row = QHBoxLayout()
        text = QHBoxLayout()
        text.addWidget(QLabel("Messages: "))
        text.addWidget(self._total)
        row.addLayout(text)
        
        radio = QGroupBox("Level")
        radio_layout = QVBoxLayout()
        radio_layout.addWidget(self._radio_fatal)
        radio_layout.addWidget(self._radio_error)
        radio_layout.addWidget(self._radio_warning)
        radio_layout.addWidget(self._radio_info)
        radio_layout.addWidget(self._radio_debug)
        radio_layout.addStretch()
        radio.setLayout(radio_layout)
        row.addWidget(radio)
        row.addWidget(self._modules)

        layout.addLayout(row)

        self.setLayout(layout)

    def addLogger(self, logger):
        logger.addHandler(self._log_handler)

    def _new_message(self, message):
        self._total.setText(str(self._log_handler.total))
