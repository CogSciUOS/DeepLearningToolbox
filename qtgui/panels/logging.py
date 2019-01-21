"""
File: logging.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QListWidget,
                             QCheckBox, QRadioButton, QButtonGroup,
                             QListWidget, QListWidgetItem,
                             QVBoxLayout, QHBoxLayout, QGroupBox,
                             QSplitter, QComboBox)

from qtgui.utils import QLogHandler
from qtgui.widgets.maximization import (QMaximizationConfig,
                                        QMaximizationControls,
                                        QMaximizationDisplay)
from .panel import Panel

import logging
import util

class LoggingPanel(Panel):
    """A panel containing elements to log messages.

    Attributes
    ----------
    _log_handler: QLogHandler
        A widget to display log messages

    """
    _levels = {
        "Fatal": logging.FATAL,
        "Error": logging.ERROR,
        "Warning": logging.WARNING,
        "Info": logging.INFO,
        "Debug": logging.DEBUG
    }

    def __init__(self, parent=None):
        """Initialization of the LoggingPael.

        Parameters
        ----------
        parent  :   QWidget
                    The parent argument is sent to the QWidget constructor.
        """
        super().__init__(parent)
        self._loggingRecorder = None
        self._initUI()

    def _initUI(self):
        """Add the UI elements

            * The ``QLogHandler`` showing the log messages

        """
        #
        # Controls
        #
        self._log_handler = QLogHandler()
        self._log_handler._message_signal.connect(self._new_message)
        self._total = QLabel()
        self._updateLogViewButton = QPushButton("Update")
        self._updateLogViewButton.clicked.connect(self._onUpdateLogView)
        self._updateLogViewButton.setEnabled(self._loggingRecorder is not None)
        self._clearLogViewButton = QPushButton("Clear")
        self._clearLogViewButton.clicked.connect(self._onClearLogView)

        self._modules = QListWidget()

        self._checkProcess = QCheckBox("Process")
        self._checkProcess.clicked.connect(self._updateFormatter)
        self._checkThread = QCheckBox("Thread")
        self._checkThread.clicked.connect(self._updateFormatter)
        self._checkName = QCheckBox("Name")
        self._checkName.clicked.connect(self._updateFormatter)
        self._checkModule = QCheckBox("Module")
        self._checkModule.clicked.connect(self._updateFormatter)
        self._checkFile = QCheckBox("File")
        self._checkFile.clicked.connect(self._updateFormatter)
        self._checkLevel = QCheckBox("Level")
        self._checkLevel.clicked.connect(self._updateFormatter)

        self._radio = {}
        self._levelButtonGroup = QButtonGroup()
        for label in self._levels.keys():
            self._radio[label] = QRadioButton(label)
            self._radio[label].clicked.connect(self._onLoggerLevelClicked)
            self._levelButtonGroup.addButton(self._radio[label])
        self._checkLoggerEnabled = QCheckBox("enabled")
        self._checkLoggerEnabled.clicked.connect(self._onLoggerEnabledClicked)

        self._buttonLoggerClearLevel = QPushButton("Clear Level")
        self._buttonLoggerClearLevel.clicked.connect(self._onClearLevel)

        self._effectiveLevel = QLabel()


        self._loggerList = QListWidget()
        self._loggerList.setSortingEnabled(True)
        self._loggerList.currentItemChanged.connect(self._onCurrentLoggerChanged)

        self._loggerList_refresh = QPushButton("Refresh")
        self._loggerList_refresh.clicked.connect(self._updateLoggerList)

        self._rootLoggerLevel = QComboBox()
        for name, level in self._levels.items():
            self._rootLoggerLevel.addItem(name, level)
        self._rootLoggerLevel.currentIndexChanged.connect(self._onRootLevelChanged)

        self._updateLoggerList()
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
        row.addWidget(self._updateLogViewButton)
        row.addWidget(self._clearLogViewButton)
        row.addWidget(self._checkProcess)
        row.addWidget(self._checkThread)
        row.addWidget(self._checkName)
        row.addWidget(self._checkModule)
        row.addWidget(self._checkFile)
        row.addWidget(self._checkLevel)
        row.addStretch()
        layout.addLayout(row)

        row = QHBoxLayout()
        column = QVBoxLayout()
        column.addWidget(self._loggerList)
        column.addWidget(self._loggerList_refresh)
        row.addLayout(column)
        
        column = QVBoxLayout()

        box = QGroupBox("Root Logger")
        boxLayout = QVBoxLayout()
        boxLayout.addWidget(self._rootLoggerLevel)
        box.setLayout(boxLayout)
        column.addWidget(box)

        box = QGroupBox("Logger Details")
        boxLayout = QVBoxLayout()
        boxLayout.addWidget(self._checkLoggerEnabled)
        line = QHBoxLayout()
        line.addWidget(QLabel("Effective Level: "))
        line.addWidget(self._effectiveLevel)
        boxLayout.addLayout(line)
        for button in self._radio.values():
            boxLayout.addWidget(button)
        boxLayout.addWidget(self._buttonLoggerClearLevel)
        box.setLayout(boxLayout)
        column.addWidget(box)

        column.addStretch()
        row.addLayout(column)

        row.addWidget(self._modules)

        layout.addLayout(row)
        
        self.setLayout(layout)

    def addLogger(self, logger):
        """Add a logger to this :py:class:LoggingPanel.
        LogRecords emitted by that logger will be processed.
        """
        logger.addHandler(self._log_handler)
        if self._loggingRecorder is not None:
            logger.addHandler(self._loggingRecorder)

    def removeLogger(self, logger):
        """Remove a logger from this :py:class:LoggingPanel.
        LogRecords emitted by that logger will no longer be processed.
        """
        logger.removeHandler(self._log_handler)
        if self._loggingRecorder is not None:
            logger.removeHandler(self._loggingRecorder)

    def setLoggingRecorder(self, recorder: util.RecorderHandler) -> None:
        """Set a logging recorder for this :py:class:LoggingPanel.
        Having a logging recorder allows to replay the log messages
        recorded by that recorder.
        """
        self._loggingRecorder = recorder
        self._onUpdateLogView()
        self._updateLogViewButton.setEnabled(recorder is not None)

    def _new_message(self, message):
        total = str(len(self._log_handler))
        if self._loggingRecorder is not None:
            total += "/" + str(len(self._loggingRecorder))
        self._total.setText(total)

    def _updateFormatter(self):
        format = ""
        if self._checkProcess.isChecked():
            format += "[%(processName)s] "
        if self._checkThread.isChecked():
            format += "[%(threadName)s] "
        if self._checkName.isChecked():
            format += "(%(name)s) "
        if self._checkModule.isChecked():
            format += "%(module)s "
        if self._checkFile.isChecked():
            format += "%(filename)s:%(lineno)d: "
        if self._checkLevel.isChecked():
            format += "%(levelname)s: "
        format += "%(message)s"
        formatter = logging.Formatter(fmt=format, datefmt="%(asctime)s")
        self._log_handler.setFormatter(formatter)

    def _onClearLogView(self):
        """Update the log view.
        """
        self._log_handler.clear()

    def _onUpdateLogView(self):
        """Update the log view.
        """
        if self._loggingRecorder is not None:
            self._loggingRecorder.replay(self._log_handler)


    def _decorateLoggerItem(self, item: QListWidgetItem,
                            logger: logging.Logger) -> None:
        """Decorate an entry in the logger list reflecting the properties
        of the logger.
        """
        item.setForeground(self._colorForLogLevel(logger.getEffectiveLevel()))
        font = item.font()
        font.setBold(bool(logger.level))
        item.setFont(font)
        item.setBackground(Qt.lightGray if logger.disabled else Qt.white)

    def _updateLoggerList(self):
        self._loggerList.clear()
        self._updateLogger(None)
        for name, logger in logging.Logger.manager.loggerDict.items():
            if not isinstance(logger, logging.Logger):
                continue
            level = logger.getEffectiveLevel()
            item = QListWidgetItem(name)
            self._decorateLoggerItem(item, logger)
            self._loggerList.addItem(item)

        index = self._rootLoggerLevel.findData(logging.Logger.root.level)
        self._rootLoggerLevel.setCurrentIndex(index)

    def _onCurrentLoggerChanged(self, item: QListWidgetItem,
                                previous: QListWidgetItem) -> None:
        """A logger was selected in the logger list.
        """
        logger = (None if item is None else
                  logging.Logger.manager.loggerDict[item.text()])
        self._updateLogger(logger)

    def _onRootLevelChanged(self, index: int) -> None:
        logging.Logger.root.setLevel(self._rootLoggerLevel.currentData())
        self._updateLoggerList()

    def _updateLogger(self, logger: logging.Logger):
        """Update the logger group to reflect the currently selected
        logger. If ther is no current logger (logger is None), then
        the logger group is cleared and disabled.
        """
        if logger is None or not logger.level:
            checked = self._levelButtonGroup.checkedButton()
            if checked is not None:
                self._levelButtonGroup.setExclusive(False)
                checked.setChecked(False)
                self._levelButtonGroup.setExclusive(True)

        self._checkLoggerEnabled.setCheckable(logger is not None)
        for button in self._levelButtonGroup.buttons():
            button.setCheckable(logger is not None)

        if logger is None:
            self._effectiveLevel.setText("")
        else:
            self._checkLoggerEnabled.setChecked(not logger.disabled)
            self._effectiveLevel.setText(str(logger.getEffectiveLevel()))
            if logger.level:
                button = self._buttonForForLogLevel(logger.level)
                if button is not None:
                    button.setChecked(True)

    def _onLoggerEnabledClicked(self, checked: bool) -> None:
        """A logger enable/disable button was pressed.
        """
        for item in self._loggerList.selectedItems():
            logger = logging.Logger.manager.loggerDict[item.text()]
            logger.disabled = not checked
            self._decorateLoggerItem(item, logger)

    def _onLoggerLevelClicked(self, checked: bool) -> None:
        """A logger level radio button was pressed.
        """
        checked = self._levelButtonGroup.checkedButton()
        level = 0 if checked is None else self._levels[checked.text()]
        for item in self._loggerList.selectedItems():
            logger = logging.Logger.manager.loggerDict[item.text()]
            logger.setLevel(level)
            self._decorateLoggerItem(item, logger)

    def _onClearLevel(self) -> None:
        """Clear the individual log level of the current logger.
        """
        logger = None
        for item in self._loggerList.selectedItems():
            logger = logging.Logger.manager.loggerDict[item.text()]
            logger.setLevel(0)
            self._decorateLoggerItem(item, logger)
        self._updateLogger(logger)

    def _buttonForForLogLevel(self, level):
        for label, _level in self._levels.items():
            if level == _level:
                return self._radio[label]
        return None       

    def _colorForLogLevel(self, level):
        if level <= logging.DEBUG: return Qt.blue
        if level <= logging.INFO: return Qt.green
        if level <= logging.WARNING: return Qt.darkYellow
        if level <= logging.ERROR: return Qt.red
        if level <= logging.FATAL: return Qt.magenta
        return Qt.black
