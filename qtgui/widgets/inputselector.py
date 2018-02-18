from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QFontMetrics, QIntValidator, QIcon
from PyQt5.QtWidgets import QWidget, QPushButton, QRadioButton, QLineEdit
from PyQt5.QtWidgets import QLabel, QGroupBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QSizePolicy

import observer
import numpy as np

from qtgui.datasources import DataArray, DataFile, DataDirectory

class QInputSelector(QWidget, observer.Observer):
    '''A Widget to select input data (probably images).  There are
    different modes of selection: from an array, from a file, from a
    directory or from some predefined dataset.

    Modes: there are currently different modes ('array' or 'dir').
    For each mode there exist a corresponding data source. The widget
    has a current mode and will provide data only from the
    corresponding data source.

    ATTENTION: this mode concept may be changed in future versions! It
    seems more plausible to just maintain a list of data sources.

    Attributes
    ----------
    _index  :   int
                The index of the current data entry.
    '''
    _index: int = None

    _controller = None

    def __init__(self, parent=None):
        '''Initialization of the QNetworkView.

        Parameters
        ---------
        parent  :   QWidget
                    The parent argument is sent to the QWidget constructor.
        '''
        super().__init__(parent)

        self.initUI()

    def setController(self, controller):
        super().setController(controller)

    def modelChanged(self, model, info):
        source = model._sources[model._current_mode]
        if isinstance(source, DataArray):
            info = (source.getFile()
                    if isinstance(source, DataFile)
                    else source.getDescription())
            if info is None:
                info = ''
            if len(info) > 40:
                info = info[0:info.find('/', 10) + 1] + \
                    '...' + info[info.rfind('/', 0, -20):]
            self._modeButton['array'].setText('Array: ' + info)
        elif isinstance(source, DataDirectory):
            self._modeButton['dir'].setText('Directory: ' +
                                            source.getDirectory())
        ############################################################################################
        #                              Disable buttons, if necessary                               #
        ############################################################################################
        valid = len(model) > 0
        for elem in {self.firstButton,
                     self.prevButton,
                     self.nextButton,
                     self.lastButton,
                     self.randomButton,
                     self._indexField}:
            elem.setEnabled(valid)

        n_elems = len(model)
        self.infoLabel.setText('of ' + str(n_elems - 1) if valid else '*')
        if valid:
            self._indexField.setValidator(QIntValidator(0, n_elems))
            self._indexField.setText(str(model._current_index))

        mode = model._current_mode
        self._modeButton[mode].setChecked(True)

    def _newNavigationButton(self, label: str, icon: str=None):
        button = QPushButton()
        icon = QIcon.fromTheme(icon, QIcon())
        if icon.isNull():
            button.setText(label)
        else:
            button.setIcon(icon)
        button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        button.clicked.connect(self._navigationButtonClicked)
        return button

    def initUI(self):
        '''Initialize the user interface.'''

        self.firstButton = self._newNavigationButton('|<', 'go-first')
        self.prevButton = self._newNavigationButton('<<', 'go-previous')
        self.nextButton = self._newNavigationButton('>>', 'go-next')
        self.lastButton = self._newNavigationButton('>|', 'go-last')
        self.randomButton = self._newNavigationButton('random')

        self._indexField = QLineEdit()
        self._indexField.setMaxLength(8)
        self._indexField.setAlignment(Qt.AlignRight)
        self._indexField.setValidator(QIntValidator())
        self._indexField.textChanged.connect(self._editIndex)
        self._indexField.textEdited.connect(self._editIndex)
        self._indexField.setSizePolicy(
            QSizePolicy.Maximum, QSizePolicy.Maximum)
        self._indexField.setMinimumWidth(
            QFontMetrics(self.font()).width('8') * 8)

        self.infoLabel = QLabel()
        self.infoLabel.setMinimumWidth(
            QFontMetrics(self.font()).width('8') * 8)
        self.infoLabel.setSizePolicy(
            QSizePolicy.Maximum, QSizePolicy.Expanding)

        self._modeButton = {
            'array': QRadioButton('Array'),
            'dir': QRadioButton('Directory')
        }
        self._modeButton['array'].clicked.connect(
            lambda: self._controller.mode_changed('array'))
        self._modeButton['dir'].clicked.connect(lambda: self._controller.mode_changed('dir'))

        self._openButton = QPushButton('Open...')
        self._openButton.clicked.connect(self._openButtonClicked)
        self._openButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        sourceBox = QGroupBox('Data sources')
        modeLayout = QVBoxLayout()
        modeLayout.addWidget(self._modeButton['array'])
        modeLayout.addWidget(self._modeButton['dir'])
        sourceLayout = QHBoxLayout()
        sourceLayout.addLayout(modeLayout)
        sourceLayout.addWidget(self._openButton)
        sourceBox.setLayout(sourceLayout)

        navigationBox = QGroupBox('Navigation')
        navigationLayout = QHBoxLayout()
        navigationLayout.addWidget(self.firstButton)
        navigationLayout.addWidget(self.prevButton)
        navigationLayout.addWidget(self._indexField)
        navigationLayout.addWidget(self.infoLabel)
        navigationLayout.addWidget(self.nextButton)
        navigationLayout.addWidget(self.lastButton)
        navigationLayout.addWidget(self.randomButton)
        navigationBox.setLayout(navigationLayout)

        layout = QHBoxLayout()
        layout.addWidget(sourceBox)
        layout.addWidget(navigationBox)
        self.setLayout(layout)

    def _editIndex(self, text):
        '''Event handler for the edit field.'''
        self._controller.editIndex(text)

    def _navigationButtonClicked(self):
        '''Callback for clicking the 'next' and 'prev' sample button.'''
        if self.sender() == self.firstButton:
            self._controller.rewind()
        elif self.sender() == self.prevButton:
            self._controller.rewind_one()
        elif self.sender() == self.nextButton:
            self._controller.advance_one()
        elif self.sender() == self.lastButton:
            self._controller.advance()
        elif self.sender() == self.randomButton:
            self._controller.random()

    def _openButtonClicked(self):
        '''An event handler for the 'Open' button. We need to pass this widget as parent in case a
        file must be selected.'''
        self._controller.on_open_button_clicked(self)


from PyQt5.QtWidgets import QWidget, QPushButton, QLabel
from PyQt5.QtWidgets import QHBoxLayout, QSizePolicy


class QInputInfoBox(QWidget):

    def __init__(self, parent=None):
        '''Create a new QInputInfoBox.

        parent  :   QWidget
                    Parent widget
        '''
        super().__init__(parent)
        self._initUI()
        self.showInfo()

    def _initUI(self):
        '''Initialise the UI'''
        self._metaLabel = QLabel()
        self._dataLabel = QLabel()
        self._button = QPushButton('Statistics')
        self._button.setCheckable(True)
        self._button.toggled.connect(self.update)
        self._button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        layout = QHBoxLayout()
        layout.addWidget(self._metaLabel)
        layout.addWidget(self._dataLabel)
        layout.addWidget(self._button)
        self.setLayout(layout)

    def showInfo(self, data: np.ndarray=None, description: str=None):
        '''Show info for the given (image) data.

        Parameters
        ----------
        data:
            the actual data
        description:
            some string describing the origin of the data
        '''
        self._meta_text = '<b>Input image:</b><br>\n'
        self._meta_text += f'Description: {description}\n'

        self._data_text = ''
        if data is not None:
            self._data_text += f'Input shape: {data.shape}, dtype={data.dtype}<br>\n'
            self._data_text += 'min = {}, max={}, mean={:5.2f}, std={:5.2f}\n'.format(
                data.min(), data.max(), data.mean(), data.std())
        self.update()

    def update(self):
        self._metaLabel.setText(self._meta_text)
        self._dataLabel.setText(
            self._data_text if self._button.isChecked() else '')
