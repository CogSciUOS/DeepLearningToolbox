from os import listdir, stat
from os.path import isfile, isdir, join, basename
from random import randint

import numpy as np
from scipy.misc import imread

from PyQt5.QtWidgets import QWidget, QFileDialog

# FIXME[todo]: add docstrings!


class DataSource:
    """An abstract base class for different types of data sources.  The
    individual elements of a data source can be accessed using an
    array like notation.
    """

    """A shot description of the dataset.
    """
    _description: np.ndarray = None

    def __init__(self, description=None):
        self._description = description

    def __getitem__(self, index: int):
        """Provide access to the records in this data source."""
        pass

    def __len__(self):
        """Get the number of entries in this data source."""
        pass

    def getDescription(self) -> str:
        return self._description


class DataArray(DataSource):
    """A DataSource the stores all entries in an array (like the MNIST
    character data). That means, that all entries will have the same
    sizes.
    """

    """An array of input data. Can be None.
    """
    _array: np.ndarray = None

    def __init__(self, array: np.ndarray = None, description: str = None):
        super().__init__(description)
        if array is not None:
            self.setArray(array, description)

    def setArray(self, array, description="array"):
        self._array = array
        self._description = description

    def __getitem__(self, index: int):
        if self._array is None or index is None:
            return None, None
        data = self._array[index]
        info = "Image " + str(index) + " from " + self._description
        return data, info

    def __len__(self):
        if self._array is None:
            return 0
        return len(self._array)


class DataFile(DataArray):

    """The name of the file from which the data are read.
    """
    _filename: str = None

    def __init__(self, filename: str = None):
        super().__init__()
        if filename is not None:
            self.setFile(filename)

    def setFile(self, filename: str):
        """Set the data file to be used.
        """
        self._filename = filename

        data = np.load(filename, mmap_mode='r')
        data = np.load(filename)
        self.setArray(data, basename(self._filename))

    def getFile(self) -> str:
        return self._filename

    def selectFile(self, parent: QWidget = None):
        filters = "Numpy Array (*.npy);; All Files (*)"
        filename, filter = \
            QFileDialog.getOpenFileName(parent,
                                        "Select input data archive",
                                        self._filename, filters)
        if filename is None or not isfile(filename):
            raise FileNotFoundError()
        self.setFile(filename)


class DataSet(DataArray):

    def __init__(self, name: str = None):
        super().__init__()
        self.load(name)

    def load(self, name: str):
        if name == 'mnist':
            from keras.datasets import mnist
            data = mnist.load_data()[0][0]
            self.setArray(data, "MNIST")
        else:
            raise ValueError("Unknown dataset: {}".format(name))

    def getName(self) -> str:
        return "MNIST"


class DataDirectory(DataSource):
    """A data directory contains data entries (e.g., images), in
    individual files. Each files is only read when accessed.
    """

    """A directory containing input data files. Can be None.
    """
    _dirname: str = None

    """A list of filenames in the dataDir. Will be None if no
    directory was selected. An empty list indicates that no
    suitable files where found in the directory.
    """
    _filenames: list = None

    def __init__(self, dirname: str = None):
        super().__init__()
        self.setDirectory(dirname)

    def setDirectory(self, dirname: int):
        self._dirname = dirname
        if self._dirname is None:
            self._filenames = None
        else:
            self._filenames = [f for f in listdir(self._dirname)
                               if isfile(join(self._dirname, f))]

    def getDirectory(self) -> str:
        return self._dirname

    def __getitem__(self, index):
        if self._filenames is None:
            return None, None
        filename = self._filenames[index]
        data = imread(join(self._dirname, filename))
        return data, filename

    def __len__(self):
        if self._filenames is None:
            return 0
        return len(self._filenames)

    def selectDirectory(self, parent: QWidget = None):
        dirname = QFileDialog.getExistingDirectory(parent, "Select Directory")
        if dirname is None or not isdir(dirname):
            raise FileNotFoundError()
        self.setDirectory(dirname)


from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QFontMetrics, QIntValidator, QIcon
from PyQt5.QtWidgets import QWidget, QPushButton, QRadioButton, QLineEdit
from PyQt5.QtWidgets import QLabel, QGroupBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QSizePolicy


class QInputSelector(QWidget):
    """A Widget to select input data (probably images).  There are
    different modes of selection: from an array, from a file, from a
    directory or from some predefined dataset.


    Modes: there are currently different modes ('array' or 'dir').
    For each mode there exist a corresponding data source. The widget
    has a current mode and will provide data only from the
    corresponding data source.

    ATTENTION: this mode concept may be changed in future versions! It
    seems more plausible to just maintain a list of data sources.
    """

    """The current mode: can be 'array' or 'dir'
    """
    _mode: str = None

    """The available data sources. A dictionary with mode names as keys
    and DataSource objects as values.
    """
    _sources: dict = {}

    """The index of the current data entry.
    """
    _index: int = None

    """A signal emitted when new input data are selected.
    The signal will carry the new data and some text explaining
    the data origin. (np.ndarray, str)
    """
    selected = pyqtSignal(object, str)

    def __init__(self, number: int = None, parent=None):
        """Initialization of the QNetworkView.

        Arguments
        ---------
        parent : QWidget
        parent : QWidget
            The parent argument is sent to the QWidget constructor.
        """
        super().__init__(parent)

        self.initUI()

    def _newNavigationButton(self, label: str, icon: str = None):
        button = QPushButton()
        icon = QIcon.fromTheme(icon, QIcon())
        if icon.isNull():
            button.setText(label)
        else:
            button.setIcon(icon)
        button.clicked.connect(self._navigationButtonClicked)
        button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        return button

    def initUI(self):
        """Initialize the user interface.
        """

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
        self._indexField.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self._indexField.setMinimumWidth(QFontMetrics(self.font()).width("8")*8)

        self.infoLabel = QLabel()
        self.infoLabel.setMinimumWidth(QFontMetrics(self.font()).width("8")*8)
        self.infoLabel.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Expanding)

        self._modeButton = {}
        self._modeButton['array'] = QRadioButton("Array")
        self._modeButton['array'].clicked.connect(
            lambda: self._setMode('array'))

        self._modeButton['dir'] = QRadioButton("Directory")
        self._modeButton['dir'].clicked.connect(lambda: self._setMode('dir'))

        self._openButton = QPushButton('Open...')
        self._openButton.clicked.connect(self._openButtonClicked)
        self._openButton.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum)

        sourceBox = QGroupBox("Data sources")
        modeLayout = QVBoxLayout()
        modeLayout.addWidget(self._modeButton['array'])
        modeLayout.addWidget(self._modeButton['dir'])
        sourceLayout = QHBoxLayout()
        sourceLayout.addLayout(modeLayout)
        sourceLayout.addWidget(self._openButton)
        sourceBox.setLayout(sourceLayout)

        navigationBox = QGroupBox("Navigation")
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
        """Event handler for the edit field.
        """
        try:
            index = int(text)
            if index < 0:
                raise ValueError
        except ValueError:
            index = self._index
            self._indexField.setText(str(index))

        if index != self._index:
            self.setIndex(index)

    def _navigationButtonClicked(self):
        """Callback for clicking the "next" and "prev" sample button.
        """
        if self._index is None:
            index = None
        elif self.sender() == self.firstButton:
            index = 0
        elif self.sender() == self.prevButton:
            index = self._index - 1
        elif self.sender() == self.nextButton:
            index = self._index + 1
        elif self.sender() == self.lastButton:
            index = len(self._sources[self._mode])
        elif self.sender() == self.randomButton:
            index = randint(0, len(self._sources[self._mode]))
        else:
            index = None
        self.setIndex(index)

    def _openButtonClicked(self):
        """An event handler for the 'Open' button.
        """
        try:
            source = self._sources.get(self._mode)
            if self._mode == 'array':
                if not isinstance(source, DataFile):
                    source = DataFile()
                source.selectFile(self)
            elif self._mode == 'dir':
                if not isinstance(source, DataDirectory):
                    source = DataDirectory()
                source.selectDirectory(self)
            self._setSource(source)
        except FileNotFoundError:
            pass

    def _setMode(self, mode: str):
        """Set the current mode.

        Arguments
        ---------
        mode: the mode (currently either 'array' or 'dir').
        """
        if self._mode != mode:
            self._mode = mode

            source = self._sources.get(mode)
            elements = 0 if source is None else len(source)
            valid = (elements > 1)

            self.firstButton.setEnabled(valid)
            self.prevButton.setEnabled(valid)
            self.nextButton.setEnabled(valid)
            self.lastButton.setEnabled(valid)
            self.randomButton.setEnabled(valid)
            self._indexField.setEnabled(valid)
            self.infoLabel.setText("of " + str(elements-1) if valid else "*")
            if valid:
                self._indexField.setValidator(QIntValidator(0, elements))

            if mode is not None:
                self._modeButton[mode].setChecked(True)

            self._index = None
            self.setIndex(0 if valid else None)

    def _setSource(self, source: DataSource):
        if source is None:
            return
        if isinstance(source, DataArray):
            mode = 'array'
            info = (source.getFile()
                    if isinstance(source, DataFile)
                    else source.getDescription())
            if info is None:
                info = ""
            if len(info) > 40:
                info = info[0:info.find(
                    '/', 10)+1] + '...' + info[info.rfind('/', 0, -20):]
            self._modeButton['array'].setText("Array: " + info)
        elif isinstance(source, DataDirectory):
            mode = 'dir'
            self._modeButton['dir'].setText("Directory: " +
                                            source.getDirectory())
        else:
            return

        self._sources[mode] = source
        self._mode = None
        self._setMode(mode)

    def setDataArray(self, data: np.ndarray = None):
        """Set the data array to be used.

        Arguments
        ---------
        data:
            An array of data. The first axis is used to select the
            data record, the other axes belong to the actual data.
        """
        self._setSource(DataArray(data))

    def setDataFile(self, filename: str):
        """Set the data file to be used.
        """
        self._setSource(DataFile(filename))

    def setDataDirectory(self, dirname: str = None):
        """Set the directory to be used for loading data.
        """
        self._setSource(DataDirectory(dirname))

    def setDataSet(self, name: str):
        """Set a data set to be used.

        Arguments
        ---------
        name:
            The name of the dataset. The only dataset supported up to now
            is "mnist".
        """
        self._setSource(DataSet(name))

    def setIndex(self, index=None):
        """Set the index of the entry in the current data source.

        The method will emit the "selected" signal, if a new(!) entry
        was selected.
        """

        source = self._sources.get(self._mode)
        if index is None or source is None or len(source) < 1:
            index = None
        elif index < 0:
            index = 0
        elif index >= len(source):
            index = len(source) - 1

        if self._index != index:

            self._index = index
            if source is None or index is None:
                data, info = None, None
            else:
                data, info = source[index]

            # FIXME[bug]: there is an error in PyQt forbidding to emit None
            # signals.
            if data is None:
                data = np.ndarray(())
            if info is None:
                info = ""
            self.selected.emit(data, info)

        self._indexField.setText("" if index is None else str(index))


from PyQt5.QtWidgets import QWidget, QPushButton, QLabel
from PyQt5.QtWidgets import QHBoxLayout, QSizePolicy


class QInputInfoBox(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._initUI()
        self.showInfo()

    def _initUI(self):
        self._metaLabel = QLabel()
        self._dataLabel = QLabel()
        self._button = QPushButton('Statistics')
        self._button.setCheckable(True)
        self._button.toggled.connect(self.update)
        self._button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Expanding)

        layout = QHBoxLayout()
        layout.addWidget(self._metaLabel)
        layout.addWidget(self._dataLabel)
        layout.addWidget(self._button)
        self.setLayout(layout)

    def showInfo(self, data: np.ndarray = None, description: str = None):
        """Show info for the given (image) data.

        Arguments
        ---------
        data:
            the actual data
        description:
            some string describing the origin of the data
        """
        self._meta_text = "<b>Input image:</b><br>\n"
        self._meta_text += "Description: {}\n".format(description)

        self._data_text = ''
        if data is not None:
            self._data_text += "Input shape: {}, dtype={}<br>\n".format(
                data.shape, data.dtype)
            self._data_text += "min = {}, max={}, mean={:5.2f}, std={:5.2f}\n".format(
                data.min(), data.max(), data.mean(), data.std())
        self.update()

    def update(self):
        self._metaLabel.setText(self._meta_text)
        self._dataLabel.setText(
            self._data_text if self._button.isChecked() else '')
