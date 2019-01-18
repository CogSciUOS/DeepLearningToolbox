from controller import DataSourceController, DataSourceObserver
from model import Model, ModelObserver

from PyQt5.QtWidgets import QWidget, QGroupBox, QHBoxLayout

class QInputSelector(QWidget, DataSourceObserver):
    '''A Widget to select input data (probably images).  There are
    different modes of selection: from an array, from a file, from a
    directory or from some predefined data source.

    Modes: there are currently different modes ('array' or 'dir').
    For each mode there exist a corresponding data source. The widget
    has a current mode and will provide data only from the
    corresponding data source.

    FIXME[attention]
    ATTENTION: this mode concept may be changed in future versions! It
    seems more plausible to just maintain a list of data sources.

    .. warning:: This docstring must be changed once the mode concept
    is thrown overboard

    Attributes
    ----------
    _index  :   int
                The index of the current data entry.

    '''
    _index: int = None

    def __init__(self, parent=None):
        '''Initialization of the QInputSelector.

        Parameters
        ---------
        parent  :   QWidget
                    The parent argument is sent to the QWidget constructor.
        '''
        super().__init__(parent)
        self._initUI()


    def setController(self, controller: DataSourceController) -> None:
        """Set the controller for this QInputSelector. Will trigger
        observation of the controller.

        Parameters
        ----------
        controller: DataSourceController
            Controller for mediating commands for the activations panel.

        """
        super().setController(controller)
        self._source_selector.setController(controller)
        self._navigator.setController(controller)


    def _initUI(self):
        self._source_selector = QInputSourceSelector()
        self._navigator = QInputNavigator()

        sourceBox = QGroupBox('Data sources')
        sourceBox.setLayout(self._source_selector.layout())

        navigationBox = QGroupBox('Navigation')
        navigationBox.setLayout(self._navigator.layout())

        layout = QHBoxLayout()
        layout.addWidget(sourceBox)
        layout.addWidget(navigationBox)
        self.setLayout(layout)




from datasources import (DataArray, DataFile, DataDirectory, DataWebcam,
                         DataVideo, Predefined)

from PyQt5.QtWidgets import (QWidget, QPushButton, QRadioButton, QGroupBox,
                             QHBoxLayout, QVBoxLayout, QSizePolicy,
                             QInputDialog, QComboBox,
                             QFileDialog, QListView, QAbstractItemView,
                             QTreeView)


class QInputSourceSelector(QWidget, DataSourceObserver):
    '''The QInputSourceSelector provides a controls to select a data
    source. It is mainly a graphical user interface to the datasource
    module, adding some additional logic.

    The QInputSourceSelector provides four main ways to select input
    data:
    1. a collection of predefined data sets from which the user can
       select via a dropdown box
    2. a file or directory, that the user can select via a file browser
    3. a camera
    4. a URL (not implemented yet)
    '''
    
    def __init__(self, parent=None):
        '''Initialization of the QInputNavigator.

        Parameters
        ---------
        parent  :   QWidget
                    The parent argument is sent to the QWidget constructor.
        '''
        super().__init__(parent)
        self._initUI()

    def datasource_changed(self, controller, info):
        '''The QInputSourceSelector is only affected by changes of
        the DataSource.
        '''

        if info.datasource_changed:
            source = controller.get_datasource()

            if isinstance(source, Predefined):
                self._radioButtons['Name'].setChecked(True)
                id = source.get_public_id()
                index = self._datasetDropdown.findText(id)
                if index == -1:
                    pass # should not happen!
                elif index != self._datasetDropdown.currentIndex():
                    self._datasetDropdown.setCurrentIndex(index)
            elif isinstance(source, DataWebcam):
                self._radioButtons['Webcam'].setChecked(True)
            elif isinstance(source, DataVideo):
                self._radioButtons['Video'].setChecked(True)
            elif isinstance(source, DataFile):
                self._radioButtons['Filesystem'].setChecked(True)
            elif isinstance(source, DataDirectory):
                self._radioButtons['Filesystem'].setChecked(True)
            else:
                self._radioButtons['Filesystem'].setChecked(True)

            self._datasetDropdown.setEnabled(self._radioButtons['Name'].isChecked())
        # FIXME[old]
        # if isinstance(source, DataArray):
        #     info = (source.getFile()
        #             if isinstance(source, DataFile)
        #             else source.getDescription())
        #     if info is None:
        #         info = ''
        #     if len(info) > 40:
        #         info = info[0:info.find('/', 10) + 1] + \
        #             '...' + info[info.rfind('/', 0, -20):]
        #     self._radioButtons['Name'].setText('Name: ' + info)
        # elif isinstance(source, DataDirectory):
        #     self._radioButtons['Filesystem'].setText('File: ' +
        #                                     source.getDirectory())
        #####################################################################
        #                Disable buttons, if necessary                      #
        #####################################################################


    def _initUI(self):
        '''Initialize the user interface.'''

        # Data sources
        self._radioButtons = {
            'Name': QRadioButton('Predefined'),
            'Filesystem': QRadioButton('Filesystem'),
            'Webcam': QRadioButton('Webcam'),
            'Video': QRadioButton('Video')
        }
        self._radioButtons['Video'].setEnabled(False)
        radioLayout = QVBoxLayout()
        for b in self._radioButtons.values():
            b.clicked.connect(self._radioButtonChecked)
            radioLayout.addWidget(b)

        self._openButton = QPushButton('Open')
        self._openButton.clicked.connect(self._openButtonClicked)
        self._openButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self._datasetDropdown = QComboBox()
        size_policy = self._datasetDropdown.sizePolicy()
        size_policy.setRetainSizeWhenHidden(True)
        self._datasetDropdown.setSizePolicy(size_policy)
        dataset_names = Predefined.get_data_source_ids()
        self._datasetDropdown.addItems(dataset_names)
        self._datasetDropdown.setEnabled(False)
        self._datasetDropdown.currentIndexChanged.connect(self._predefinedSelectionChange)
        
        buttonsLayout = QVBoxLayout()
        buttonsLayout.addWidget(self._datasetDropdown)
        buttonsLayout.addWidget(self._openButton)

        sourceLayout = QHBoxLayout()
        sourceLayout.addLayout(radioLayout)
        sourceLayout.addLayout(buttonsLayout)
        self.setLayout(sourceLayout)

    def _radioButtonChecked(self):
        '''Callback for clicking the radio buttons.'''
        name = self.sender().text()
        if name == 'Name':
            self._datasetDropdown.setEnabled(True)
            self._openButton.setText('Load')
        elif name == 'Filesystem':
            self._datasetDropdown.setEnabled(False)
            self._openButton.setText('Open')
        elif name == 'Webcam':
            self._datasetDropdown.setEnabled(False)
            self._openButton.setText('Run')
        elif name == 'Video':
            self._datasetDropdown.setEnabled(False)
            self._openButton.setText('Play')
        self._datasetDropdown.setEnabled(self._radioButtons['Name'].isChecked())

    def _openButtonClicked(self):
        '''An event handler for the ``Open`` button.'''
        mode = None
        if self._radioButtons['Name'].isChecked():
            self._datasetDropdown.setVisible(True)
            name = self._datasetDropdown.currentText()
            print(f"!!!{name}!!!")
            data_source = Predefined.get_data_source(name)
        elif self._radioButtons['Filesystem'].isChecked():
            mode = 'Filesystem'
            # CAUTION: I've converted the C++ from here
            # http://www.qtcentre.org/threads/43841-QFileDialog-to-select-files-AND-folders
            # to Python. I'm pretty sure this makes use of
            # implemention details of the QFileDialog and is thus
            # susceptible to sudden breakage on version change. It's
            # retarded that there is no way for the file dialog to
            # accept either files or directories at the same time so
            # this is necessary.
            #
            # UPDATE: It appears setting the selection mode is
            # unnecessary if only single selection is desired. The key
            # insight appears to be using the non-native file dialog
            dialog = QFileDialog(self)
            dialog.setFileMode(QFileDialog.Directory)
            dialog.setOption(QFileDialog.DontUseNativeDialog, True)
            nMode = dialog.exec_()
            fname = dialog.selectedFiles()[0]
            import os
            if os.path.isdir(fname):
                data_source = DataDirectory(fname)
            else:
                data_source = DataFile(fname)
        elif self._radioButtons['Webcam'].isChecked():
            mode = 'Webcam'
            data_source = DataWebcam()
        elif self._radioButtons['Video'].isChecked():
            mode = 'Video'
            # FIXME[hack]: use file browser ...
            # FIXME[problem]: the opencv embedded in anoconda does not
            # have ffmpeg support, and hence cannot read videos
            data_source = DataVideo("/net/home/student/k/krumnack/AnacondaCON.avi")
        self._controller.onSourceSelected(data_source)

    def _predefinedSelectionChange(self,i):
        if self._radioButtons['Name'].isChecked():
            self._datasetDropdown.setVisible(True)
            name = self._datasetDropdown.currentText()
            data_source = Predefined.get_data_source(name)
            self._controller.onSourceSelected(data_source)




from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontMetrics, QIntValidator, QIcon
from PyQt5.QtWidgets import QHBoxLayout, QSizePolicy
from PyQt5.QtWidgets import QWidget, QPushButton, QLineEdit, QLabel


class QInputNavigator(QWidget, DataSourceObserver):

    def __init__(self, parent=None):
        '''Initialization of the QInputNavigator.

        Parameters
        ---------
        parent  :   QWidget
                    The parent argument is sent to the QWidget constructor.
        '''
        super().__init__(parent)
        self._initUI()


    def datasource_changed(self, controller, info) -> None:
        datasource = controller.get_datasource()
        n_elems = 0 if datasource is None else len(datasource)
        valid = n_elems > 0

        if info.datasource_changed:
            if datasource is not None:
                self.infoDataset.setText(datasource.getDescription())
            self.infoLabel.setText('of ' + str(n_elems - 1) if valid else '*')
            if valid:
                self._indexField.setValidator(QIntValidator(0, n_elems))
            # Disable buttons, if necessary
            for elem in {self.firstButton,
                         self.prevButton,
                         self.nextButton,
                         self.lastButton,
                         self.randomButton,
                         self._indexField}:
                elem.setEnabled(valid)

        if info.index_changed and valid:
            inputIndex = controller.get_index()
            self._indexField.setText(str(inputIndex))

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

    def _initUI(self):
        '''Initialize the user interface.'''

        #
        # Navigation in indexed data source
        #
        self.firstButton = self._newNavigationButton('|<', 'go-first')
        self.prevButton = self._newNavigationButton('<<', 'go-previous')
        self.nextButton = self._newNavigationButton('>>', 'go-next')
        self.lastButton = self._newNavigationButton('>|', 'go-last')
        self.randomButton = self._newNavigationButton('random')

        # _indexField: A text field to manually enter the index of
        # desired input.
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

        self.infoDataset = QLabel()
        
        navigationLayout = QHBoxLayout()
        navigationLayout.addWidget(self.firstButton)
        navigationLayout.addWidget(self.prevButton)
        navigationLayout.addWidget(self._indexField)
        navigationLayout.addWidget(self.infoLabel)
        navigationLayout.addWidget(self.nextButton)
        navigationLayout.addWidget(self.lastButton)
        navigationLayout.addWidget(self.randomButton)
        navigationMainLayout = QVBoxLayout()
        navigationMainLayout.addWidget(self.infoDataset)
        navigationMainLayout.addLayout(navigationLayout)
        self.setLayout(navigationMainLayout)
        #navigationBox.setLayout(navigationMainLayout)

    # FIXME[design]: can't this be directly connected to the controller?
    def _editIndex(self, text):
        '''Event handler for the edit field.'''
        self._controller.edit_index(text)

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


from PyQt5.QtWidgets import QWidget, QPushButton, QLabel

import numpy as np


class QInputInfoBox(QWidget, DataSourceObserver, ModelObserver):

    # FIXME[hack]: imageView: there should be no explicit reference
    # between widgets We need imageView._show_raw here. Think of some
    # suitable mechanism and then remove this hack ...
    def __init__(self, parent=None, imageView=None):
        '''Create a new QInputInfoBox.

        parent  :   QWidget
                    Parent widget
        '''
        super().__init__(parent)
        self._initUI()
        self._description = ''
        self._description2 = ''
        self._show_raw = False
        self._model = None
        self._showInfo()

    def _initUI(self):
        '''Initialise the UI'''
        self._metaLabel = QLabel()
        self._dataLabel = QLabel()
        self._button = QPushButton('Statistics')
        self._button.setCheckable(True)
        self._button.toggled.connect(self.update)
        self._button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout1 = QHBoxLayout()
        layout1.addWidget(self._metaLabel)
        layout1.addWidget(self._button)

        layout = QVBoxLayout()
        layout.addLayout(layout1)
        layout.addWidget(self._dataLabel)
        self.setLayout(layout)

    def setController(self, controller) -> None:
        # FIXME[hack]: we need a more reliable way to observe multiple observable!
        self.observe(controller.get_observable())

    def datasource_changed(self, controller, info):
        if info.index_changed:
            datasource = controller.get_datasource()
            if datasource is not None:
                index = controller.get_index()
                self._description = datasource.getName(index)
            else:
                self._description = ''

    def modelChanged(self, model, info):
        self._model = model
        if info.input_changed:
            self._description2 = model.input_data_description
            self._showInfo()

    def onModeChange(self, showRaw: bool):
        """The display mode was changed.

        Arguments
        ---------
        mode: bool
            The new display mode (False=raw, True=reshaped).
        """
        self._show_raw = showRaw
        self._showInfo()

    def _showInfo(self, data: np.ndarray=None):
        '''Show info for the given (image) data.

        Parameters
        ----------
        data:
            the actual data
        description:
            some string describing the origin of the data
        '''
        if self._model is not None:
            if self._show_raw:
                data = self._model.raw_input_data
            else:
                data = self._model.input_data
        
        self._meta_text = '<b>Input image:</b><br>\n'
        self._meta_text += f'Description 1: {self._description}<br>\n'
        self._meta_text += f'Description 2: {self._description2}<br>\n'

        self._data_text = ('<b>Raw input:</b><br>\n'
                           if self._show_raw else
                           '<b>Network input:</b><br>\n')
        if data is not None:
            self._data_text += (f'Input shape: {data.shape}, '
                                f'dtype={data.dtype}<br>\n')
            self._data_text += ('min = {}, max={}, mean={:5.2f}, '
                                'std={:5.2f}\n'.
                                format(data.min(), data.max(),
                                       data.mean(), data.std()))
        self.update()

    def update(self):
        self._metaLabel.setText(self._meta_text)
        self._dataLabel.setText(
            self._data_text if self._button.isChecked() else '')
