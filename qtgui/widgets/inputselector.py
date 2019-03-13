
from datasources import (Datasource, DataArray, DataFile, DataDirectory,
                         DataWebcam, DataVideo, Predefined,
                         Controller as DatasourceController)
from toolbox import Toolbox, ToolboxController
from qtgui.utils import QObserver

from PyQt5.QtWidgets import (QWidget, QPushButton, QRadioButton, QGroupBox,
                             QHBoxLayout, QVBoxLayout, QSizePolicy,
                             QInputDialog, QComboBox,
                             QFileDialog, QListView, QAbstractItemView,
                             QTreeView)

class QInputSourceSelector(QWidget, QObserver, Toolbox.Observer,
                           Datasource.Observer):
    """The QInputSourceSelector provides a controls to select a data
    source. It is mainly a graphical user interface to the datasource
    module, adding some additional logic.

    The QInputSourceSelector provides four main ways to select input
    data:
    1. a collection of predefined data sets from which the user can
       select via a dropdown box
    2. a file or directory, that the user can select via a file browser
    3. a camera
    4. a URL (not implemented yet)
    """

    _toolbox: ToolboxController=None

    # FIXME[old]:
    _controller: DatasourceController=None
    
    def __init__(self, toolbox: ToolboxController=None, parent=None):
        """Initialization of the :py:class:`QInputSourceSelector`.

        Parameters
        ---------
        parent: QWidget
            The parent argument is sent to the QWidget constructor.
        """
        super().__init__(parent)
        self._initUI()
        self._layoutUI()
        self.setToolboxController(toolbox)

    def _initUI(self):
        '''Initialize the user interface.'''

        #
        # Differnt types of Datasources
        #
        self._radioButtons = {
            'Name': QRadioButton('Predefined'),
            'Filesystem': QRadioButton('Filesystem'),
            'Webcam': QRadioButton('Webcam'),
            'Video': QRadioButton('Video')
        }
        self._radioButtons['Video'].setEnabled(False)
        for b in self._radioButtons.values():
            b.clicked.connect(self._radioButtonChecked)

        #
        # A list of predefined datasources
        #
        dataset_names = Predefined.get_data_source_ids()
        self._datasetDropdown = QComboBox()
        self._datasetDropdown.addItems(dataset_names)
        self._datasetDropdown.currentIndexChanged.\
            connect(self._predefinedSelectionChange)
        self._datasetDropdown.setEnabled(False)

        #
        # A button to select the Datasource
        #
        self._openButton = QPushButton('Open')
        self._openButton.clicked.connect(self._openButtonClicked)

    def _layoutUI(self):

        size_policy = self._datasetDropdown.sizePolicy()
        size_policy.setRetainSizeWhenHidden(True)
        self._datasetDropdown.setSizePolicy(size_policy)

        self._openButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        radioLayout = QVBoxLayout()
        for b in self._radioButtons.values():
            radioLayout.addWidget(b)

        buttonsLayout = QVBoxLayout()
        buttonsLayout.addWidget(self._datasetDropdown)
        buttonsLayout.addWidget(self._openButton)

        layout = QHBoxLayout()
        layout.addLayout(radioLayout)
        layout.addLayout(buttonsLayout)
        self.setLayout(layout)

    def datasource_changed(self, controller, info):
        '''The QInputSourceSelector is only affected by changes of
        the Datasource.
        '''
        if info.datasource_changed:
            self._setDatasource(controller.get_datasource())

    def _setDatasource(self, datasource: Datasource):
        if isinstance(datasource, Predefined):
            self._radioButtons['Name'].setChecked(True)
            id = datasource.get_public_id()
            index = self._datasetDropdown.findText(id)
            if index == -1:
                pass # should not happen!
            elif index != self._datasetDropdown.currentIndex():
                self._datasetDropdown.setCurrentIndex(index)
        elif isinstance(datasource, DataWebcam):
            self._radioButtons['Webcam'].setChecked(True)
        elif isinstance(datasource, DataVideo):
            self._radioButtons['Video'].setChecked(True)
        elif isinstance(datasource, DataFile):
            self._radioButtons['Filesystem'].setChecked(True)
        elif isinstance(datasource, DataDirectory):
            self._radioButtons['Filesystem'].setChecked(True)
        else:
            self._radioButtons['Filesystem'].setChecked(True)

        self._datasetDropdown.setEnabled(self._radioButtons['Name'].isChecked())
        # FIXME[old]
        # if isinstance(datasource, DataArray):
        #     info = (datasource.getFile()
        #             if isinstance(datasource, DataFile)
        #             else datasource.get_description())
        #     if info is None:
        #         info = ''
        #     if len(info) > 40:
        #         info = info[0:info.find('/', 10) + 1] + \
        #             '...' + info[info.rfind('/', 0, -20):]
        #     self._radioButtons['Name'].setText('Name: ' + info)
        # elif isinstance(datasource, DataDirectory):
        #     self._radioButtons['Filesystem'].setText('File: ' +
        #                                     datasource.getDirectory())
        #####################################################################
        #                Disable buttons, if necessary                      #
        #####################################################################


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
        """An event handler for the ``Open`` button. Pressing this
        button will select a datasource. How exactly this works
        depends on the type of the Datasource, which is selected by
        the radio buttons.
        """

        datasource = None
        if self._radioButtons['Name'].isChecked():
            # Name: this will select a predefined Datasource based on its
            # name which is selectend in the _datasetDropdown QComboBox.

            #self._datasetDropdown.setVisible(True)
            name = self._datasetDropdown.currentText()
            print(f"!!!{name}!!!")
            datasource = Predefined.get_data_source(name)

        elif self._radioButtons['Filesystem'].isChecked():
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
                datasource = DataDirectory(fname)
            else:
                datasource = DataFile(fname)
        elif self._radioButtons['Webcam'].isChecked():
            datasource = DataWebcam()
        elif self._radioButtons['Video'].isChecked():
            # FIXME[hack]: use file browser ...
            # FIXME[problem]: the opencv embedded in anoconda does not
            # have ffmpeg support, and hence cannot read videos
            datasource = DataVideo("/net/home/student/k/krumnack/AnacondaCON.avi")

        print("We have selected the following Datasource:")
        if datasource is None:
            print("  -> no Datasource")
        else:
            print(f"  type: {type(datasource)}")
            print(f"  prepared:  {datasource.prepared}")

        try:
            datasource.prepare()
            print(f"  len:  {len(datasource)}")
            print(f"  description:  {datasource.datasource.get_description()}")
        except Exception as ex:
            print(f"  preparation failed ({ex})!")
            datasource = None

        # FIXME[hack]: not really implemented. what should happen is:
        # - change the datasource for the DatasourceView/Controller
        #    -> this should notify the observer 'observable_changed'
        # what may happen in response is
        # - add datasource to some list (e.g. toolbox)
        # - emit some pyqt signal?
        #if getattr(self, '_controller', None) is not None: # FIXME[old]
        #    self._controller.onSourceSelected(datasource)
        if datasource is not None:
            if self._toolbox:
                # Set the datasource of the Toolbox.
                # This will also insert the dataset in the Toolbox's list
                # if datasources, if it is not already in there.
                self._toolbox.datasource_controller(datasource)

    def _predefinedSelectionChange(self,i):
        if self._radioButtons['Name'].isChecked():
            self._datasetDropdown.setVisible(True)
            name = self._datasetDropdown.currentText()
            datasource = Predefined.get_data_source(name)

            if getattr(self, '_controller', None) is not None: # FIXME[old]
                self._controller.onSourceSelected(datasource)
            if self._toolbox is not None:
                self._toolbox.set_datasource(datasource)

    def setToolboxController(self, toolbox: ToolboxController) -> None:
        self._exchangeView('_toolbox', toolbox,
                           interests=Toolbox.Change('datasources_changed'))

    def toolbox_changed(self, toolbox: Toolbox, change):
        self._setDatasource(self._toolbox.datasource)


from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontMetrics, QIntValidator, QIcon
from PyQt5.QtWidgets import QHBoxLayout, QSizePolicy
from PyQt5.QtWidgets import QWidget, QPushButton, QLineEdit, QLabel


class QInputNavigator(QWidget, QObserver, Datasource.Observer):
    """
    A QInputNavigator displays widgets to navigate in the Datasource.
    The actual set of widgets depends on the type of Datasource and
    will be adapated when the Datasource is changed.
    """
    _controller: DatasourceController = None

    def __init__(self, datasource: DatasourceController=None, parent=None):
        '''Initialization of the QInputNavigator.

        Parameters
        ---------
        parent: QWidget
            The parent argument is sent to the QWidget constructor.
        datasource: DatasourceController
            A Controller allowing to navigate in the Datasource.
        '''
        super().__init__(parent)
        self._initUI()
        self._layoutUI()
        self.setDatasourceController(datasource)

    def _initUI(self):
        """Initialize the user interface.

        """

        #
        # Navigation in indexed data source
        #
        self.firstButton = self._newNavigationButton('|<', 'go-first')
        self.prevButton = self._newNavigationButton('<<', 'go-previous')
        self.nextButton = self._newNavigationButton('>>', 'go-next')
        self.lastButton = self._newNavigationButton('>|', 'go-last')
        self.randomButton = self._newNavigationButton('random')
        self.prepareButton = self._newNavigationButton('prepare')
        def slot(checked: bool) -> None:
            if checked:
                self._controller.prepare()
            else:
                self._controller.unprepare()
        self.prepareButton.setCheckable(True)
        self.prepareButton.toggled.connect(slot)

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
        self._layout = None

        self._buttonList = [
            self.firstButton, self.prevButton,
            self._indexField, self.infoLabel,
            self.nextButton, self.lastButton #, self.randomButton
        ]

        
    def _layoutUI(self):
        if self._layout is None:
            self._layout = QVBoxLayout()
            self._layout.addWidget(self.infoDataset)
            buttons2 = QHBoxLayout()
            buttons2.addWidget(self.prepareButton)
            buttons2.addStretch()
            buttons2.addWidget(self.randomButton)
            self._layout.addLayout(buttons2)
            self._buttons = None
            self.setLayout(self._layout)

        if self._controller is None or not self._controller:
            # no datasource controller or no datasource: remove buttons
            if self._buttons is not None:
                for button in self._buttonList:
                    button.setParent(None)
                self._layout.removeItem(self._buttons)
                self._buttons = None
        else:
            # we have a datasource: add the buttons
            if self._buttons is None:
                self._buttons = QHBoxLayout()
                for button in self._buttonList:
                    self._buttons.addWidget(button)
                self._layout.addLayout(self._buttons)

    def _enableUI(self):
        enabled = bool(self._controller) and self._controller.prepared
        for button in self._buttonList:
            button.setEnabled(enabled)
        self.prepareButton.setEnabled(bool(self._controller))
        if self._controller:
            self.prepareButton.setChecked(self._controller.prepared)

    def datasource_changed(self, datasource: Datasource, info) -> None:
        n_elems = 0 if datasource is None else len(datasource)
        valid = n_elems > 0

        if info.observable_changed or info.state_changed:
            self._layoutUI()
            self._enableUI()
            if datasource:
                text = datasource.get_description()
                if not datasource.prepared:
                    text += " (unprepared)"
            else:
                text = "No datasource"
            self.infoDataset.setText(text)

            self.infoLabel.setText('of ' + str(n_elems - 1) if valid else '*')
            if valid:
                self._indexField.setValidator(QIntValidator(0, n_elems))

        if info.index_changed:
            if self._controller:
                index = self._controller.get_index() or 0
            else:
                index = 0
            if self._controller:
                self._indexField.setText(str(index))
            enabled = bool(self._controller) and self._controller.prepared
            self.firstButton.setEnabled(enabled and index > 0)
            self.prevButton.setEnabled(enabled and index > 0)
            self.nextButton.setEnabled(enabled and index+1 < n_elems)
            self.lastButton.setEnabled(enabled and index+1 < n_elems)
            
            
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


    # FIXME[design]: can't this be directly connected to the controller?
    def _editIndex(self, text):
        '''Event handler for the edit field.'''
        self._controller.edit_index(text)

    def _navigationButtonClicked(self):
        '''Callback for clicking the 'next' and 'prev' sample button.'''
        if self._controller is None:
            return
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

    def setDatasourceController(self,
                                datasource: DatasourceController) -> None:
        self._exchangeView('_controller', datasource,
                           interests=Datasource.Change('observer_changed',
                                                       'index_changed',
                                                       'state_changed'))




from datasources import Datasource, Controller as DatasourceController
from model import Model, ModelObserver

from PyQt5.QtWidgets import QWidget, QGroupBox, QHBoxLayout

from qtgui.utils import QObserver
from toolbox import Toolbox, ToolboxController

class QInputSelector(QWidget, QObserver, Datasource.Observer):
    """A Widget to select input data (probably images).

    This Widget consists of two subwidgets:
    1. a :py:class:`QInputNavigator` to select a :py:class:`DataSource` and
    2. a :py:class:`QInputNavigator` to navigate in the
    :py:class:`DataSource`.

    There are different modes of selection: from an array, from a file,
    from a directory or from some predefined data source.
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
    _source_selector: QInputSourceSelector
        A widget to change the currently selected datasource.
    _navigator: QInputNavigator
        A widget to navigate in the currently selected datasource.
    _index: int
        The index of the current data entry.
    """
    _source_selector: QInputSourceSelector = None
    _navigator: QInputNavigator = None
    _index: int = None # FIXME[old]: seems not to be used

    def __init__(self, toolbox: ToolboxController=None, parent=None):
        '''Initialization of the QInputSelector.

        Parameters
        ----------
        parent  :   QWidget
                    The parent argument is sent to the QWidget constructor.
        '''
        super().__init__(parent)
        self._initUI()
        self.setToolboxController(toolbox)

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

    def setController(self, controller: DatasourceController) -> None:
        """Set the controller for this QInputSelector. Will trigger
        observation of the controller.

        Parameters
        ----------
        controller: DatasourceController
            Controller for mediating commands for the activations panel.

        """
        super().setController(controller)
        self._source_selector.setController(controller)
        self._navigator.setController(controller)

    def setToolboxController(self, toolbox: ToolboxController):
        self._source_selector.setToolboxController(toolbox)
        self.setDatasourceController(toolbox.datasource_controller
                                     if toolbox else None)

    def setDatasourceController(self,
                                datasource: DatasourceController) -> None:
        """Set the controller for this QInputSelector. Will trigger
        observation of the controller.

        Arguments
        ---------
        controller: DatasourceController
            Controller for mediating commands for the activations panel.

        """
        self.setController(datasource)
        self._navigator.setDatasourceController(datasource)
        

    def datasource_changed(self, datasource, change):
        pass  # FIXME[hack]: for some reason, this gets registered as a Datasource.Observer, but what changes are we interestend in?



from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel

import numpy as np

from toolbox import Toolbox, View as ToolboxView
from datasources import Datasource, View as DatasourceView


class QInputInfoBox(QWidget, QObserver, Datasource.Observer, Toolbox.Observer, ModelObserver):
    """A :py:class:`QInputInfoBox` displays information on the currently
    selected input image.

    Data can be provided in different ways:
    (1) From a :py:class:`Toolbox`, using the input data
    (2) From a :py:class:`Datasource`, using the current data
    """
    _toolbox: ToolboxView = None
    _datasource: DatasourceView = None
    _processed: bool = False

    def __init__(self, toolbox: ToolboxView=None,
                 datasource: DatasourceView=None, parent=None):
        '''Create a new QInputInfoBox.

        parent  :   QWidget
                    Parent widget
        '''
        super().__init__(parent)
        self._initUI()
        self._model = None
        self._showInfo()
        self.setToolboxView(toolbox)
        self.setDatasourceView(datasource)

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

    def setToolboxView(self, toolbox: ToolboxView) -> None:
        self._exchangeView('_toolbox', toolbox,
                           interests=Toolbox.Change('input_changed'))

    def setDatasourceView(self, datasource: Datasource) -> None:
        self._exchangeView('_datasource', datasource,
                           interests=Datasource.Change('data_changed'))

    def setController(self, controller) -> None:
        # FIXME[hack]: we need a more reliable way to observe multiple observable!
        self.observe(controller.get_observable(), interests=None)

    def toolbox_changed(self, toolbox: Toolbox,
                        change: Toolbox.Change) -> None:
        if change.input_changed:
            if not self._toolbox:
                data = None
            elif self._processed:
                data = self._toolbox.input_data
            else:
                data = self._toolbox.input_data
            label = self._toolbox.input_label
            description = self._toolbox.input_description
            self._showInfo(data=data, description=description)

    def datasource_changed(self, datasource, change):
        if change.data_changed:
            if self._datasource:
                data, label = datasource.data
                self._showInfo(data=data, label=label)
            else:
                self._showInfo()

    def modelChanged(self, model, info):
        self._model = model
        if info.input_changed:
            description = model.input_data_description
            self._showInfo(description=description)

    @pyqtSlot(bool)
    def onModeChanged(self, processed: bool):
        """The display mode was changed.

        Arguments
        ---------
        processed: bool
            The new display mode (False=raw, True=processed).
        """
        self.setMode(processed)

    def setMode(self, processed):
        if processed != self._processed:
            self._processed = processed
            self._showInfo()
        
    def _showInfo(self, data: np.ndarray=None, label=None,
                  description: str=''):
        '''Show info for the given (image) data.
        '''       
        self._meta_text = '<b>Input image:</b><br>\n'
        self._meta_text += f'Description: {description}<br>\n'
        if label is not None:
            self._meta_text += f'Label: {label}<br>\n'

        self._data_text = ('<b>Preprocessed input:</b><br>\n'
                           if self._processed else
                           '<b>Raw input:</b><br>\n')
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

