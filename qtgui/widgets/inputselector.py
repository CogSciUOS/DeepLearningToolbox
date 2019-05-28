

from datasources import (Datasource, DataArray, Random,
                         Controller as DatasourceController)

from ..utils import QObserver, protect

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
        self.prepareButton = self._newNavigationButton('prepare')
        self.prepareButton.setCheckable(True)
        self.randomButton = self._newNavigationButton('random')
        self.loopButton = self._newNavigationButton('loop')
        self.loopButton.setCheckable(True)

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
        self._stateLabel = QLabel()
        self._layout = None

        self._buttonList = [
            self.firstButton, self.prevButton,
            #self._indexField, self.infoLabel,
            self.nextButton, self.lastButton #, self.randomButton
        ]

        
    def _layoutUI(self):
        if self._layout is None:
            # We have no Layout yet: create initial Layout
            self._layout = QVBoxLayout()
            self._layout.addWidget(self.infoDataset)
            self._layout.addWidget(self._stateLabel)
            buttons2 = QHBoxLayout()
            buttons2.addWidget(self.prepareButton)
            buttons2.addStretch()
            buttons2.addWidget(self.randomButton)
            buttons2.addWidget(self.loopButton)
            buttons3 = QHBoxLayout()
            buttons3.addWidget(self._indexField)
            buttons3.addWidget(self.infoLabel)
            buttons3.addStretch()
            self._layout.addLayout(buttons2)
            self._layout.addLayout(buttons3)
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
        enabled = bool(self._controller)
        self.prepareButton.setEnabled(bool(self._controller))
        enabled = enabled and self._controller.prepared
        for button in self._buttonList + [self.randomButton, self.loopButton,
                                          self._indexField, self.infoLabel]:
            button.setEnabled(enabled)
        enabled = enabled and self._controller.isinstance(Random)
        self.randomButton.setEnabled(enabled)
        if self._controller:
            self.prepareButton.setChecked(self._controller.prepared)


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


    @protect
    def _editIndex(self, text):
        '''Event handler for the edit field.'''
        if self._controller is None:
            return
        self._controller.edit_index(text)

    @protect
    def _navigationButtonClicked(self, checked: bool):
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
        elif self.sender() == self.prepareButton:
            if checked:
                self._controller.prepare()
            else:
                self._controller.unprepare()
        elif self.sender() == self.loopButton:
            self._controller.loop()

    def setDatasourceController(self,
                                datasource: DatasourceController) -> None:
        interests = Datasource.Change('observable_changed', 'busy_changed',
                                      'state_changed')
        self._exchangeView('_controller', datasource, interests=interests)

    def datasource_changed(self, datasource: Datasource, info) -> None:
        """React to chanes in the datasource. Changes of interest are:
        (1) change of datasource ('observable_changed'): we may have to
            adapt the controls to reflect the type of Datasource.
        (2) usability of the datasource ('state_changed', 'busy_changed'):
            we may have to disable some controls depending on the state
            (prepared, busy).
        (3) change of selected image (data_changed): we may want to
            reflect the selection in our controls.
        """
        if info.observable_changed:
            self._layoutUI()

        if info.observable_changed or info.state_changed:
            self._updateDescription(datasource)

        if info.state_changed or info.busy_changed:
            self._updateState()
            self._enableUI()

        if info.data_changed:
            if self._controller and self._controller.isinstance(DataArray):
                n_elems = 0 if datasource is None else len(datasource)
                index = self._controller.index or 0
                self._indexField.setText(str(index))
                enabled = bool(self._controller) and self._controller.prepared
                self.firstButton.setEnabled(enabled and index > 0)
                self.prevButton.setEnabled(enabled and index > 0)
                self.nextButton.setEnabled(enabled and index+1 < n_elems)
                self.lastButton.setEnabled(enabled and index+1 < n_elems)


    def _updateDescription(self, datasource):
        info = ''
        if datasource:
            text = datasource.get_description()
            if self._controller and self._controller.isinstance(DataArray):
                elements = len(datasource)
                if elements:
                    self._indexField.setValidator(QIntValidator(0, elements))
                    info = 'of ' + str(elements - 1)
        else:
            text = "No datasource"
        self.infoDataset.setText(text)
        self.infoLabel.setText(info)
    
    def _updateState(self):
        if not self._controller:
            text = "none"
        elif not self._controller.prepared:
            text = "unprepared"
        elif  self._controller.busy:
            text = "busy"
        else:
            text = "ready"
        self._stateLabel.setText("State: " + text)


from datasources import Datasource, Controller as DatasourceController

from ..utils import QObserver
from ..widgets.datasource import QDatasourceSelectionBox

from PyQt5.QtWidgets import QWidget, QGroupBox, QHBoxLayout

from toolbox import Toolbox, ToolboxController

class QInputSelector(QWidget, QObserver, Datasource.Observer):
    """A Widget to select input data (probably images).

    This Widget consists of two subwidgets:
    1. a :py:class:`QDatasourceSelectionBox` to select a
       :py:class:`DataSource` and
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
    _source_selector: QDatasourceSelectionBox
        A widget to change the currently selected datasource.
    _navigator: QInputNavigator
        A widget to navigate in the currently selected datasource.
    _index: int
        The index of the current data entry.
    """
    _source_selector: QDatasourceSelectionBox = None
    _navigator: QInputNavigator = None

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
        self._source_selector = QDatasourceSelectionBox()
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



from toolbox import Toolbox, View as ToolboxView
from datasources import Datasource, View as DatasourceView
from tools.activation import Engine as ActivationEngine

import numpy as np

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout


class QInputInfoBox(QWidget, QObserver, Datasource.Observer, Toolbox.Observer,
                    ActivationEngine.Observer):
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
        self._layoutUI()
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

    def _layoutUI(self):
        self._metaLabel.setWordWrap(True)
        self._dataLabel.setWordWrap(True)

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

    def setDatasourceView(self, datasource: DatasourceView) -> None:
        self._exchangeView('_datasource', datasource,
                           interests=Datasource.Change('data_changed'))

    def datasource_changed(self, datasource, change):
        if change.data_changed:
            if self._datasource:
                data, label = datasource.data
                self._showInfo(data=data, label=label)
            else:
                self._showInfo()

    def setController(self, controller) -> None:
        # FIXME[hack]: we need a more reliable way to observe multiple observable!
        self.observe(controller.get_observable(), interests=None)

    def activation_changed(self, model, info):
        self._model = model
        if info.input_changed:
            description = model.input_data_description
            self._showInfo(description=description)

    @pyqtSlot(bool)
    @protect
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

