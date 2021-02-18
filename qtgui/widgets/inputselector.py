# standard imports
from typing import Any

# Qt imports
from PyQt5.QtWidgets import QWidget, QGroupBox, QHBoxLayout
from PyQt5.QtWidgets import QSizePolicy

# toolbox imports
from toolbox import Toolbox
from datasource import Datasource

# GUI imports
from ..utils import QObserver, protect
from .datasource import QDatasourceSelectionBox, QInputNavigator


# FIXME[old]: this class seems to be no longer in use - check
# and recycle parts that may still be useful!
class QInputSelector(QWidget, QObserver, qobservables={
        # FIXME[hack]: check what we are really interested in ...
        Datasource: Datasource.Change.all()}):
    """A Widget to select input data (probably images).

    This Widget consists of two subwidgets:
    1. a :py:class:`QDatasourceSelectionBox` to select a
       :py:class:`Datasource` and
    2. a :py:class:`QInputNavigator` to navigate in the
       :py:class:`Datasource`.

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

    def __init__(self, toolbox: Toolbox=None, parent=None):
        '''Initialization of the QInputSelector.

        Parameters
        ----------
        parent  :   QWidget
                    The parent argument is sent to the QWidget constructor.
        '''
        super().__init__(parent)
        self._initUI()
        self.setToolbox(toolbox)

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




# FIXME[old]: seems not to be used anymore
class QInputNavigator(QWidget, QObserver, qobservables={
        Datasource: {'busy_changed', 'state_changed', 'data_changed'}}):
    """A :py:class:`QInputNavigator` displays widgets to navigate in the
    Datasource.  The actual set of widgets depends on the type of
    Datasource and will be adapated when the Datasource is changed.

    """
    # FIXME[rename]: rename this property once the old Controller is gone
    _q_controller: QDatasourceController = None
    #
    _infoDatasource: QLabel = None

    # display the state of the Datasource:
    # "none" / "unprepared" / "busy" / "ready"
    _stateLabel = None

    # prepare or unprepare the Datasource
    _prepareButton = None

    _navigator: QDatasourceNavigator = None

    def __init__(self, datasource: Datasource = None, **kwargs):
        """Initialization of the QInputNavigator.

        Arguments
        ---------
        datasource: Datasource
            A Controller allowing to navigate in the Datasource.
        """
        super().__init__(**kwargs)
        self._initUI()
        self._layoutUI()
        self.setDatasource(datasource)

    def _initUI(self):
        """Initialize the user interface.

        """
        self._q_controller = QDatasourceController()

        self._prepareButton = QPushButton('Prepare')
        self._prepareButton.setSizePolicy(QSizePolicy.Maximum,
                                          QSizePolicy.Maximum)
        self._prepareButton.setCheckable(True)
        self._prepareButton.clicked.connect(self._prepareButtonClicked)
        
        self._infoDatasource = QLabel()
        self._stateLabel = QLabel()
        
        self._navigator = QDatasourceNavigator()

    def _layoutUI(self):
        """The layout of the :py:class:`QInputNavigator` may change
        depending on the :py:class:`Datasource` it controls.

        """
        # FIXME[todo]: we may also add some configuration options
        
        # We have no Layout yet: create initial Layout
        layout = QVBoxLayout()
        layout.addWidget(self._q_controller)
        layout.addWidget(self._infoDatasource)
        layout.addWidget(self._stateLabel)

        # buttons 2: control the datsource
        buttons2 = QHBoxLayout()
        buttons2.addWidget(self._prepareButton)
        buttons2.addStretch()
        layout.addLayout(buttons2)

        layout.addWidget(self._navigator)

        self.setLayout(layout)
            
        #if self._controller is None or not self._controller:
        #    # no datasource controller or no datasource: remove buttons
        #    if self._buttons is not None:
        #        for button in self._buttonList:
        #            button.setParent(None)
        #        self._layout.removeItem(self._buttons)
        #        self._buttons = None
        #else:
        #    # we have a datasource: add the buttons
        #    if self._buttons is None:
        #        self._buttons = QHBoxLayout()
        #        for button in self._buttonList:
        #            self._buttons.addWidget(button)
        #        self._layout.addLayout(self._buttons)

    def _enableUI(self):
        enabled = bool(self._controller)
        self._prepareButton.setEnabled(bool(self._controller))
        if bool(self._controller):
            self._prepareButton.setChecked(self._controller.prepared)

    @protect
    def _prepareButtonClicked(self, checked: bool):
        '''Callback for clicking the 'next' and 'prev' sample button.'''
        if not self._controller:
            return
        if checked:
            self._controller.prepare()
        else:
            self._controller.unprepare()

    def setDatasource(self, datasource: Datasource) -> None:
        self._navigator.setDatasource(datasource)

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
            self._updateState()
            self._enableUI()

        if info.observable_changed or info.state_changed:
            self._updateDescription(datasource)

        if info.state_changed or info.busy_changed:
            self._updateState()
            self._enableUI()

    def _updateDescription(self, datasource):
        info = ''
        if datasource:
            text = datasource.description
        else:
            text = "No datasource"
        self._infoDatasource.setText(text)
    
    def _updateState(self):
        if not self._controller:
            text = "none"
        elif not self._controller.prepared:
            text = "unprepared"
        elif self._controller.busy:
            text = "busy"
            busy_message = self._controller.busy_message
            if busy_message:
                text += f" ({busy_message})"
        else:
            text = "ready"
        self._stateLabel.setText("State: " + text)
