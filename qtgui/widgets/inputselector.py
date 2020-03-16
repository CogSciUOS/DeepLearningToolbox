

from datasource import (Datasource, Random, Indexed,
                         Controller as DatasourceController)

from ..utils import QObserver, protect
from .datasource import QLoopButton


from datasource import Datasource, Controller as DatasourceController

from ..utils import QObserver
from ..widgets.datasource import QDatasourceSelectionBox, QInputNavigator

from PyQt5.QtWidgets import QWidget, QGroupBox, QHBoxLayout
from PyQt5.QtWidgets import QHBoxLayout, QSizePolicy


from toolbox import Toolbox, ToolboxController

class QInputSelector(QWidget, QObserver, Datasource.Observer):
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
from datasource import Datasource, View as DatasourceView
from tools.activation import (Engine as ActivationEngine,
                              View as ActivationView)

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
    _activation: ActivationView = None
    _processed: bool = False

    def __init__(self, toolbox: ToolboxView=None,
                 datasource: DatasourceView=None,
                 activations: ActivationView=None, parent=None):
        '''Create a new QInputInfoBox.

        parent  :   QWidget
                    Parent widget
        '''
        super().__init__(parent)
        self._initUI()
        self._layoutUI()
        self._showInfo()
        self.setToolboxView(toolbox)
        self.setDatasourceView(datasource)
        self.setActivationView(activations)

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

    #
    # Toolbox.Observer
    #

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

    #
    # Datasource.Observer
    #

    def setDatasourceView(self, datasource: DatasourceView) -> None:
        self._exchangeView('_datasource', datasource,
                           interests=Datasource.Change('data_changed'))

    def datasource_changed(self, datasource, change):
        if change.data_changed:
            # the data provided by the Datasource has changed
            if self._datasource:
                data, label = datasource.data
                self._showInfo(data=data, label=label)
            else:
                self._showInfo()

    #
    # ActivationEngine.Observer
    #

    def setActivationView(self, toolbox: ActivationView) -> None:
        interests = ActivationEngine.Change('input_changed')
        self._exchangeView('_activation', toolbox, interests=interests)

    def activation_changed(self, engine: ActivationEngine, info):
        if info.input_changed:
            self._updateInfo()

    def _updateInfo(self):
        if self._activation is None:
            data = self._toolbox.input_data if self._toolbox else None
        elif self._processed:
            data = self._activation.input_data
        else:
            data = self._activation.raw_input_data
        label = self._toolbox.input_label
        description = self._toolbox.input_description
        self._showInfo(data=data, label=label, description=description)
        

    # FIXME[old]
    def setController(self, controller) -> None:
        # FIXME[hack]: we need a more reliable way to observe multiple observable!
        self.observe(controller.get_observable(), interests=None)


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
            self._updateInfo()
        
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

