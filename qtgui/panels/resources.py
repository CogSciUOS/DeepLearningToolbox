"""
File: resources.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
"""

from toolbox import Toolbox, Controller as ToolboxController
from network import Network, Controller as NetworkController
from datasources import Datasource, Controller as DatasourceController

from .panel import Panel
from ..utils import QObserver, protect
from ..widgets import QNetworkView, QNetworkBox, QNetworkSelector
from ..widgets import QInputInfoBox, QModelImageView
from ..widgets.inputselector import QInputNavigator
from ..widgets.network import QNetworkList
from ..widgets.datasource import QDatasourceList, QDatasourceSelectionBox

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QResizeEvent
from PyQt5.QtWidgets import (QWidget, QPushButton, QGroupBox,
                             QVBoxLayout, QHBoxLayout, QGridLayout)


class ResourcesPanel(Panel, QObserver, Toolbox.Observer):
    '''A Panel for managing resources used by the Toolbox.

    _inputView: QModelImageView
        An image view area to display the currently selected input
        image from the Toolbox.
    _input_info: QInputInfoBox
        An info box displaying additional information on the currently
        selected input image.

    _datasourceList: QDatasourceList
    _datasourceSelector: QDatasourceSelectionBox
    _datasourceNavigator: QInputNavigator
    '''
    _toolboxController: ToolboxController = None
    _networkController: NetworkController = None

    _datasourceList: QDatasourceList = None
    _datasourceSelector: QDatasourceSelectionBox = None
    _datasourceNavigator: QInputNavigator = None
    _inputView: QModelImageView = None
    _input_info: QInputInfoBox = None

    def __init__(self, toolbox: ToolboxController=None,
                 network1: NetworkController=None,
                 datasource1: DatasourceController=None,
                 parent: QWidget=None):
        super().__init__(parent)
        self._initUI()
        self._layoutUI()
        self.setToolboxController(toolbox)
        self.setNetworkController(network1)
        self.setDatasourceController(datasource1)

    def _initUI(self):

        self._networkList = QNetworkList()

        @protect
        def clicked(checked: bool):
            network = self._toolboxController.hack_new_model()
        self._button = QPushButton("Add Network 1")
        self._button.clicked.connect(clicked)

        @protect
        def clicked(checked: bool):
            network = self._toolboxController.hack_new_model2()
        self._button2 = QPushButton("Add Network 2")
        self._button2.clicked.connect(clicked)

        @protect
        def clicked(checked: bool):
            network = self._toolboxController.hack_new_alexnet()
        self._button3 = QPushButton("Load Alexnet")
        self._button3.clicked.connect(clicked)

        @protect
        def currentIndexChanged(index: int):
            self._networkController(self._networkSelector.currentData())
        self._networkSelector = QNetworkSelector()
        self._networkSelector.currentIndexChanged.connect(currentIndexChanged)

        self._networkView = QNetworkView()
        self._networkBox = QNetworkBox()


        #
        # Datasources
        #

        self._datasourceList = QDatasourceList()
        self._datasourceSelector = QDatasourceSelectionBox()

        @protect
        def clicked(checked: bool):
            datasource = self._toolboxController.hack_new_model()  # FIXME[todo]!
        self._buttonAddDatasource = QPushButton("Add datasource 1")
        self._buttonAddDatasource.clicked.connect(clicked)

        #
        # Input data
        #

        # QModelImageView: a widget to display the input data
        self._inputView = QModelImageView()

        self._input_info = QInputInfoBox()
        self._inputView.modeChanged.connect(self._input_info.onModeChanged)
        
        # QInputSelector: a widget to select the input
        # (combined datasource selector and datasource navigator)
        self._datasourceNavigator = QInputNavigator()


    def _layoutUI(self):

        layout = QVBoxLayout()
        layout.addWidget(self._networkList)
        layout.addWidget(self._button)
        layout.addWidget(self._button2)
        layout.addWidget(self._button3)
        layout.addWidget(self._networkSelector)
        layout.addStretch()

        layout2 = QHBoxLayout()
        layout2.addLayout(layout)
        layout2.addWidget(self._networkView)
        layout2.addWidget(self._networkBox)
        layout2.addStretch()

        self._networkGroupBox = QGroupBox('Networks')
        self._networkGroupBox.setLayout(layout2)

        layout = QVBoxLayout()
        row = QHBoxLayout()
        column = QVBoxLayout()
        column.addWidget(self._datasourceList)
        column.addWidget(self._buttonAddDatasource)
        row.addLayout(column)
        row.addWidget(self._datasourceSelector)
        layout.addLayout(row)
        layout.addWidget(self._datasourceNavigator)

        # FIXME[layout]
        #layout.setSpacing(0)
        #layout.setContentsMargins(0, 0, 0, 0)
        self._input_info.setMinimumWidth(200)
        # keep image view square (FIXME[question]: does this make
        # sense for every input?)
        self._inputView.heightForWidth = lambda w: w
        self._inputView.hasHeightForWidth = lambda: True

        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(self._inputView)
        row.addStretch()
        layout.addLayout(row)
        layout.addWidget(self._input_info)

        layout.addStretch()

        self._datasourceGroupBox = QGroupBox('Data sources')
        self._datasourceGroupBox.setLayout(layout)

        layout = QGridLayout()
        self._networkGroupBox.sizePolicy().setHorizontalStretch(1)
        self._datasourceGroupBox.sizePolicy().setHorizontalStretch(1)
        layout.addWidget(self._networkGroupBox, 0, 0)
        layout.addWidget(self._datasourceGroupBox, 0, 1)
        self.setLayout(layout)

    def setToolboxController(self, toolbox: ToolboxController) -> None:
        self._exchangeView('_toolboxController', toolbox)
        self._networkList.setToolboxView(toolbox)
        self._networkSelector.setToolboxView(toolbox)
        self._datasourceList.setToolboxView(toolbox)
        self._datasourceSelector.setToolboxController(toolbox)
        self._inputView.setToolboxView(toolbox)
        self._input_info.setToolboxView(toolbox)

    def setNetworkController(self, network: NetworkController) -> None:
        self._networkController = network
        self._networkList.setNetworkView(network)
        self._networkSelector.setNetworkView(network)
        self._networkBox.setNetworkView(network)

    def setDatasourceController(self,
                                datasource: DatasourceController) -> None:
        self._datasourceList.setDatasourceView(datasource)
        self._datasourceNavigator.setDatasourceController(datasource)

    def setEnabled(self):
        enabled = self._toolboxController is not None
        self._button.setEnabled(enabled)

    def toolbox_changed(self, toolbox, change):
        self._toolboxController(toolbox)

        if 'toolbox_changed' in change:
            self.setEnabled()

    def resizeEvent(self, event: QResizeEvent) -> None:
        halfWidth = event.size().width() * 5 // 11
        self._networkGroupBox.setMinimumWidth(halfWidth)
        self._datasourceGroupBox.setMinimumWidth(halfWidth)
        # avoid image growing to large
        # FIXME[todo]: we can actually use the screen size to
        # compute the size:
        # app = QtGui.QApplication([])
        # screen_resolution = app.desktop().screenGeometry()
        # width, height = screen_resolution.width(), screen_resolution.height()
        maxsize = max(100,halfWidth-220)
        self._inputView.setMaximumSize(maxsize,maxsize)
        super().resizeEvent(event)
