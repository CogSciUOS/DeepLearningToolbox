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
from qtgui.utils import QObserver, protect
from qtgui.widgets import QNetworkView, QNetworkBox, QNetworkSelector
from qtgui.widgets import QInputSelector, QInputInfoBox, QModelImageView
from qtgui.widgets.network import QNetworkList
from qtgui.widgets.datasource import QDatasourceList

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QResizeEvent
from PyQt5.QtWidgets import (QPushButton, QGroupBox,
                             QVBoxLayout, QHBoxLayout, QGridLayout)


class ResourcesPanel(Panel, QObserver, Toolbox.Observer):
    '''A Panel for managing resources used by the Toolbox.

    _input_view: QModelImageView
        An image view area to display the currently selected input
        image from the Toolbox.
    _input_info: QInputInfoBox
        An info box displaying additional information on the currently
        selected input image.
    _input_selector: QInputSelector
        A control element to select an input image. Consists of
        two parts: a datasource selector to choose a Datasource and
        a datasource navigator to select an image in the Datasource.
    '''
    _toolboxController: ToolboxController = None
    _networkController: NetworkController = None

    _datasourceList: QDatasourceList = None
    _input_view: QModelImageView
    _input_info: QInputInfoBox
    _input_selector: QInputSelector
    
    def __init__(self, toolbox: ToolboxController=None,
                 network1: NetworkController=None,
                 datasource1: DatasourceController=None,
                 parent=None):
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

        @protect
        def clicked(checked: bool):
            datasource = self._toolboxController.hack_new_model()  # FIXME[todo]!
        self._buttonAddDatasource = QPushButton("Add datasource 1")
        self._buttonAddDatasource.clicked.connect(clicked)

        #
        # Input data
        #

        # QModelImageView: a widget to display the input data
        self._input_view = QModelImageView()

        self._input_info = QInputInfoBox()
        self._input_view.modeChanged.connect(self._input_info.onModeChanged)
        
        # QInputSelector: a widget to select the input
        # (combined datasource selector and datasource navigator)
        self._input_selector = QInputSelector()

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
        layout.addWidget(self._datasourceList)
        layout.addWidget(self._buttonAddDatasource)

        # FIXME[layout]
        #layout.setSpacing(0)
        #layout.setContentsMargins(0, 0, 0, 0)
        self._input_info.setMinimumWidth(200)
        # keep image view square (FIXME[question]: does this make
        # sense for every input?)
        self._input_view.heightForWidth = lambda w: w
        self._input_view.hasHeightForWidth = lambda: True

        layout.addWidget(self._input_view)
        layout.addWidget(self._input_info)
        layout.addWidget(self._input_selector)

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
        self._input_selector.setToolboxController(toolbox)
        self._input_view.setToolboxView(toolbox)
        self._input_info.setToolboxView(toolbox)
        self._datasourceList.setToolboxView(toolbox)

    def setNetworkController(self, network: NetworkController) -> None:
        self._networkController = network
        self._networkList.setNetworkView(network)
        self._networkSelector.setNetworkView(network)
        self._networkBox.setNetworkView(network)

    def setDatasourceController(self,
                                datasource: DatasourceController) -> None:
        self._datasourceList.setDatasourceView(datasource)

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
        self._input_view.setMaximumSize(halfWidth-20,halfWidth-20)
        super().resizeEvent(event)
