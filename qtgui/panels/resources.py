"""
File: resources.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QListWidget, QListWidgetItem, QPushButton,
                             QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout)

from .panel import Panel
from qtgui.utils import QObserver
from qtgui.widgets import QNetworkView, QNetworkBox, QNetworkSelector
from qtgui.widgets import QInputSelector, QInputInfoBox, QModelImageView

from toolbox import Toolbox, ToolboxController, ToolboxView
from network import Network, Controller as NetworkController
from datasources import Datasource, Controller as DatasourceController

class ResourcesPanel(Panel, QObserver, Toolbox.Observer,
                     Network.Observer, Datasource.Observer):
    '''A Panel for managing resources used by the Toolbox.
    '''
    _toolboxController: ToolboxController=None
    _networkController : NetworkController=None
    _datasourceController : DatasourceController=None

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

        def itemClicked(item: QListWidgetItem):
            self._networkController(item.data(Qt.UserRole))
        self._networkList = QListWidget()
        self._networkList.itemClicked.connect(itemClicked)

        def clicked(checked: bool):
            network = self._toolboxController.hack_new_model()
        self._button = QPushButton("Add Network 1")
        self._button.clicked.connect(clicked)

        def clicked(checked: bool):
            network = self._toolboxController.hack_new_model2()
        self._button2 = QPushButton("Add Network 2")
        self._button2.clicked.connect(clicked)

        def clicked(checked: bool):
            network = self._toolboxController.hack_new_alexnet()
        self._button3 = QPushButton("Load Alexnet")
        self._button3.clicked.connect(clicked)

        def currentIndexChanged(index: int):
            self._networkController(self._networkSelector.currentData())
        self._networkSelector = QNetworkSelector()
        self._networkSelector.currentIndexChanged.connect(currentIndexChanged)

        self._networkView = QNetworkView()
        self._networkBox = QNetworkBox()


        #
        # Datasources
        #

        def itemClicked(item: QListWidgetItem):
            self._datasourceController(item.data(Qt.UserRole))
        self._datasourceList = QListWidget()
        self._datasourceList.itemClicked.connect(itemClicked)

        def clicked(checked: bool):
            datasource = self._toolboxController.hack_new_model()  # FIXME[todo]!
        self._buttonAddDatasource = QPushButton("Add datasource 1")
        self._buttonAddDatasource.clicked.connect(clicked)

        #
        # Input data
        #

        # QModelImageView: a widget to display the input data
        self._input_view = QModelImageView(self)
        # FIXME[layout]
        # keep image view square (TODO: does this make sense for every input?)
        self._input_view.heightForWidth = lambda w: w
        self._input_view.hasHeightForWidth = lambda: True

        # QInputSelector: a widget to select the input to the network
        # (data array, image directory, webcam, ...)
        # the 'next' button: used to load the next image
        self._input_selector = QInputSelector()

        # FIXME[hack]
        self._input_info = QInputInfoBox(imageView=self._input_view)
        #self._input_view.modeChange.connect(self._input_info.onModeChange)

        # FIXME[layout]
        self._input_info.setMinimumWidth(200)


        

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

        box1 = QGroupBox('Networks')
        box1.setLayout(layout2)

        layout = QVBoxLayout()
        layout.addWidget(self._datasourceList)
        layout.addWidget(self._buttonAddDatasource)

        # FIXME[layout]
        #layout.setSpacing(0)
        #layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._input_view)
        layout.addWidget(self._input_info)
        layout.addWidget(self._input_selector)

        layout.addStretch()

        box2 = QGroupBox('Data sources')
        box2.setLayout(layout)

        layout = QGridLayout()
        box1.sizePolicy().setHorizontalStretch(1)
        box2.sizePolicy().setHorizontalStretch(1)
        layout.addWidget(box1, 0, 0)
        layout.addWidget(box2, 0, 1)
        self.setLayout(layout)

    def setToolboxController(self, toolbox: ToolboxController) -> None:
        self._exchangeView('_toolboxController', toolbox)
        self._networkSelector.setToolboxView(toolbox)
        self._input_selector.setToolboxController(toolbox)

    def setNetworkController(self, network: NetworkController) -> None:
        self._exchangeView('_networkController', network)
        self._networkSelector.setNetworkView(network)
        self._networkBox.setNetworkView(network)

    def setDatasourceController(self,
                                datasource: DatasourceController) -> None:
        self._exchangeView('_datasourceController', datasource,
                           interests=Datasource.Change('observable_changed'))
        # FIXME[hack]: not really interested ...

    def setEnabled(self):
        enabled = self._toolboxController is not None
        self._networkList.setEnabled(enabled)
        self._button.setEnabled(enabled)

    def toolbox_changed(self, toolbox, change):
        self._toolboxController(toolbox)

        if 'toolbox_changed' in change:
            self.setEnabled()
        
        if 'networks_changed' in change:
            # Update the networks list:
            self._networkList.clear()
            for network in self._toolboxController.networks:
                item = QListWidgetItem(str(network))
                item.setData(Qt.UserRole, network)
                self._networkList.addItem(item)

        if 'datasources_changed' in change:
            # Update the datasources list:
            self._datasourceList.clear()
            for datasource in self._toolboxController.datasources:
                item = QListWidgetItem(str(datasource))
                item.setData(Qt.UserRole, datasource)
                self._datasourceList.addItem(item)


    def network_changed(self, network, change):
        for i in range(self._networkList.count()):
            if self._networkList.item(i).data(Qt.UserRole) == network:
                self._networkList.setCurrentRow(i)
                break

    def datasource_changed(self, datasource, change):
        print(f"datasource_changed({self}, {datasource}, {change})")
