"""
File: resources.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
"""

# Generic imports
import logging

# Qt imports
from PyQt5.QtGui import QResizeEvent
from PyQt5.QtWidgets import (QWidget, QGroupBox, QTabWidget,
                             QVBoxLayout, QHBoxLayout, QGridLayout)

# toolbox imports
from toolbox import Toolbox
from network import Network
from datasource import Datasource
from tools import Tool
from base import MetaRegister

# GUI imports
from .panel import Panel
from ..utils import QAttribute, QObserver, protect
from ..widgets.register import QRegisterList, QRegisterController
from ..widgets.network import (QNetworkList, QNetworkInternals,
                               QLayerSelector, QNetworkBox)
from ..widgets.datasource import (QDatasourceList,
                                  QDatasourceController, QDatasourceNavigator)
from ..widgets.data import QDataView

# logging
LOG = logging.getLogger(__name__)


class QResourceInspector(QGroupBox, QAttribute, qattributes={
        Toolbox: False, MetaRegister: False}):
    """A Widget allowing to inspect a type of widget. It provides a list
    to select a resource, a controller allowing basic operations like
    preparation, and a view for displaying its properties.

    _resourceList:
        A widget allowing to select a Resource (or a key of a Resource).

    _resourceController:
        A resource controller allowing to initialize a resource and
        to (un)prepare the resource (if :py:class:`Preparable`).

    _resourceView:
        A widget that allows to inspect the Resource.

    _toolbox: Toolbox
    """

    def __init__(self, register, *args, title: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._register = register
        self._initUI(title)
        self._layoutUI()

    def _initUI(self, title: str = None):
        self.setTitle(title or self._register.__name__)

        self._resourceList = self._initResourceList()
        self._resourceList.keySelected.connect(self.onKeySelected)
        self._resourceList.classSelected.connect(self.onClassSelected)
        self.addAttributePropagation(Toolbox, self._resourceList)
        self.addAttributePropagation(MetaRegister, self._resourceList)

        self._resourceController = self._initResourceController()
        self.addAttributePropagation(MetaRegister,
                                     self._resourceController)
        self._resourceController.keyInitialized.connect(self.onKeySelected)

        self._resourceView = self._initResourceView()

    def _initResourceList(self) -> QRegisterList:
        """Create a resource list for this :py:class:`QResourceInspector`.
        """
        return QRegisterList(register=self._register)

    def _initResourceController(self) -> QRegisterController:
        return QRegisterController(register=self._register)

    def _initResourceView(self):
        return QWidget()

    def _layoutUI(self) -> QWidget:
        layout = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(self._resourceList)
        row.addWidget(self._resourceController)
        layout.addLayout(row)
        layout.addWidget(self._resourceView)
        layout.addStretch()
        self.setLayout(layout)

    def setToolbox(self, toolbox: Toolbox) -> None:  # FIXME[old]
        self.setTitle(f"{self._register.__name__} "
                      f"({'without' if toolbox is None else 'with'} Toolbox)")

    def setResource(self, resource) -> None:
        pass  # to be implemented by subclasses

    def setClassKey(self, key: str) -> None:
        self._resourceController.setClass(key)

    @protect
    def onKeySelected(self, key: str) -> None:
        resource = (self._register[key]
                    if key and self._register.key_is_initialized(key) else None)
        LOG.info("QResourceInspector.onKeySelected: key='%s', "
                 "Resource: %s (%s), Toolbox: %s",
                 key, resource, self._register.__name__,
                 getattr(self, '_toolbox', None))
        self._resourceController.setObject(resource or key)
        self.setResource(resource)

    @protect
    def onClassSelected(self, key: str) -> None:
        """React to the selection of a class key.
        """
        self.setClassKey(key)


class QNetworkInspector(QResourceInspector, QObserver,
                        qattributes={Datasource: False}):
    """

    _resourceList: QNetworkList = None
    _layerSelector: QLayerSelector = None
    _networkBox: QNetworkBox = None
    _networkInternals: QNetworkInternals = None

    """

    def __init__(self, title: str = 'Networks', **kwargs) -> None:
        self._layerSelector = None
        self._networkBox = None
        self._networkInternals = None
        self._toolbox = None
        super().__init__(Network, title=title, **kwargs)

# FIXME[old]
#    def _initResourceList(self):
#        return QNetworkList()
#    def _initResourceList(self) -> QRegisterList:
#        LOG.debug("QNetworkInspector: initResourceList")
#        networkList = super()._initResourceList()
#        return networkList

    def _initResourceView(self):
        """The ResourceView of a :py:class:`QNetworkInspector`
        consists of a :py:class:`QLayerSelector`, a
        :py:class:`QNetworkBox` and a :py:class:`QNetworkInternals`
        widget.
        """
        LOG.debug("QNetworkInspector: initResourceView")
        self._layerSelector = QLayerSelector()
        self._networkBox = QNetworkBox()
        self._networkInternals = QNetworkInternals()

        networkView = QWidget()

        layout = QHBoxLayout()
        column = QVBoxLayout()
        column.addWidget(self._layerSelector)
        column.addWidget(self._networkBox)
        column.addStretch()
        layout.addLayout(column)
        layout.addWidget(self._networkInternals)
        networkView.setLayout(layout)
        return networkView

    def setResource(self, resource) -> None:
        self.setNetwork(resource)

    def setNetwork(self, network: Network, key: str = None) -> None:
        LOG.debug("QNetworkInspector.setNetwork(%s, '%s')", network, key)
        self._resourceController.setObject(key or network)
        self._layerSelector.setNetwork(network)
        self._networkBox.setNetwork(network)
        self._networkInternals.setNetwork(network)

    @protect
    def onKeySelected(self, key: str) -> None:
        """A slot for reacting to network key selection in a network selection
        widget. If the corresponding :py:class:`Network` is initialized,
        it will be set as the current network in this
        :py:class:`QNetworkInspector`, otherwise an initialization element
        will be shown.

        """
        network = (Network[key] if key and
                   Network.key_is_initialized(key) else None)
        LOG.info("QNetworkInspector.onNetworkKeySelected: key='%s' -> %s",
                 key, network)
        self.setNetwork(network, key)

        # FIXME[hack]
        if self._toolbox is not None and network is not None:
            LOG.info("QNetworkInspector: add network %s to toolbox", network)
            self._toolbox.add_network(network)

    @protect
    def onClassSelected(self, key: str) -> None:
        """A slot for reacting to network class selection in a network class
        selection widget.  This will switch the resource controller
        into class mode and display the selected class.
        """
        self.setNetwork(None)
        self._resourceController.setClass(key)


class QDatasourceInspector(QResourceInspector, QObserver,
        qattributes={Datasource: False},
        qobservables={Toolbox: Toolbox.Change('datasource_changed')}):
    """

    Attributes
    ----------
    The datasource resource panel consists of:

    _resourceList: QDatasourceList
        A list of datasources/classes, allowing to select a
        :py:class:`Datasource`.

    _resourceController: QDatasourceController
        A datasource controller, showing general information and
        allowing to prepare and unprepare a datasouce.
        Also allows to initialize datasources from register,
        and to import Datasource modules.

    _datasourceNavigator: QDatasourceNavigator
        A datasouce navigator allowing to navigate through a
        datasource

    _dataView: QDataView
        A data view area to display the currently selected element from
        the :py:class:`Datasource`
        image from the Toolbox.
    """

    def __init__(self, title: str = 'Datasources', **kwargs) -> None:
        super().__init__(Datasource, title=title, **kwargs)

    def _initResourceList(self) -> QRegisterList:
        datasourceList = QDatasourceList()
        datasourceList.setRegister(self._register)
        return datasourceList

    def _initResourceController(self) -> QRegisterController:
        datasourceController = QDatasourceController()
        # FIXME[hack]: connect the "Set as Toolbox Datasource"-Button
        datasourceController.button.clicked.connect(
            self.onDatasourceButtonClicked)
        return datasourceController

    def _initResourceView(self) -> QWidget:
        self._datasourceNavigator = QDatasourceNavigator()
        self.addAttributePropagation(Datasource, self._datasourceNavigator)

        self._dataView = QDataView()
        self.addAttributePropagation(Datasource, self._dataView)

        datasourceView = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self._datasourceNavigator)
        layout.addWidget(self._dataView, stretch=20)
        datasourceView.setLayout(layout)
        return datasourceView

    def setResource(self, resource: Datasource) -> None:
        """Set the resource to be used by this
        :py:class:`QDatasourceInspector`.

        Arguments
        ---------
        resource:
            The resource is expected to be a :py:class:`Datasource`.
        """
        self.setDatasource(resource)

    @protect
    def onDatasourceButtonClicked(self, checked: bool):
        """React to a click on the "Set as Toolbox Datasource"-Button.
        If a :py:class:`Toolbox` is present, this currently selected
        :py:class:`Datasource` will be assigned as the active datasource
        of the toolbox.
        """
        if self._toolbox is not None:
            # FIXME[hack] ...
            self._toolbox.datasource = self._resourceController._object

    def setClassKey(self, key: str) -> None:
        super().setClassKey(key)
        self.setDatasource(None)

    # FIXME[todo]: for this to work we have to observe the toolbox!
    def toolbox_changed(self, toolbox, change):
        LOG.debug("%s.toolbox_changed(%s, %s)",
                  type(self).__name__, type(toolbox).__name__, change)
        # if 'datasource_changed' in change:
        #    self.setDatasource(toolbox.datasource)


class QToolInspector(QResourceInspector):

    def __init__(self, title: str = 'Tools', **kwargs) -> None:
        super().__init__(Tool, title=title, **kwargs)

    def _initResourceList(self) -> QRegisterList:
        toolList = super()._initResourceList()
        toolList.keySelected.connect(self.onToolKeySelected)
        toolList.classSelected.connect(self.onToolClassSelected)
        return toolList

    def setTool(self, tool: Tool, key: str = None) -> None:
        """Set the :py:class:`Tool` to be displayed by this
        :py:class:`QToolInspector`.
        """
        self._resourceController.setObject(key or tool)

    @protect
    def onToolKeySelected(self, key: str) -> None:
        """A slot for reacting to tool key selection in a tool selection
        widget. If the corresponding :py:class:`Tool` is initialized,
        it will be set as the current tool in this
        :py:class:`QToolInspector`, otherwise an initialization element
        will be shown.
        """
        tool = (Tool[key] if key and Tool.key_is_initialized(key) else None)
        LOG.info("QToolInspector.onToolKeySelected: key='%s' -> %s",
                 key, tool)
        self.setTool(tool, key)

    @protect
    def onToolClassSelected(self, key: str) -> None:
        """A slot for reacting to tool class selection in a tool class
        selection widget.  This will switch the resource controller
        into class mode and display the selected class.
        """
        self.setTool(None)
        self._resourceController.setClass(key)


class ResourcesPanel(Panel, QAttribute, qattributes={
        Toolbox: False, Network: False, Datasource: False}):
    """A Panel for managing resources used by the Toolbox.

    The :py:class:`ResourcesPanel` can be used with or without a
    :py:class:`Toolbox`:

    * without a Toolbox, the select signals will set the observables
      of other view and control widgets.

    * with a Toolbox, the select signals will set the corresponding
      properties of the Toolbox. Views and control widgets will observe
      the Toolbox and to change their observable accordingly.

    There a currently the following sections in this panel:
    * Networks

    * Datasources

    * Tools

    _networkInspector: QNetworkInspector = None
    _datasourceInspector: QDatasourceInspector = None
    _toolInspector: QToolInspector = None
    """

    def __init__(self, toolbox: Toolbox = None, network: Network = None, **kwargs):
        super().__init__(**kwargs)
        self._initUI()
        self._layoutUI()
        self.setToolbox(toolbox)
        self.setNetwork(network)

    def _initUI(self):
        self._networkInspector = QNetworkInspector()
        self.addAttributePropagation(Toolbox, self._networkInspector)
        self.addAttributePropagation(Network, self._networkInspector)

        self._datasourceInspector = QDatasourceInspector()
        self.addAttributePropagation(Toolbox, self._datasourceInspector)
        #self.addAttributePropagation(Datasouce, self._datasourceInspector)

        self._toolInspector = QToolInspector()

    def _layoutUI(self, style: str = 'Tabs'):
        layout = QGridLayout()
        if style == 'Grid':
            self._networkInspector.sizePolicy().setHorizontalStretch(1)
            layout.addWidget(self._networkInspector, 0, 0)

            self._datasourceInspector.sizePolicy().setHorizontalStretch(1)
            layout.addWidget(self._datasourceInspector, 0, 1)
        else:
            tabs = QTabWidget()
            tabs.setTabPosition(tabs.West)
            tabs.addTab(self._networkInspector, 'Networks')
            tabs.addTab(self._datasourceInspector, 'Datasources')
            tabs.addTab(self._toolInspector, 'Tools')
            # FIXME[hack]: for developing datasouces ...
            tabs.setCurrentWidget(self._datasourceInspector)
            layout.addWidget(tabs, 0, 0)
        self.setLayout(layout)

    def resizeEvent(self, event: QResizeEvent) -> None:
        halfWidth = event.size().width() * 5 // 11
        self._networkInspector.setMinimumWidth(halfWidth)
        self._datasourceInspector.setMinimumWidth(halfWidth)
        # avoid image growing to large
        # FIXME[todo]: we can actually use the screen size to
        # compute the size:
        # app = QtGui.QApplication([])
        # screen_resolution = app.desktop().screenGeometry()
        # width, height = screen_resolution.width(), screen_resolution.height()
        # FIXME[hack]
        maxsize = max(100, halfWidth-220)
        self._datasourceInspector._dataView.setMaximumSize(maxsize, maxsize+50)
        super().resizeEvent(event)

    #
    # Datasource
    #

    def setDatasource(self, datasource: Datasource, key: str = None) -> None:
        """Set a :py:class:`Datasource` for this :py:class:`ResourcesPanel`.
        The datasource will be propagated to the
        :py:class:`QDatasourceInspector` of this :py:class:`ResourcesPanel`.

        """
        self._datasourceInspector.setDatasource(datasource, key)
