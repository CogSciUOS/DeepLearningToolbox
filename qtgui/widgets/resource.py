"""Base classes for resource widget.
A resource is everything that may require some extra installation of
hardware or software, including tools, data, and models (networks).
"""

# Generic imports
from typing import Optional
import logging

# Qt imports
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QGroupBox, QStackedWidget, QSplitter
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QSizePolicy

# toolbox imports
from toolbox import Toolbox
from dltb.network import Network, Layer
from dltb.datasource import Datasource, Datafetcher
from dltb.base.register import RegisterClass
from dltb.base.register import InstanceRegisterEntry, ClassRegisterEntry
from dltb.tool import Tool

# GUI imports
from ..utils import QAttribute, QObserver, protect
from .register import QRegisterClassView
from .register import QClassRegisterEntryController
from .register import QInstanceRegisterEntryController
from .network import QNetworkInternals, QLayerSelector, QNetworkBox
from .datasource import QDatasourceController, QDatasourceNavigator
from .data import QDataView

# logging
LOG = logging.getLogger(__name__)


class QRegisterController(QStackedWidget):
    """A :py:class:`QRegisterController` combines two controller,
    one class controller and one instance controller. Only one
    of them can be active at a time, and switching between them
    is done by either setting a :py:class:`ClassRegisterEntry`
    or an :py:class:`InstanceRegisterEntry`, respectively.

    The class controller allows to create a new instance of a class.

    The instance controller allows to instantiate an uninitialized
    instance and the prepare or unprepare that instance, once it has
    been initialized.

    """

    def __init__(self,
                 instanceEntryController:
                 QInstanceRegisterEntryController = None,
                 classEntryController:
                 QClassRegisterEntryController = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._initUI(instanceEntryController, classEntryController)

    def _initUI(self, instanceEntryController, classEntryController) -> None:
        self._instanceController = (instanceEntryController or
                                    QInstanceRegisterEntryController())
        self.addWidget(self._instanceController)

        self._classController = (classEntryController or
                                 QClassRegisterEntryController())
        self.addWidget(self._classController)

    def setInstanceEntry(self, entry: InstanceRegisterEntry) -> None:
        """Set the current instance register entry to be shown by
        this :py:class:`QRegisterController`.
        """
        self._instanceController.setRegisterEntry(entry)
        self.setCurrentWidget(self._instanceController)

    def setClassEntry(self, entry: ClassRegisterEntry) -> None:
        """Set the current class register entry to be shown by
        this :py:class:`QRegisterController`.
        """
        self._classController.setRegisterEntry(entry)
        self.setCurrentWidget(self._classController)

    def setEntry(self, entry: ClassRegisterEntry) -> None:
        """Set the current register entry to be shown by
        this :py:class:`QRegisterController`.
        """
        if isinstance(entry, InstanceRegisterEntry):
            self.setInstanceEntry(entry)
        else:
            self.setClassEntry(entry)


class QResourceInspector(QGroupBox, QObserver, qattributes={
        Toolbox: False, RegisterClass: False}, qobservables={
            InstanceRegisterEntry: {'state_changed'}}):
    """A Widget allowing to inspect a resource (this is essentially
    a :py:class:`RegisterClass`). It provides a list
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

    def __init__(self, registerClass: RegisterClass,
                 title: str = None, **kwargs):
        super().__init__(**kwargs)
        self._registerClass = registerClass
        self._resource = None
        self._initUI(title)
        self._layoutUI()

    def _initUI(self, title: str = None):
        self.setTitle(title or self._registerClass.__name__)

        self._resourceList = self._initResourceList()
        self._resourceList.setMode('instance')
        self._resourceList.instanceSelected.connect(self.onInstanceSelected)
        self._resourceList.classSelected.connect(self.onClassSelected)
        # self.addAttributePropagation(Toolbox, self._resourceList)
        self.addAttributePropagation(RegisterClass, self._resourceList)

        self._resourceController = self._initResourceController()
        self.addAttributePropagation(RegisterClass,
                                     self._resourceController)
        # self._resourceController.keyInitialized.connect(self.onKeySelected)

        self._resourceView = self._initResourceView()

    def _initResourceList(self) -> QRegisterClassView:
        """Create a resource list for this :py:class:`QResourceInspector`.
        """
        return QRegisterClassView(registerClass=self._registerClass)

    def _initResourceController(self) -> QRegisterController:
        return QRegisterController()

    def _initResourceView(self):
        # pylint: disable=no-self-use
        return QWidget()

    def _layoutUI(self) -> QWidget:
        layout = QVBoxLayout()

        #
        # The upper 'row' contains the resource list and the
        # resource controller
        #
        upper_row = QSplitter(Qt.Horizontal)
        upper_row.addWidget(self._resourceList)
        upper_row.addWidget(self._resourceController)

        #
        # The lower row just contains the resource view
        #
        lower_row = self._resourceView

        #
        # Splitter
        #
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(upper_row)
        splitter.addWidget(lower_row)
        layout = QHBoxLayout()
        layout.addWidget(splitter)

        self.setLayout(layout)

    def setToolbox(self, toolbox: Toolbox) -> None:  # FIXME[old]
        """Set the :py:class:`Toolbox` from which resources are obtained
        for this :py:class:`QResourceInspector`.
        """
        self.setTitle(f"{self._registerClass.base_class.__name__} "
                      f"({'without' if toolbox is None else 'with'} Toolbox)")

    @protect
    def onInstanceSelected(self, entry: InstanceRegisterEntry) -> None:
        """React to the selection of a resource by its key.
        """
        LOG.info("QResourceInspector.onInstanceSelected: key='%s' "
                 "[initialized=%s]",
                 entry and entry.key, entry and entry.initialized)
        self._resourceController.setEntry(entry)
        self.setInstanceRegisterEntry(entry)

    @protect
    def onClassSelected(self, entry: ClassRegisterEntry) -> None:
        """React to the selection of a class key.
        """
        LOG.info("QResourceInspector.onClassSelected: key='%s' "
                 "[initialized=%s]",
                 entry and entry.key, entry and entry.initialized)
        self._resourceController.setEntry(entry)
        self.setInstanceRegisterEntry(None)

    def setResource(self, resource: object) -> None:
        """Set the current resource for this :py:class:`QResourceInspector`.
        """
        if resource is self._resource:
            return  # nothing changed

        self._resource = resource
        self._updateResource()

    def _updateResource(self) -> None:
        """React to a change of resource.
        """
        self._resourceList.setInstance(self._resource)

    #
    # Observations
    #

    def entry_changed(self, entry: InstanceRegisterEntry,
                      info: InstanceRegisterEntry.Change) -> None:
        """The currently observed :py:class:`InstanceRegisterEntry`
        has changed.
        """
        LOG.info("QResourceInspector.state_changed: key='%s' [info=%s]",
                 entry.key, info)
        self.setResource(entry.obj)


class QNetworkInspector(QResourceInspector):
    """The `QNetworkInspector` is a :py:class:`QResourceInspector` for
    managing network. It provides the following components:

    _resourceList: QRegisterClassView = None
    
    _layerSelector: QLayerSelector = None
    _networkBox: QNetworkBox = None
    _networkInternals: QNetworkInternals = None
    """

    def __init__(self, title: str = 'Networks', **kwargs) -> None:
        self._layerSelector = None
        self._networkBox = None
        self._networkInternals = None
        self._toolbox = None
        super().__init__(registerClass=Network, title=title, **kwargs)

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

        self._layerSelector.layerClicked.connect(self._onLayerClicked)

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

    def setNetwork(self, network: Network) -> None:
        """Set the network for this :py:class:`QNetworkInspector`.
        """
        self.setResource(network)

    @protect
    def _onLayerClicked(self, layer_name: str, selected: bool) -> None:
        """React to a layer seletion in the :py:class:`QLayerSelector`.
        """
        print(f"QNetworkInspector._onLayerClicked: {layer_name} ({selected})")
        self._networkBox.setLayer(layer_name if selected else None)
        
    def layer(self) -> Optional[Layer]:
        """The currently selected layer in this :py:class:`QNetworkInspector`.
        """
        return self._layerSelector.layer()

    def setLayer(self, layer: Optional[Layer]) -> None:
        """Change the currently selected layer.
        """
        self._layerSelector.setLayer(layer)

    def _updateResource(self) -> None:
        """React to a change of resource (network) for this
        :py:class:`QNetworkInspector`.
        """
        super()._updateResource()
        network = self._resource
        LOG.debug("QNetworkInspector._updateResource: %s", network)
        self._layerSelector.setNetwork(network)
        self._networkBox.setNetwork(network)
        self._networkInternals.setNetwork(network)


class QDatasourceInspector(QResourceInspector, qattributes={
        Datasource: False, Datafetcher: False}):
    """The :py:class:`QDatasourceInspector` contains widgets for
    inspecting :py:class:`Datasource`s.

    Attributes
    ----------
    The datasource resource panel consists of:

    _resourceList: QDatasourceListWidget
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
        super().__init__(registerClass=Datasource, title=title, **kwargs)

    def _initResourceList(self) -> QRegisterClassView:
        datasourceList = QRegisterClassView()
        datasourceList.setRegisterClass(self._registerClass)
        return datasourceList

    def _initResourceController(self) -> QRegisterController:
        controller = QDatasourceController()
        self.addAttributePropagation(Toolbox, controller)
        return QRegisterController(instanceEntryController=controller)

    def _initResourceView(self) -> QWidget:
        # pylint: disable=attribute-defined-outside-init
        """Initialize the resource view. The Datasource view will consist
        of two components: a :py:class:`QDatasourceNavigator` allowing
        to navigate through the :py:class:`Datasource`, and a
        :py:class:`QDataView` for displaying selected :py:class:`Data`.
        """
        #
        # Creating the components
        #
        self._datasourceNavigator = \
            QDatasourceNavigator(datasource_selector=False, style='wide')
        self.addAttributePropagation(Datafetcher, self._datasourceNavigator)
        self.addAttributePropagation(Datasource, self._datasourceNavigator)

        self._dataView = QDataView(orientation=Qt.Horizontal)
        self._dataView.setDatafetcher(self._datasourceNavigator.datafetcher())
        self.addAttributePropagation(Datafetcher, self._dataView)

        self._datasourceView = QWidget()

        #
        # Layout
        #
        layout = QVBoxLayout()
        layout.addWidget(self._datasourceNavigator)
        layout.addWidget(self._dataView)

        self._datasourceNavigator.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Fixed)
        
        self._datasourceView.setLayout(layout)
        return self._datasourceView

    def _updateResource(self) -> None:
        """React to a change of resource (datasouce) for this
        :py:class:`QNetworkInspector`.
        """
        super()._updateResource()
        datasource = self._resource
        self.setDatasource(datasource)

    # FIXME[debug]
    def NOresizeEvent(self, event):
        print("RESIZE:")

        def _(size):
            return f"{size.width()}x{size.height()},"

        def p(policy):
            if policy == QSizePolicy.Fixed:
                return 'Fixed'
            if policy == QSizePolicy.Minimum:
                return 'Minimum'
            if policy == QSizePolicy.Maximum:
                return 'Maximum'
            if policy == QSizePolicy.Preferred:
                return 'Preferred'
            if policy == QSizePolicy.Expanding:
                return 'Expanding'
            if policy == QSizePolicy.MinimumExpanding:
                return 'MinimumExpanding'
            return f'?{policy}'

        for widget in (self._datasourceView,
                       self._datasourceNavigator,
                       self._dataView,
                       self._dataView._imageView,
                       self._dataView._dataInfo):
            print(f" - {type(widget).__name__:30}: "
                  f"size={_(widget.size()):12} "
                  f"min={_(widget.minimumSize()):12} "
                  f"max={_(widget.maximumSize()):24} "
                  f"size hint={_(widget.sizeHint()):12} "
                  f"minimum size hint={_(widget.minimumSizeHint()):12}"
                  f"policy: {p(widget.sizePolicy().horizontalPolicy())}/"
                  f"{p(widget.sizePolicy().verticalPolicy())}"
                  f", stretch={widget.sizePolicy().horizontalStretch()}/"
                  f"{widget.sizePolicy().verticalStretch()}")


class QToolInspector(QResourceInspector):
    """A :py:class:`QToolInspector` lists :py:class:`Tool`\\s.
    """

    def __init__(self, title: str = 'Tools', **kwargs) -> None:
        super().__init__(registerClass=Tool, title=title, **kwargs)

    def _initResourceList(self) -> QRegisterClassView:
        toolList = super()._initResourceList()
        toolList.instanceSelected.connect(self.onToolInstanceSelected)
        toolList.classSelected.connect(self.onToolClassSelected)
        return toolList

    def setTool(self, tool: Tool, key: str = None) -> None:
        """Set the :py:class:`Tool` to be displayed by this
        :py:class:`QToolInspector`.
        """
        # FIXME[old]: is this still needed?
        # self._resourceController.setObject(key or tool)

    @protect
    def onToolInstanceSelected(self, entry: InstanceRegisterEntry) -> None:
        """A slot for reacting to tool key selection in a tool selection
        widget. If the corresponding :py:class:`Tool` is initialized,
        it will be set as the current tool in this
        :py:class:`QToolInspector`, otherwise an initialization element
        will be shown.
        """
        LOG.info("QToolInspector.onToolKeySelected: key='%s' -> %s",
                 entry.key, entry.obj)
        self.setTool(entry.obj)

    @protect
    def onToolClassSelected(self, _entry: ClassRegisterEntry) -> None:
        """A slot for reacting to tool class selection in a tool class
        selection widget.  This will switch the resource controller
        into class mode and display the selected class.
        """
        self.setTool(None)
        # FIXME[old]: is this still needed?
        # self._resourceController.setClass(key)

