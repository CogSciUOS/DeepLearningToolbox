"""
File: activations.py
Author: Petr Byvshev, Ulf Krumnack, Rasmus Diederichsen
Email: rdiederichse@uni-osnabrueck.de
Github: https://github.com/themightyoarfish
"""

# Generic imports
import logging

# Qt imports
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
from PyQt5.QtWidgets import QWidget, QGroupBox, QSplitter
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout

# toolbox imports
from dltb.tool.activation import ActivationTool, ActivationWorker
from dltb.tool.image import ImageTool
from dltb.network import Network
from dltb.datasource import Datasource
from dltb.util.array import DATA_FORMAT_CHANNELS_FIRST
from dltb.util.image import grayscaleNormalized

from toolbox import Toolbox

# GUI imports
from .panel import Panel
from ..utils import QObserver, QPrepareButton, protect
from ..widgets.activationview import QActivationView
from ..widgets.data import QDataSelector
from ..widgets.network import QLayerSelector, QNetworkComboBox
from ..widgets.classesview import QClassesView


# logging
LOG = logging.getLogger(__name__)

# FIXME[bug]: when
# 1. loading an image and then
# 2. preparing the network
# 3. selecting a layer
# the activations are not computed.
# Desired behavior: activations should be computed.

# FIXME[todo]: show some busy widget for operations that may take some time:
# - initializing/perparing a network
# - computing activations (especially for the first run)


class ActivationsPanel(Panel, QObserver, qobservables={
        Toolbox: {'datasource_changed', 'input_changed'},
        ActivationWorker: {'data_changed', 'work_finished'}}, qattributes={
            Datasource: False, Network: False}):
    # pylint: disable=too-many-instance-attributes
    """The :py:class:`ActivationsPanel` is a graphical frontend for an
    :py:class:`ActivationTool` and the an associated
    :py:class:`ActivationWorker`. It allows to set the network for
    that tool, it feeds input to the associated
    :py:class:`ActivationWorker` and displays activation maps computed
    by the tool.

    The :py:class:`ActivationsPanel` complex panel consisting
    of three parts:
    (A) activation: display of layer and unit activations
    (B) input: select and display inputs
    (C) network: display network information and select network layers

    The :py:class:`ActivationsPanel` can be used with a
    :py:class:`Toolbox`, in which case the selection of the network
    and input data are restricted to the resources offered by that
    toolbox. If no toolbox is used, network and datasource selection
    have to configured with the _networkSelector and
    _datasourceNavigator components.

    Attributes
    ----------

    The :py:class:`ActivationsPanel` holds a reference
    to the underlying :py:class:`dltb.tool.activation.ActivationTool`:

    _activationTool: ActivationTool
        The :py:class:`ActivationsPanel` holds a reference to the
        tool allowing to access the network currently used.
        It also acts as a controller allowing to change that network.

    _activationWorker: ActivationWorker
        The :py:class:`ActivationsPanel` will observe this worker
        to reflect its state. The worker is also propagated to
        the _activationView.

    In addition, the :py:class:`ActivationsPanel` also holds the
    following protected attributes based on the current state of the
    underlying activation worker:

    _layerID: str
        The currently selected unit in the activation map.

    _unit: int
        The currently selected unit in the activation map.

    _receptiveField:
        The currently selected receptive field. A receptiv field can
        be selected in the _activationView and can be displayed
        in the _imageView.


    Graphical components
    --------------------

    The :py:class:`ActivationsPanel` contains the following graphical
    components.

    (A) activation:

    _activationView: QActivationView = None
        A :py:class:`QActivationView` for displaying activations of
        a network layer.

    (B) input:

    _imageView: QImageView
        A :py:class:`QImageView` for displaying the current data
        used as input for the network.
    _dataView: QDataView

    #_inputInfoBox: QDataInfoBox

    _datasourceNavigator: QDatasourceNavigator

    (C) network:

    _networkSelector: QNetworkComboBox
    _networkPrepareButton: QPrepareButton

    _layerSelector: QLayerSelector
        A widget for different network related activities: selecting
        a network from a list of networks, selecting a layer for a network,
        displaying network information and network output.

    _classesView: QClassesView = None
        A :py:class:`QClassesView` for displaying classification results
        in case the current :py:class:`Network` is a
        :py:class:`Classifier`.

    """

    def __init__(self, toolbox: Toolbox = None,
                 network: Network = None,
                 datasource: Datasource = None,
                 **kwargs) -> None:
        """Initialization of the ActivationsPael.

        Parameters
        ----------
        toolbox: Toolbox
        network: Network
        parent: QWidget
            The parent argument is sent to the QWidget constructor.
        """
        self._layerID = None
        self._unit = None
        self._receptiveField = None
        LOG.info("ActivationsPanel(toolbox=%s, network=%s, datasource=%s)",
                 toolbox, network, datasource)
        super().__init__(**kwargs)
        self._initUI()
        self._layoutUI()

        self._activationTool = \
            ActivationTool(data_format=DATA_FORMAT_CHANNELS_FIRST)
        worker = ActivationWorker(tool=self._activationTool)
        self.setActivationWorker(worker)

        self.setToolbox(toolbox)
        self.setNetwork(network)
        self.setDatasource(datasource)

        if network is None and self._networkSelector.count() > 0:
            self._networkSelector.setCurrentIndex(0)

    def _initUI(self):
        """Initialize all UI elements. These are
        * The ``QActivationView`` showing the unit activations on the left
        * The ``QImageView`` showing the current input image
        * A ``QDatasourceNavigator`` to show datasource navigation controls
        * A ``QLayerSelector``, a widget to select a layer in a network
        * A ``QDataInfoBox`` to display information about the input
        """

        #
        # (A) Activations
        #

        # QActivationView: a canvas to display a layer activation
        self._activationView = QActivationView(worker=self._activationWorker)
        self._activationView.unitChanged.connect(self.onUnitChanged)
        self._activationView.positionChanged.connect(self.onPositionChanged)
        self.addAttributePropagation(ActivationWorker, self._activationView)

        # QClassesView: display classification results
        self._classesView = QClassesView()
        self._classesViewBox = QGroupBox('Classification')
        self._classesViewBox.setCheckable(True)
        classesViewLayout = QVBoxLayout()
        classesViewLayout.addWidget(self._classesView)
        self._classesViewBox.setLayout(classesViewLayout)

        #
        # (B) Input
        #

        # QImageView: a widget to display the input data
        self._dataSelector = QDataSelector()
        self.addAttributePropagation(Toolbox, self._dataSelector)

        self._imageView = self._dataSelector.imageView()
        # self.addAttributePropagation(Toolbox, self._imageView)

        # QDataView: display data related to the current input data
        self._dataView = self._dataSelector.dataView()
        self._dataView.addAttribute('filename')
        self._dataView.addAttribute('basename')
        self._dataView.addAttribute('directory')
        self._dataView.addAttribute('path')
        self._dataView.addAttribute('regions')
        self._dataView.addAttribute('image')

        # QDatasourceNavigator: navigate through the datasource
        self._datasourceNavigator = self._dataSelector.datasourceNavigator()
        self.addAttributePropagation(Toolbox, self._datasourceNavigator)

        # self.addAttributePropagation(Toolbox, self._datasourceNavigator)

        #
        # (C) Network
        #

        # QNetworkComboBox: a widget to select a network
        self._networkSelector = QNetworkComboBox()
        self.addAttributePropagation(Toolbox, self._networkSelector)
        self.addAttributePropagation(Network, self._networkSelector)
        # FIXME[note]: the 'activated' signal is emitted on every
        # activation, not just on changes - we use this during development
        # as it may help to trigger the event on purpose.  However, in future
        # 'activated' may be replaced by 'currentIndexChanged' which
        # will only be emitted if the selected network actually changed.
        self._networkSelector.networkSelected.\
            connect(self.onNetworkSelected)
        # self._networkSelector.currentIndexChanged[str].\
        #    connect(self.onNetworkSelectorChanged)

        # QPrepareButton: a button to (un)prepare the network
        self._networkPrepareButton = QPrepareButton()

        # QLayerSelector: a widget to select a network layer
        self._layerSelector = QLayerSelector()
        self._layerSelector.layerClicked.connect(self.onLayerSelected)
        self.addAttributePropagation(Network, self._layerSelector)

    def _layoutUI(self):
        """Layout the graphical user interface.

        The standard layout will consist of three columns:
        The activation maps are shown in the left column.
        The center column contains widgets for displaying and selecting
        the input data.
        The right column contains controls for selecting network and
        network layer.
        """

        #
        # Activations (left column)
        #
        activationLayout = QVBoxLayout()
        activationLayout.addWidget(self._activationView)
        # FIXME[layout]
        self._activationView.setMinimumWidth(200)
        self._activationView.resize(400, self._activationView.height())

        activationBox = QGroupBox('Activation')
        activationBox.setLayout(activationLayout)

        #
        # Input data (center column)
        #
        inputLayout = QVBoxLayout()
        inputLayout.addWidget(self._dataSelector)

        inputBox = QGroupBox('Input')
        inputBox.setLayout(inputLayout)

        #
        # Network (right column)
        #
        rightLayout = QVBoxLayout()
        networkBox = QGroupBox("Network")
        networkLayout = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(self._networkSelector, stretch=2)
        row.addWidget(self._networkPrepareButton, stretch=1)
        networkLayout.addLayout(row)
        networkLayout.addWidget(self._layerSelector)
        networkBox.setLayout(networkLayout)
        rightLayout.addWidget(networkBox)

        # classes
        rightLayout.addWidget(self._classesViewBox)

        #
        # Attach widgets to window
        #
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(activationBox)
        splitter.addWidget(inputBox)

        rightWidget = QWidget()
        rightWidget.setLayout(rightLayout)
        rightLayout.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(rightWidget)

        layout = QHBoxLayout()
        layout.addWidget(splitter)
        self.setLayout(layout)

    def setActivationWorker(self, worker: ActivationWorker) -> None:
        """Set the underlying :py:class:`ActivationWorker`.
        """
        LOG.debug("ActivationsPanel.setActivationWorker(%s)", worker)

    def network(self) -> Network:
        """The network used by this :py:class:`ActivationsPanel`, i.e. by the
        network of the underlying :py:class:`ActivtionTool`.
        """
        return (None if self._activationTool is None
                else self._activationTool.network)

    def setNetwork(self, network: Network) -> None:
        """Set the network to be used by this
        :py:class:`ActivationsPanel`, i.e. by the underlying
        :py:class:`ActivtionTool`. If no activation tool is
        set, setting the network will have no effect (and it will
        not be remembered in case an tool will be set in the future).
        """
        self._networkPrepareButton.setPreparable(network)
        if self._activationTool is not None:
            self._activationTool.network = network
        print(f"ActivationsPanel.setNetwork({network}): "
              f"{isinstance(network, ImageTool)}")
        self._imageView.setImageTool(network if isinstance(network, ImageTool)
                                     else None)

    def setDatasource(self, datasource: Datasource) -> None:
        """Set the datsource to be used for selecting input data.

        """
        self._datasourceNavigator.setDatasource(datasource)

    #
    # Toolbox interface
    #

    def toolbox_changed(self, toolbox: Toolbox, change: Toolbox.Change):
        # pylint: disable=invalid-name
        """React to changes of the toolbox. The :py:class:`ActivationsPanel`
        will reflect two types of changes:
        (1) `input_change`: new input data for the toolbox will be used
            as input for the underlying :py:class:`ActivationWorker`, and
        (2) `datasource_changed`: the toolbox datasource will be used
            for selecting inputs by the datasource navigator of them
            :py:class:`ActivationsPanel`.
        """
        LOG.debug("ActivationsPanel.toolbox_changed: %s", change)
        if change.input_changed:
            data = toolbox.input_data if toolbox is not None else None
            self._imageView.setData(data)
            self._dataView.setData(data)
            if self._activationWorker is not None:
                self._activationWorker.work(data)
        elif change.datasource_changed:
            self.setDatasource(toolbox.datasource)

    #
    # ActivationWorker
    #

    def worker_changed(self, worker: ActivationWorker,
                       info: ActivationWorker.Change) -> None:
        # pylint: disable=invalid-name
        """React to changes of the underlying :py:class:`ActivationWorker`.

        The QClassesView is only interested if the classification result
        changes.
        """
        LOG.debug("QActivationsPanel.worker_changed: %s", info)
        if info.tool_changed:
            network = worker.tool.network

            self._networkSelector.setNetwork(network)
            self._networkPrepareButton.setPreparable(network)
            self._layerSelector.setNetwork(network)

            enabled = (network is not None and network.prepared and
                       network.is_classifier())
            self._classesViewBox.setEnabled(enabled)
        if info.work_finished:
            self.updateImageMask()

    #
    # update
    #

    def updateImageMask(self) -> None:
        """Display the current activations as mask in the image view.
        """
        if self._activationWorker is None or self._unit is None:
            self._imageView.setMask(None)
            return

        if self._layerID is None:
            self._imageView.setMask(None)
            return

        activations = self._activationWorker.activations(self._layerID,
                                                         unit=self._unit)

        # For convolutional layers add a activation mask on top of the
        # image, if a unit is selected
        if activations is not None and activations.ndim > 1:
            # exclude dense layers
            activationMask = grayscaleNormalized(activations)
            self._imageView.setMask(activationMask)
            # field = engine.receptive_field
            # if field is not None:
            #    self.addMark(QRect(field[0][1], field[0][0],
            #                       field[1][1]-field[0][1],
            #                       field[1][0]-field[0][0]))
        else:
            self._imageView.setMask(None)

    def updateReceptiveField(self) -> None:
        """Show the current receptive field in the image view.
        """
        # field: coordinates of the receptive field in coordinates
        # of the network input layer in format ((x1,y1), (x2,y2))
        field = self._receptiveField
        network = self.network()
        print(f"AcivationsPanel: receptive field={field}, network: {network.get_input_shape(False, False)}")
        if field is None or network is None:
            self._imageView.setReceptiveField(None, None)
        else:
            rect = QRect(QPoint(*field[0]), QPoint(*field[1]))
            reference = QSize(*network.get_input_shape(False, False))
            self._imageView.setReceptiveField(rect, reference)

    #
    # Slots
    #

    @protect
    def onNetworkSelected(self, network: Network) -> None:
        """The signal `networkSelected` is sent whenever the currentIndex
        in the combobox changes either through user interaction or
        programmatically.
        """
        LOG.info("QActivationsPanel.onNetworkSelected(%s)", network)
        self.setNetwork(network)

    @protect
    def onUnitChanged(self, unit: int) -> None:
        """React to the (de)selection of a unit in the activation map.
        """
        self._unit = None if unit == -1 else unit
        self.updateImageMask()

    @protect
    def onPositionChanged(self, position: QPoint) -> None:
        """React to the (de)selection of a position in the activation map.
        """
        layer = self._activationView.layer()
        if layer is None or position == QPoint(-1, -1):
            self._receptiveField = None
        else:
            point = (position.x(), position.y())
            self._receptiveField = layer.receptive_field(point)
        print(f"AcivationsPanel: New receptive field is: {self._receptiveField}")
        self.updateReceptiveField()

    @protect
    def onLayerSelected(self, layer_id: str, selected: bool) -> None:
        """A slot for reacting to layer selection signals.

        Parameters
        ----------
        layer_id: str
            The id of the network :py:class:`Layer`. It is assumed
            that this id refers to a layer in the current
            `py:meth:`ActivationTool.network`
            of the :py:class:`ActivationTool`.
        selected: bool
            A flag indicating if the layer was selected (`True`)
            or deselected (`False`).
        """
        LOG.info("QActivationsPanel.onLayerSelected('%s', %s)",
                 layer_id, selected)
        if self._activationTool is None:
            return  # we will not do anything without an ActivationTool

        self._layerID = layer_id if selected else None
        layer = self.network()[layer_id] if selected else None
        self._activationView.setLayer(layer)
