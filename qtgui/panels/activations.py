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
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QSplitter

# toolbox imports
from toolbox import Toolbox
from network import Network
from datasource import Datasource
from tools.activation import Engine as ActivationEngine
import util.image

# GUI imports
from .panel import Panel
from ..utils import QObserver, protect
from ..widgets.activationview import QActivationView
from ..widgets.data import QDataView
from ..widgets.image import QImageView
from ..widgets.network import QLayerSelector, QNetworkSelector
from ..widgets.classesview import QClassesView
from ..widgets.datasource import QDatasourceNavigator
from ..widgets.register import QPrepareButton

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
        ActivationEngine: {'network_changed'}}):
    # pylint: disable=too-many-instance-attributes
    """The :py:class:`ActivationsPanel` is basically a graphical frontend
    for an :py:class:`ActivationEngine`. It allows to set the network
    for that engine, it feeds input to the engine and displays
    activation maps computed by the the engine.

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
    to the underlying :py:class:`tools.activation.Engine`:

    _activation: ActivationEngine
        The :py:class:`ActivationsPanel` will observe this engine
        to reflect the network currently used by the engine and
        it also acts as a controller allowing to change that network.
        This engine is also propagated to the _activationView.

    In addition, the :py:class:`ActivationsPanel` also holds the
    following protected attributes based on the current state of the
    underlying activation engine:

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

    _networkSelector: QNetworkSelector
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
                 activation: ActivationEngine = None,
                 **kwargs) -> None:
        """Initialization of the ActivationsPael.

        Parameters
        ----------
        toolbox: Toolbox
        network: Network
        parent: QWidget
            The parent argument is sent to the QWidget constructor.
        """
        self._activation = None
        self._layerID = None
        self._unit = None
        self._receptiveField = None
        LOG.info("ActivationsPanel(toolbox=%s, network=%s, "
                 "datasource=%s, activation=%s)",
                 toolbox, network, datasource, activation)
        super().__init__(**kwargs)
        self._initUI()
        self._layoutUI()
        self.setToolbox(toolbox)
        self.setNetwork(network)
        self.setActivationEngine(activation)
        self.setDatasource(datasource)

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
        self._activationView = QActivationView(engine=self._activation)
        self._activationView.unitChanged.connect(self.onUnitChanged)
        self._activationView.positionChanged.connect(self.onPositionChanged)

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
        self._imageView = QImageView()
        self.addAttributePropagation(Toolbox, self._imageView)

        # QDataView: display data related to the current input data
        self._dataView = QDataView()
        self._dataView.addAttribute('filename')
        self._dataView.addAttribute('basename')
        self._dataView.addAttribute('directory')
        self._dataView.addAttribute('path')
        self._dataView.addAttribute('regions')
        self._dataView.addAttribute('image')

        # QDatasourceNavigator: navigate through the datasource
        self._datasourceNavigator = QDatasourceNavigator()
        self.addAttributePropagation(Toolbox, self._datasourceNavigator)

        #
        # (C) Network
        #

        # QNetworkSelector: a widget to select a network
        self._networkSelector = QNetworkSelector()
        self.addAttributePropagation(Toolbox, self._networkSelector)
        # FIXME[note]: the 'activated' signal is emitted on every
        # activation, not just on changes - we use this during development
        # as it may help to trigger the event on purpose.  However, in future
        # 'activated' may be replaced by 'currentIndexChanged' which
        # will only be emitted if the selected network actually changed.
        # self._networkSelector.activated[str].\
        #    connect(self.onNetworkSelectorChanged)
        self._networkSelector.currentIndexChanged[str].\
            connect(self.onNetworkSelectorChanged)

        # QPrepareButton: a button to (un)prepare the network
        self._networkPrepareButton = QPrepareButton()

        # QLayerSelector: a widget to select a network layer
        self._layerSelector = QLayerSelector()
        self._layerSelector.layerClicked.connect(self.onLayerSelected)

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

        # FIXME[layout]
        # keep image view square (TODO: does this make sense for every input?)
        self._imageView.heightForWidth = lambda w: w
        self._imageView.hasHeightForWidth = lambda: True
        # FIXME[hack]
        self._imageView.setMaximumSize(500, 500)

        # FIXME[layout]
        inputLayout.setSpacing(0)
        inputLayout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(self._imageView)
        row.addStretch()
        inputLayout.addLayout(row)
        #inputLayout.addWidget(self._dataView)
        inputLayout.addWidget(self._datasourceNavigator)
        inputLayout.addStretch()

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
        layout = QHBoxLayout()
        layout.addWidget(splitter)
        layout.addLayout(rightLayout)
        self.setLayout(layout)

    def setActivationEngine(self, activation: ActivationEngine) -> None:
        """Set the underlying :py:class:`ActivationEngine`.
        """
        LOG.debug("ActivationsPanel.setActivationEngine(%s)", activation)
        interests = ActivationEngine.Change('network_changed')
        self._exchangeView('_activation', activation, interests=interests)
        self._layerSelector.setActivationEngine(activation)
        self._imageView.setActivationEngine(activation)
        # self._inputInfoBox.setActivationEngine(activation)
        self._activationView.setActivationEngine(activation)
        self._classesView.setActivationEngine(activation)
        if activation is not None:
            self._networkSelector.setNetwork(activation.network)

    def network(self) -> Network:
        """The network used by this :py:class:`ActivationsPanel`, i.e. by the
        network of the underlying :py:class:`ActivtionEngine`.
        """
        return None if self._activation is None else self._activation.network

    def setNetwork(self, network: Network) -> None:
        """Set the network to be used by this
        :py:class:`ActivationsPanel`, i.e. by the underlying
        :py:class:`ActivtionEngine`. If no activation engine is
        set, setting the network will have no effect (and it will
        not be remembered in case an engine will be set in the future).
        """
        if self._activation is not None:
            self._activation.network = network

    def setDatasource(self, datasource: Datasource) -> None:
        """Set the datsource to be used for selecting input data.

        """
        self._datasourceNavigator.setDatasource(datasource)

    def toolbox_changed(self, toolbox: Toolbox, change: Toolbox.Change):
        # pylint: disable=invalid-name
        """React to changes of the toolbox. The :py:class:`ActivationsPanel`
        will reflect two types of changes:
        (1) `input_change`: new input data for the toolbox will be used
            as input for the underlying :py:class:`ActivationEngine`, and
        (2) `datasource_changed`: the toolbox datasource will be used
            for selecting inputs by the datasource navigator of them
            :py:class:`ActivationsPanel`.
        """
        LOG.debug("ActivationsPanel.toolbox_changed: %s", change)
        if change.input_changed:
            data = toolbox.input_data if toolbox is not None else None
            self._imageView.setData(data)
            self._dataView.setData(data)
        elif change.datasource_changed:
            self.setDatasource(toolbox.datasource)

    def activation_changed(self, activation: ActivationEngine,
                           info: ActivationEngine.Change) -> None:
        # pylint: disable=invalid-name
        """React to changes of the underlying :py:class:`ActivationEngine`.

        The QClassesView is only interested if the classification result
        changes.
        """
        LOG.debug("QActivationsPanel.activation_changed: %s", info)
        if info.network_changed:
            network = activation.network

            self._networkSelector.setNetwork(network)
            self._networkPrepareButton.setPreparable(network)
            self._layerSelector.setNetwork(network)

            enabled = (network is not None and network.prepared and
                       network.is_classifier())
            self._classesViewBox.setEnabled(enabled)

    @protect
    def onNetworkSelectorChanged(self, key: str) -> None:
        """The signal `currentIndexChanged` is sent whenever the currentIndex
        in the combobox changes either through user interaction or
        programmatically.
        """
        #network = self._networkSelector.currentNetwork()
        network = None if not key else Network[key]
        LOG.info("QActivationsPanel.onNetworkSelectorChanged('%s', %s)",
                 key, network)
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
        self.updateReceptiveField()

    def updateImageMask(self) -> None:
        """Display the current activations as mask in the image view.
        """
        if self._activationEngine is None or self._unit is None:
            self._imageView.setMask(None)
            return

        if self._layerID is None:
            self._imageView.setMask(None)
            return

        activation = self._activationEngine.get_activation(self._layerID,
                                                           unit=self._unit)

        # For convolutional layers add a activation mask on top of the
        # image, if a unit is selected
        if activation is not None and activation.ndim > 1:
            # exclude dense layers
            activationMask = util.image.grayscaleNormalized(activation)
            self._imageView.setMask(activationMask)
            #field = engine.receptive_field
            #if field is not None:
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
        if field is None or network is None:
            self._imageView.setReceptiveField(None, None)
        else:
            rect = QRect(QPoint(*field[0]), QPoint(*field[1]))
            reference = QSize(*network.get_input_shape(False, False))
            self._imageView.setReceptiveField(rect, reference)

    @protect
    def onLayerSelected(self, layer_id: str, selected: bool) -> None:
        """A slot for reacting to layer selection signals.

        Parameters
        ----------
        layer_id: str
            The id of the network :py:class:`Layer`. It is assumed
            that this id refers to a layer in the current
            `py:meth:`ActivationEngine.network`
            of the :py:class:`ActivationEngine`.
        selected: bool
            A flag indicating if the layer was selected (`True`)
            or deselected (`False`).
        """
        LOG.info("QActivationsPanel.onLayerSelected('%s', %s)",
                 layer_id, selected)
        if self._activationEngine is None:
            return  # we will not do anything without an ActivationEngine

        self._layerID = layer_id if selected else None
        layer = self._activationEngine.network[layer_id] if selected else None
        self._activationView.setLayer(layer)
