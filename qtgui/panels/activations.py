"""
File: activations.py
Author: Petr Byvshev, Ulf Krumnack, Rasmus Diederichsen
Email: rdiederichse@uni-osnabrueck.de
Github: https://github.com/themightyoarfish
"""

# Generic imports
from typing import Optional
import logging

# Qt imports
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
from PyQt5.QtWidgets import QWidget, QGroupBox, QSplitter
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout

# toolbox imports
from dltb.base.data import Data, Datalike
from dltb.tool.activation import ActivationTool, ActivationWorker
from dltb.tool.image import ImageTool
from dltb.network import Network
from dltb.datasource import Datasource, Datafetcher
from dltb.util.array import DATA_FORMAT_CHANNELS_FIRST
from dltb.util.image import grayscaleNormalized

from toolbox import Toolbox

# GUI imports
from .panel import Panel
from ..utils import QObserver, QPrepareButton, protect
from ..widgets.activationview import QActivationView
from ..widgets.data import QDataView
from ..widgets.network import QLayerSelector, QNetworkComboBox
from ..widgets.network import QNetworkComboBox
from ..widgets.classesview import QClassesView
from ..widgets.datasource import QDatasourceNavigator


# logging
LOG = logging.getLogger(__name__)

# FIXME[todo]: show some busy widget for operations that may take some time:
# - initializing/perparing a network
# - computing activations (especially for the first run)


class ActivationsPanel(Panel, QObserver, qobservables={
        Toolbox: {'datasource_changed', 'input_changed'},
        ActivationWorker: {'tool_changed', 'data_changed', 'worker_changed',
                           'work_finished'},
        Datafetcher: {'data_changed'}}, qattributes={
            Datasource: False, Network: False}):
    # pylint: disable=too-many-instance-attributes
    """The :py:class:`ActivationsPanel` is a graphical frontend for an
    :py:class:`ActivationTool` and the an associated
    :py:class:`ActivationWorker`. It allows to set the network for
    that tool, it feeds input to the associated
    :py:class:`ActivationWorker` and displays activation maps computed
    by the tool.

    The :py:class:`ActivationsPanel` is a complex panel consisting
    of three parts:
    (A) activation: display of layer and unit activations
    (B1) input: select inputs
    (B2) input: display inputs
    (C) network: display network information and select network layers

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

    _layer: str
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

    _networkBox: QNetworkBox
        An box to display some details on the current network and the
        selected Layer.

    _classesView: QClassesView = None
        A :py:class:`QClassesView` for displaying classification results
        in case the current :py:class:`Network` is a
        :py:class:`Classifier`.

    Stand-alone and Toolbox mode
    ----------------------------

    The :py:class:`ActivationsPanel` can be used with a
    :py:class:`Toolbox`, in which case the selection of the network
    and input data are performed via the toolbox.
    - the toolbox network selector is used for selecting a Network
    - datasource selection affects the toolbox datasource
    - datasource navigation controls the toolbox Datafetcher
      (which in turn sets the toolbox input data)
    - the toolbox input data is used as input for the activation worker

    If no toolbox is used, network and datasource selection are done
    locally in the panel. The panel will create a private `Datafetcher`
    and `Data` obtained from that fetcher are used as input.

    In both cases, the properties can be get and set via the specific
    methods: `network`, `datasource`.

    network:
        the :py:class:`Network` is propagated to the
        :py:class:`QLayerSelector `and the :py:class:` and
        the :py:class:`QClassesView`
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
        self._layer = None
        self._unit = None
        self._receptiveField = None
        LOG.info("QActivationsPanel(toolbox=%s, network=%s, datasource=%s)",
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
        self._classesViewBox.toggled.connect(self.onClassesViewBoxToggled)
        classesViewLayout = QVBoxLayout()
        classesViewLayout.addWidget(self._classesView)
        self._classesViewBox.setLayout(classesViewLayout)
        self.addAttributePropagation(ActivationWorker, self._classesView)

        #
        # (B) Input
        #

        # QDataView: a widget to display the input data
        self._dataView = QDataView()
        self.addAttributePropagation(Toolbox, self._dataView)
        self.addAttributePropagation(Datafetcher, self._dataView)

        # display data related to the current input data
        self._dataView.addAttribute('filename')
        self._dataView.addAttribute('basename')
        self._dataView.addAttribute('directory')
        self._dataView.addAttribute('path')
        self._dataView.addAttribute('regions')
        self._dataView.addAttribute('image')

        # store reference to the QImageView in the QDataView
        self._imageView = self._dataView.imageView()

        #
        # (B2) Datasource navigation
        #
        self._datasourceNavigator = \
            QDatasourceNavigator(datasource_selector=True, style='wide')
        self.addAttributePropagation(Toolbox, self._datasourceNavigator)
        self.setDatafetcher(self._datasourceNavigator.datafetcher())
        LOG.info("QActivationsPanel.initGUI: datafetcher=%s",
                 self.datafetcher())

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

        self._networkBox = QNetworkBox()
        self.addAttributePropagation(Network, self._networkBox)

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
        inputLayout.addWidget(self._dataView)

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
        networkLayout.addWidget(self._networkBox)
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

        leftLayout = QVBoxLayout()
        leftLayout.addWidget(splitter)
        leftLayout.addWidget(self._datasourceNavigator)

        leftWidget = QWidget()
        leftWidget.setLayout(leftLayout)
        leftLayout.setContentsMargins(0, 0, 0, 0)

        rightWidget = QWidget()
        rightWidget.setLayout(rightLayout)
        rightLayout.setContentsMargins(0, 0, 0, 0)

        splitter2 = QSplitter(Qt.Horizontal)
        splitter2.addWidget(leftWidget)
        splitter2.addWidget(rightWidget)

        layout = QHBoxLayout()
        layout.addWidget(splitter2)
        self.setLayout(layout)

    def setActivationWorker(self, worker: ActivationWorker) -> None:
        """Set the underlying :py:class:`ActivationWorker`.
        """
        LOG.debug("QActivationsPanel.setActivationWorker(%s)", worker)
        self._classesViewBox.setChecked(worker.classification)

    def setToolbox(self, toolbox: Toolbox) -> None:
        """Set a :py:class:`Toolbox` to be used by this `QActivationsPanel`.

        Arguments
        ---------
        toolbox:
            The :py:class:`Toolbox` object to be used. If `None`,
            toolbox functionality will be disabled.
        """
        LOG.debug("QActivationsPanel.setToolbox(%s)", toolbox)
        self.setData(None if toolbox is None else toolbox.input_data)

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
        LOG.debug("QActivationsPanel.setNetwork(%s): ImageTool: %s",
                  network, isinstance(network, ImageTool))
        self._networkPrepareButton.setPreparable(network)
        if self._activationTool is not None:
            self._activationTool.network = network
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
        LOG.debug("QActivationsPanel.toolbox_changed: %s", change)
        if change.input_changed:
            data = toolbox.input_data if toolbox is not None else None
            self.setData(data)
        elif change.datasource_changed:
            self.setDatasource(toolbox.datasource)

    #
    # Datafetcher
    #

    # FIXME[todo/concept]: this is not yet implemented, and it is not
    # clear if that is really needed.
    def datafetcher_changed(self, datafetcher: Datafetcher,
                            info: Datafetcher.Change) -> None:
        # pylint: disable=invalid-name
        """React to a change in the state of the controlled
        :py:class:`Datafetcher`.
        """
        LOG.debug("QActivationsPanel: Datafetcher %s changed %s",
                  datafetcher, info)
        if info.data_changed:
            self.setData(datafetcher.data)

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
            self._classesViewBox.setEnabled(worker.ready and
                                            network.is_classifier())
            if worker.ready:
                worker.work(self.data())

        if info.data_changed:
            self._dataView.setData(worker.data)

        if info.worker_changed:
            self._classesViewBox.setChecked(worker.classification)

        if info.work_finished:
            self.updateImageMask()

    #
    # update
    #

    def data(self) -> Optional[Data]:
        """The :py:class:`Data` currently used by this `QActivationView`.
        """
        # if self._toolbox is not None:
        #     return self._toolbox.input_data

        # if self._dataView is not None:
        #     return self._dataView.data()

        worker = self._activationWorker
        return None if worker is None else worker.data

    def setData(self, data: Datalike) -> None:
        """Update the data to be displayed in the `ActivationsPanel`.
        """
        worker = self._activationWorker
        if worker is not None:
            LOG.debug("QActivationPanel: setData(%s) - worker "
                      "ready/busy: %s/%s", data, worker.ready, worker.busy)
            worker.work(data)
        else:
            LOG.debug("QActivationPanel: setData(%s) - no worker!", data)

    def updateImageMask(self) -> None:
        """Display the current activations as mask in the image view.
        """
        if self._activationWorker is None or self._unit is None:
            self._imageView.setMask(None)
            return

        if self._layer is None:
            self._imageView.setMask(None)
            return

        # For convolutional layers add a activation mask on top of the
        # image, if a unit is selected
        activations = self._activationWorker.activations(self._layer,
                                                         unit=self._unit)
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
        print(f"AcivationsPanel: receptive field={field}, "
              f"network: {network and network.get_input_shape(False, False)}")
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
        LOG.info("QActivationsPanel: position changed: "
                 "new receptive field is: %s", self._receptiveField)
        self.updateReceptiveField()

    @protect
    def onLayerSelected(self, layer: str, selected: bool) -> None:
        """A slot for reacting to layer selection signals.

        Parameters
        ----------
        layer: str
            The (key of) the network :py:class:`Layer`. It is assumed
            that this key refers to a layer in the current
            `py:meth:`ActivationTool.network`
            of the :py:class:`ActivationTool`.
        selected: bool
            A flag indicating if the layer was selected (`True`)
            or deselected (`False`).
        """
        LOG.info("QActivationsPanel.onLayerSelected('%s', %s) - tool: %s",
                 layer, selected, self._activationTool is not None)
        if self._activationTool is None:
            return  # we will not do anything without an ActivationTool

        self._layer = layer if selected else None
        layer = self.network()[layer] if selected else None
        self._activationView.setLayer(layer)
        self._networkBox.setLayer(layer)

    @protect
    def onClassesViewBoxToggled(self, on: bool) -> None:
        # parameter name "on" is prescribed by Qt API.
        # pylint: disable=invalid-name
        """The checkbox of the `QGroupBox` containining the `QClassesView` has
        been toggled.  The underlying :py:class:`ActivationWorker` is
        informed whether class information are still desired.

        Arguments
        ---------
        on:
             If `True`, class predictions are to be displayed, if `False`
             that functionality is disabled.
        """
        activationWorker = self._activationWorker
        if activationWorker is not None:
            activationWorker.classification = on
