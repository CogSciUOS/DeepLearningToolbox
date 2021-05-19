"""
QNetworkListWidget: a list of networks

QNetworkInternals: details of a network
"""

# standard imports
import logging

# Qt imports
from PyQt5.QtCore import Qt, QCoreApplication, QEvent, pyqtSignal
from PyQt5.QtGui import QPaintEvent, QPalette
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QPushButton, QLabel
from PyQt5.QtWidgets import QVBoxLayout

# toolbox imports
from toolbox import Toolbox
from dltb.network import Network, Layer

# GUI imports
from .register import QRegisterListWidget, QRegisterComboBox
from .tensorflow import QTensorflowInternals
from .register import ToolboxAdapter
from ..utils import QObserver, protect

# logging
LOG = logging.getLogger(__name__)


class NetworkAdapter(ToolboxAdapter, qobservables={
        Toolbox: {'network_changed', 'networks_changed'}}):
    # pylint: disable=abstract-method
    """A :py:class:`ToolboxAdapter` that is especially interested in
    :py:class:`Network`.

    Signals
    -------

    networkSelected(Network):
        The signal is sent whenever a new :py:class:`Network` is selected.
        The selected :py:class:`Network` may be `None`, indicating that
        currently no network is selected.
    """

    networkSelected = pyqtSignal(object)

    def __init__(self, **kwargs) -> None:
        """
        """
        super().__init__(register=Network.instance_register, **kwargs)

    def updateFromToolbox(self) -> None:
        """Update the list from the :py:class:`Toolbox`.
        """
        self.updateFromIterable(map(lambda network:
                                    self._register[network.key],
                                    self._toolbox.networks))

    def toolbox_changed(self, _toolbox: Toolbox,
                        change: Toolbox.Change) -> None:
        # pylint: disable=invalid-name
        """React to a change in the :py:class:`Toolbox`. The only change
        of interest is a change of the current network. This
        will be reflected in the list.
        """
        # if change.network_changed:  # the current network has changed
        #     self._formatAllItems()
        if change.networks_changed:  # the list of networks has changed
            self.updateFromToolbox()

    def network(self) -> Network:
        """The currently selected Network in this
        :py:class:`NetworkAdapter`.
        """
        # items are of type InstanceRegisterEntry
        item = self._currentItem()
        return None if item is None else item.obj

    def setNetwork(self, network: Network) -> None:
        """Set the current :py:class:`Network`.
        """
        if isinstance(network, str):
            network = self._register[network]
        elif isinstance(network, Network):
            network = self._register[network.key]
        self._setCurrentItem(network)

    @protect
    def _oncurrentIndexChanged(self, _index: int) -> None:
        """A forward to map item selection to Network selection.
        """
        self.networkSelected.emit(self.network())

    def debug(self) -> None:
        super().debug()
        print(f"debug: NetworkAdapter[{type(self).__name__}]:")
        print(f"debug:   * Current key: {type(self._currentItem().key)}, "
              f"type: {type(self._currentItem())}, "
              f"network: {type(self.network())}")


class QNetworkListWidget(NetworkAdapter, QRegisterListWidget):
    """A list displaying the :py:class:`Network`s of a
    :py:class:`Toolbox`.

    By providing a :py:class:`Toolbox`, the list becomes clickable,
    and selecting a Network from the list will set current network of
    the :py:class:`Toolbox`, and vice versa, i.e. changing the current
    network in the :py:class:`Toolbox` will change the current item in
    the list.

    """

    def __init__(self, **kwargs) -> None:
        """
        """
        super().__init__(**kwargs)

    def debug(self) -> None:
        super().debug()
        print(f"debug: QNetworkListWidget[{type(self).__name__}]:")
        print(f"debug:   * Current index: {self.currentRow()}")


class QNetworkComboBox(NetworkAdapter, QRegisterComboBox):
    """A widget to select a :py:class:`network.Network` from a list of
    :py:class:`network.Network`s.


    The class provides the common :py:class:`QComboBox` signals, including

    activated[str/int]:
        An item was selected (no matter if the current item changed)

    currentIndexChanged[str/int]:
        An new item was selected.

    """

    def __init__(self, **kwargs) -> None:
        """
        """
        super().__init__(**kwargs)
        self.currentIndexChanged.connect(self._oncurrentIndexChanged)

    def debug(self) -> None:
        super().debug()
        print(f"debug: QNetworkComboBox[{type(self).__name__}]:")
        print(f"debug:   * Current index: {self.currentIndex()}")


#
# QNetworkBox
#


class QNetworkBox(QLabel, QObserver, qobservables={
        # FIXME[hack]: check what we are really interested in ...
        Network: Network.Change.all()}):
    """
    .. class:: QNetworkBox

    A simple widget to display information on a network and a selected
    layer.
    """

    def __init__(self, network: Network = None, **kwargs) -> None:
        """Create a new :py:class:``QNetworkInfoBox``"""
        super().__init__(**kwargs)
        self.setWordWrap(True)
        self._networkText = ''
        self._layerText = ''
        self.setNetwork(network)

    def network_changed(self, network: Network,
                        _change: Network.Change) -> None:
        # pylint: disable=invalid-name
        """React to a change of the observed :py:class:`Network`.
        """

        #
        # Set Network info text
        #
        networkText = ''
        # networkText += '<b>Network info:</b> '
        if network is not None:
            # network_name = type(network).__name__
            # networkText += 'FIXME[todo]: obtain network information ...'
            networkText += ('<br>\n<b>class:</b> '
                            f'{network.__class__.__name__}')
            networkText += f'<br>\n<b>name:</b> {network}'
            # if not network.empty():
            #    networkText += ('<br>\n<b>input layer:</b> '
            #                     f'{network.input_layer_id()}')
            #    networkText += ('<br>\n<b>output layer:</b> '
            #                     f'{network.output_layer_id()}')
        else:
            networkText += "No network"
        self._networkText = networkText

        self.update()
        # QMetaObject.invokeMethod(self, "setText", Q_ARG(str, text))

    def paintEvent(self, event: QPaintEvent):
        """Process a QPaintEvent
        """
        # text = ('<b>Network Info Box</b><br>' +
        #        self._networkText + self._layerText)
        # self.setText(text)
        super().paintEvent(event)

#
# Network Internals
#

class QNetworkInternals(QWidget, QObserver, qobservables={
        Network: {'state_changed'}}):
    """A Widget to inspect the internals of a :py:class:`Network`.
    """

    def __init__(self, network: Network = None, **kwargs) -> None:
        """
        Arguments
        ---------
        toolbox: Toolbox
        network: Network
        parent: QWidget
        """
        super().__init__(**kwargs)
        self._initUI()
        self._layoutUI()
        self.setNetwork(network)

    def _initUI(self) -> None:
        """Initialize the user interface."""
        self._labelNetworkType = QLabel()
        self._ternsorflowInternals = QTensorflowInternals()

    def _layoutUI(self) -> None:
        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(self._labelNetworkType)
        layout.addWidget(self._ternsorflowInternals)

    def network_changed(self, network: Network,
                        change: Network.Change) -> None:
        # pylint: disable=invalid-name

        LOG.debug("QNetworkInternals.network_changed(%s, %s)", network, change)
        self.update(network)

    def setNetwork(self, network: Network) -> None:
        self.update(network)

    def update(self, network):
        self._labelNetworkType.setText(f"{network} "
                                       f"({type(network).__name__}), "
                                       f"prepared={network.prepared}, "
                                       f"graph={hasattr(network, '_graph')}"
                                       if network else "")
        # FIXME[hack]:
        if network and hasattr(network, '_graph'):
            self._ternsorflowInternals.setGraph(network._graph)




##############################################################################
#                                                                            #
#                                 QLayerSelector                             #
#                                                                            #
##############################################################################


class QLayerSelector(QWidget, QObserver, qobservables={
        Network: {'state_changed'}}):
    """A simple widget to display information on a network and select a
    layer.

    The Layers are presented as a stack of buttons. Clicking at a
    button will set the corresponding layer in the
    py:class:`ActivationEngine`.

    The :py:class:`QLayerSelector` observes a py:class:`Network` and an
    py:class:`ActivationEngine`. If the network is changed, the
    new network architecture needs to be displayed.

    Attributes
    ----------
    _network: Network
        The network currently shown by this :py:class:`QLayerSelector`
    _layerButtons: dict[str, QPushButton]
        A mapping from `layer_id`s to buttons for (un)selecting that
        layer.
    _layerLayout: QVBoxLayout
        A Layout in which the _layerButtons are arranged.
    _exclusive: Union[bool, Layer]
        This attribute indicates if the :py:class:`QLayerSelector`
        operates in exclusive mode. If `False` multiple values
        can be selected. Otherwise the attribute holds the currently
        selected :py:class:`Layer` or `True` if no layer is selected.

    FIXME[old]:
    _activation: ActivationEngine
        Controller for this UI element
    _currentSelected: int
        Layer id of the currently selected layer.

    Signals
    -------
    layerClicked(layer_id: str, selected: bool):
        Emitted when a Layer was clicked. The boolean status indicates
        if the Layer has been selected (`True`) or deselected (`False`)
    """

    layerClicked = pyqtSignal(str, bool)

    # FIXME[todo]: this should not be part of the QLayerSelector
    _networkInfo: QNetworkBox = None

    # FIXME[old]: is this still used? remove ...
    _UPDATE_NETWORK = QEvent.registerEventType()

    def __init__(self, network: Network = None, exclusive: bool = True,
                 **kwargs) -> None:
        """Initialization of the QLayerSelector.

        Parameters
        ----------
        parent: QWidget
            The parent argument is sent to the QWidget constructor.
        """
        super().__init__(**kwargs)
        self._currentSelected = None  # FIXME[old]
        self._exclusive = exclusive
        self._layerButtons = {}

        # FIXME[hack]: make a nicer solution and then remove this!
        self.labelInput = None
        self._labelOutput = None

        self.initUI()
        self.layoutUI()

        self._network = None
        self.setNetwork(network)

    def initUI(self) -> None:
        """Initialize the user interface.
        """
        self._networkInfo = QNetworkBox(parent=self)
        self._networkInfo.setMinimumWidth(300)

    def layoutUI(self) -> None:
        """Layout the user interface.
        """
        self._layerLayout = QVBoxLayout()

        info_layout = QVBoxLayout()
        info_layout.addWidget(self._networkInfo)

        layout = QVBoxLayout()
        layout.addLayout(self._layerLayout)
        layout.addLayout(info_layout)
        layout.addStretch(1)
        self.setLayout(layout)

    def selectLayer(self, layer: Layer, selected: bool = True) -> None:
        """(De)select a layer in this :py:class:`QLayerSelector`.
        """
        button = self._layerButtons[layer.id]
        self._markButton(button, selected)

    def layerSelected(self, layer: Layer) -> bool:
        """Check if the given :py:class:`Layer` is selected.
        """
        if layer is None:
            raise ValueError("None is not a valid layer.")
        if layer.id not in self._layerButtons:
            raise KeyError(f"'{layer.id}' is not a valid layer id:"
                           f"{list(self._layerButtons.keys())}")
        button = self._layerButtons[layer.id]
        return self._buttonIsMarked(button)

    #
    # Buttons
    #

    def _clearButtons(self) -> None:
        """Remove all widgets from this :py:class:`QLayerSelector`.
        """
        # FIXME[todo]: (still not sure what is the correct way to do:
        # widget.deleteLater(), widget.close(),
        # widget.setParent(None), or
        # layout.removeWidget(widget) + widget.setParent(None))
        for button in self._layerButtons.values():
            button.deleteLater()
        self._layerButtons = {}
        if self.labelInput is not None:
            self.labelInput.deleteLater()
            self.labelInput = None
        if self._labelOutput is not None:
            self._labelOutput.deleteLater()
            self._labelOutput = None
        self._currentSelected = None

    def _initButtonsFromNetwork(self, network: Network) -> None:
        """Update the buttons to reflect the layers of the
        underlying network.

        Arguments
        ---------
        network: Network
            The :py:class:`Network` from which this
            :py:class:`QLayerSelector` can select layers.
        """
        LOG.info("QLayerSelector._initButtonsFromNetwork(%s)", network)
        # remove the old network buttons
        self._clearButtons()

        layout = self._layerLayout
        if network is None:
            self.labelInput = QLabel("No network")
            layout.addWidget(self.labelInput)
        elif not network.prepared:
            self.labelInput = QLabel(f"Unprepared network '{network.key}'")
            layout.addWidget(self.labelInput)
        else:
            # add a label describing the network input
            self.labelInput = \
                QLabel(f"input = {self._network.get_input_shape(False)}")
            layout.addWidget(self.labelInput)

            # a column of buttons to select the network layer
            # FIXME[hack]: layerId vs. layer.id
            #for layerId in network.layer_dict.keys():
            for layer in network.layer_dict.values():
                #button = QPushButton(layerId, self)
                button = QPushButton(layer.id, self)
                #self._layerButtons[layerId] = button
                self._layerButtons[layer.id] = button
                layout.addWidget(button)
                button.resize(button.sizeHint())
                button.clicked.connect(self.onLayerButtonClicked)

            # add a label describing the network output
            self._labelOutput = \
                QLabel(f"output = {self._network.get_output_shape(False)}")
            layout.addWidget(self._labelOutput)

    def _markButton(self, button: QPushButton, state: bool = True) -> None:
        """Mark a button as selected.

        Arguments
        ---------
        state: a flag indicating if the button is selected or deselected.
        """
        if button is not None:
            button.setStyleSheet('background-color: red' if state else '')

    def _buttonIsMarked(self, button: QPushButton) -> bool:
        """Check if a button in marked.
        """
        return bool(button.styleSheet())

    @protect
    def onLayerButtonClicked(self, checked: bool):
        """Callback for clicking one of the layer buttons.
        If we have an ActivationEngine set, we will set that layer in
        the engine.
        """
        LOG.info("QLayerSelector.onLayerButtonClicked(checked=%s): "
                 "sender=%s, current_selected=%s",
                 checked, self.sender().text(), self._currentSelected)

        if self._network is None:
            return

        layer = self._network[self.sender().text()]
        select = not self.layerSelected(layer)
        self.selectLayer(layer, select)

        if self._exclusive is not False:
            if self._exclusive is not True and self._exclusive is not layer:
                self.selectLayer(self._exclusive, False)
            if select:
                self._exclusive = layer

        self.layerClicked.emit(layer.id, select)

    def selectLayers(self, activation, info) -> None:
        # FIXME[old]:
        # def activation_changed(self, activation: ActivationEngine,
        #                        info: ActivationEngine.Change) -> None:
        # pylint: disable=invalid-name
        """The QLayerSelector is interested in network related changes, i.e.
        changes of the network itself or the current layer.
        """
        LOG.debug("QLayerSelector.activation_changed(%s, %s)",
                  activation, info)

        # FIXME[design]: The 'network_changed' message can mean that
        # either the current model has changed or that the list of
        # models was altered (or both).
        # FIXME[old]: should be redundant with network.observable_changed
        if False and info.network_changed:
            #
            # Respond to network change
            #

            # As we may add/remove QWidgets, we need to make sure that
            # this method is executed in the main thread
            event = QEvent(QLayerSelector._UPDATE_NETWORK)
            event.activation = activation
            QCoreApplication.postEvent(self, event)

        return  # FIXME[old]

        if ((info.network_changed or info.layer_changed) and
                activation is not None):
            #
            # Respond to layer change
            #

            layerId = activation.layer_id
            try:
                layer_index = activation.idForLayer(layerId)
            except ValueError:
                print(f"ERROR: {self.__class__.__name__}."
                      "activation_changed(): {error}")
                layer_index = None
            if layer_index != self._currentSelected:

                # unmark the previously active layer
                if self._currentSelected is not None:
                    oldItem = self._layerButtons[self._currentSelected]
                    self._markButton(oldItem, False)

                self._currentSelected = layer_index
                if self._currentSelected is not None:
                    # and mark the new layer
                    newItem = self._layerButtons[self._currentSelected]
                    self._markButton(newItem, True)

    #
    # Network
    #

    def setNetwork(self, network: Network) -> None:
        self._networkInfo.setNetwork(network)

    def network_changed(self, network: Network,
                        change: Network.Change) -> None:
        # pylint: disable=invalid-name
        """React to changes of the network.
        """
        if change.state_changed:
            # we are only interested in changes of network.prepared
            if bool(self._layerButtons) != network.prepared:
                self._initButtonsFromNetwork(network)

    # FIXME[old]
    def event(self, event: QEvent) -> bool:
        """React to the UPDATE_NETWORK event.
        """
        if event.type() == QLayerSelector._UPDATE_NETWORK:
            self._initButtonsFromNetwork(event.activation._network)
            return True
        return super().event(event)
