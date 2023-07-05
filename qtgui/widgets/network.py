"""
QNetworkListWidget: a list of networks

QNetworkInternals: details of a network
"""

# standard imports
from typing import Optional, Iterable
import logging

# Qt imports
from PyQt5.QtCore import Qt, QCoreApplication, QEvent, pyqtSignal
from PyQt5.QtGui import QPaintEvent, QPalette
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QPushButton, QLabel
from PyQt5.QtWidgets import QVBoxLayout

# toolbox imports
from toolbox import Toolbox
from dltb.network import Network, Networklike, as_network, Layer, Layerlike
from dltb.network import StridingLayer, Dense, Conv2D, MaxPooling2D
from dltb.network import Dropout, Flatten

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
    _network: Optional[Network] = None
    _layer: Optional[Layer] = None

    _networkText: str = ''
    _layerText: str = ''

    def __init__(self, network: Network = None, **kwargs) -> None:
        """Create a new :py:class:``QNetworkInfoBox``"""
        super().__init__(**kwargs)
        self.setWordWrap(True)
        self.setNetwork(network)

    def network_changed(self, network: Network,
                        _change: Network.Change) -> None:
        # pylint: disable=invalid-name
        """React to a change of the observed :py:class:`Network`.
        """
        self._update()
        # QMetaObject.invokeMethod(self, "setText", Q_ARG(str, text))

    def setNetwork(self, network: Optional[Networklike]) -> None:
        """Set the network currently displayed in the
        :py:class:`QNetworkBox`.

        """
        theNetwork = as_network(network)
        if theNetwork is self._network:
            return  # nothing to do

        super().setNetwork(theNetwork)
        self.setLayer(None)

    def setLayer(self, layer: Optional[Layerlike]) -> None:
        """Set the current layer to be displayed in this 
        :py:class:`QNetworkBox`.
        """
        try:
            if layer is None or self._network is None:
                theLayer = None
            else:
                theLayer = self._network[layer]
        except KeyError as error:
            theLayer = None
            raise error
        finally:
            if theLayer is not self._layer:
                self._layer = theLayer
                self._update()

    def _update(self) -> None:
        """
        """
        #
        # Set Network info text
        #
        networkText = ""
        # networkText += '<b>Network info:</b> '
        if self._network is not None:
            # network_name = type(self._network).__name__
            # networkText += 'FIXME[todo]: obtain network information ...'
            networkText += ('<br>\n<b>class:</b> '
                            f'{self._network.__class__.__name__}')
            networkText += f'<br>\n<b>name:</b> {self._network}'
            # if not self._network.empty():
            #    networkText += ('<br>\n<b>input layer:</b> '
            #                     f'{self._network.input_layer_id()}')
            #    networkText += ('<br>\n<b>output layer:</b> '
            #                     f'{self._network.output_layer_id()}')
        else:
            networkText += "No network"
        self._networkText = networkText

        #
        # Layer
        #
        layerText = "<b>Layer Info</b><br>"
        if self._layer is not None:
            layerText += self._createLayerText(self._layer)
        else:
            layerText += "No Layer <br><br><br>"
        self._layerText = layerText

        #
        # Set the info text
        #
        text = ('<b>Network Info Box</b><br>' +
                self._networkText + '<br>' + self._layerText)
        self.setText(text)
        super().update()

    def _createLayerText(self, layer: Layer) -> str:
        text = f"Type: {type(layer).__name__}<br>"
        if isinstance(Layer, StridingLayer):
            text += (f"Striding with size {layer.strides}<br>"
                     f"and padding {layer.padding}")
        if isinstance(Layer, Dense):
            text += "Dense layer"
        if isinstance(Layer, Conv2D):
            text += ("2D Convolutional layer with<br>"
                     f"{layer.filters}filters of size {layer.filter_size}")
        return text
        
    def paintEvent(self, event: QPaintEvent):
        """Process a QPaintEvent
        """
        # text = ('<b>Network Info Box</b><br>' +
        #        self._networkText + '<br>' + self._layerText)
        # self.setText("A" + text + "B")
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




# ########################################################################## #
#                                                                            #
#                                 QLayerSelector                             #
#                                                                            #
# ########################################################################## #


class QLayerSelector(QWidget, QObserver, qobservables={
        Network: {'state_changed'}}):
    """A simple widget to select :py:class:`Layer`s in a :py:class:`Network`. 

    The Layers are presented as a stack of buttons. Clicking at a
    button will emit the `layerClicked` signal.

    The :py:class:`QLayerSelector` observes a py:class:`Network`. If
    the network is changed, the new network architecture needs to be
    displayed.

    Attributes
    ----------
    _network: Network
        The network currently shown by this :py:class:`QLayerSelector`
        (inherited from `Network` QObserver).

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

    _currentSelected: int
        Layer id of the currently selected layer.

    Signals
    -------
    layerClicked(layer_id: str, selected: bool):
        Emitted when a Layer was clicked. The boolean status indicates
        if the Layer has been selected (`True`) or deselected (`False`)

    """

    layerClicked = pyqtSignal(str, bool)

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

        self._initUI()
        self._layoutUI()

        self._network = None
        self.setNetwork(network)

    def _initUI(self) -> None:
        """Initialize the user interface.
        """
        self.setMinimumWidth(300)

    def _layoutUI(self) -> None:
        """Layout the user interface.
        """
        self._layerLayout = QVBoxLayout()

        layout = QVBoxLayout()
        layout.addLayout(self._layerLayout)
        layout.addStretch(1)
        self.setLayout(layout)

    def _asLayer(self, layer: Layerlike) -> Layer:
        return layer if isinstance(layer, Layer) else self.network()[layer]

    def _asLayerId(self, layer: Layerlike) -> str:
        return layer.key if isinstance(layer, Layer) else layer

    def selectLayer(self, layer: Optional[Layerlike],
                    selected: bool = True) -> None:
        """(De)select a layer in this :py:class:`QLayerSelector`.

        Arguments
        ---------
        layer:
            the :py:class:`Layer` to be selected :py:class:`Layer`.
            `None` if no layer is selected.
        """
        if layer is not None:
            self._markButton(self._button(layer), selected)
        elif self._exclusive is False:
            self.selectLayers(())
        elif self._exclusive is not True:
            self._markButton(self._exclusive, False)
            self._exclusive = True

    def selectLayers(self, layers: Iterable[Layerlike]) -> None:
        """Set the currently selected :py:class:`Layer`.
        """
        if exclusive is not False:
            return RuntimeError("Cannot select multiple Layers in a "
                                "non-exclusive QLayerSelector.")
        state = { layerId: False for layerId in self._layerButtons }
        for layer in layers:
            state[self._asLayerId(layer)] = True

        for layerId, selected in state.items():
            self._markButton(self._layerButtons[button], selected)

    def layerSelected(self, layer: Layerlike) -> bool:
        """Check if the given :py:class:`Layer` is selected.
        """
        return self._buttonIsMarked(self._button(layer))

    def layer(self) -> Optional[Layer]:
        """The currently selected :py:class:`Layer`. `None` if no
        layer is selected.
        """
        exclusive = self._exclusive
        if exclusive is False:
            return RuntimeError("Cannot obtain Layer for an exclusive "
                                "QLayerSelector.")

        if exclusive is True:
            return None  # no layer selected

        return self._layerForLayerId(excusive)

    def layers(self) -> Iterable[Layer]:
        """The currently selected :py:class:`Layer`. `None` if no
        layer is selected.
        """
        for button in self._layerButtons:
            if self._buttonIsMarked(button):
                yield self._layerForButton(button)

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
            # FIXME[hack]: layerId vs. layer.key
            #for layerId in network.layer_dict.keys():
            for layer in network.layer_dict.values():
                #button = QPushButton(layerId, self)
                button = QPushButton(layer.key, self)
                #self._layerButtons[layerId] = button
                self._layerButtons[layer.key] = button
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

    def _button(self, layer: Layerlike) -> QPushButton:
        """The button associated with a given layer.
        """
        layerId = self._asLayerId(layer)
        if layerId not in self._layerButtons:
            raise KeyError(f"'{layerId}' is not a valid layer id:"
                           f"{list(self._layerButtons.keys())}")
        return self._layerButtons[layerId]

    def _layerIdForButton(self, button: QPushButton) -> Layer:
        """Get the layer-ID associated with a button.
        """
        return button.text()

    def _layerForButton(self, button: QPushButton) -> Layer:
        """Get the :py:class:`Layer` associated with a button.
        """
        return self.network()[self._layerIdForButton(button)]

    @protect
    def onLayerButtonClicked(self, checked: bool):
        """Callback for clicking one of the layer buttons.
        If we have an ActivationEngine set, we will set that layer in
        the engine.
        """
        button = self.sender()
        LOG.info("QLayerSelector.onLayerButtonClicked(checked=%s): "
                 "sender=%s", checked, button.text())

        if self._network is None:
            # should not happen: there should be no buttons to click
            # if we do not have a Network
            return  # just ignore it

        layer = self._layerForButton(button)
        select = not self.layerSelected(layer)
        self.selectLayer(layer, select)

        if self._exclusive is not False:
            if self._exclusive is not True and self._exclusive is not layer:
                self.selectLayer(self._exclusive, False)
            if select:
                self._exclusive = layer

        self.layerClicked.emit(layer.key, select)

    #
    # Network
    #

    def setNetwork(self, network: Optional[Networklike]) -> None:
        self._initButtonsFromNetwork(network)
    
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
