"""
QNetworkList: a list of networks

QNetworkInternals: details of a network
"""

# standard imports
from typing import Iterator
import logging

# Qt imports
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent, QPalette
from PyQt5.QtWidgets import QListWidgetItem

# toolbox imports
from toolbox import Toolbox
from network import Network, Layer

# GUI imports
from .register import QRegisterList
from ..utils import QObserver, protect

# logging
LOG = logging.getLogger(__name__)


class QNetworkList(QRegisterList, qobservables={
        Toolbox: {'networks_changed'}}):
    """A list displaying the :py:class:`Network`s of a
    :py:class:`Toolbox`.

    By providing a :py:class:`Toolbox`, the list becomes clickable,
    and selecting a Network from the list will set current network of
    the :py:class:`Toolbox`, and vice versa, i.e. changing the current
    network in the :py:class:`Toolbox` will change the current item in
    the list.

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
        self.setNetwork(network)

    def _toolboxIterator(self, toolbox: Toolbox) -> Iterator:
        return (((str(data), data) for data in self._toolbox.networks)
                if self._toolbox else super()._toolboxIterator())

    def setNetwork(self, network: Network = None) -> None:
        self.setEnabled(network is not None)

    def currentNetwork(self) -> Network:
        item = self.currentItem()
        return item and item.data(Qt.UserRole) or None

    def network_changed(self, network: Network,
                        change: Network.Change) -> None:
        LOG.debug("QNetworkList.network_changed(%s, %s)", network, change)

    @protect
    def onItemClicked(self, item: QListWidgetItem):
        self._network(item.data(Qt.UserRole))


from .tensorflow import QTensorflowInternals
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout


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


###############################################################################
###                           QNetworkBox                                   ###
###############################################################################

from network import Network
from tools.activation import Engine as ActivationEngine

from ..utils import QObserver, protect

from PyQt5.QtCore import QMetaObject, Q_ARG
from PyQt5.QtWidgets import QLabel


class QNetworkBox(QLabel, QObserver, qobservables={
        # FIXME[hack]: check what we are really interested in ...
        Network: Network.Change.all(),
        ActivationEngine: ActivationEngine.Change.all()}):
    """
    .. class:: QNetworkBox

    A simple widget to display information on a network and a selected
    layer.
    """

    def __init__(self, network: Network = None, **kwargs) -> None:
        """Create a new :py:class:``QNetworkInfoBox``"""
        super().__init__(**kwargs)
        self.setWordWrap(True)
        self._network_text = ''
        self._layer_text = ''
        self.setNetwork(network)

    def activation_changed(self, model: ActivationEngine,
                           info: ActivationEngine.Change) -> None:
        # FIXME[old]
        if info.network_changed:
            self.network_changed(model._network, Network.Change.all())

        #
        # Set Layer info text
        #
        layer_text = ''
        #layer_id = model._layer
        layer_id = None  # FIXME[old]
        if False and info.layer_changed:  # FIXME[old]
            layer_text += '<br>\n<br>\n'
            layer_text += '<b>Layer info:</b><br>\n'
            if layer_id:
                layer = self._network.layer_dict[layer_id]
                layer_info = '\n'.join('<b>{}</b>: {}<br>'.format(key, val)
                                       for key, val in layer.info.items())
                layer_text += layer_info
            else:
                layer_text += "No layer selected"
        self._layer_text = layer_text
        self.update()

    def network_changed(self, network: Network,
                        change: Network.Change) -> None:

        #
        # Set Network info text
        #
        network_text = ''
        #network_text += '<b>Network info:</b> '
        if network is not None:
            network_name = type(network).__name__
            # network_text += 'FIXME[todo]: obtain network information ...'
            network_text += ('<br>\n<b>class:</b> '
                             f'{network.__class__.__name__}')
            network_text += f'<br>\n<b>name:</b> {network}'
            #if not network.empty():
            #    network_text += ('<br>\n<b>input layer:</b> '
            #                     f'{network.input_layer_id()}')
            #    network_text += ('<br>\n<b>output layer:</b> '
            #                     f'{network.output_layer_id()}')
        else:
            network_text += "No network"
        self._network_text = network_text

        self.update()
        #QMetaObject.invokeMethod(self, "setText", Q_ARG(str, text))

    def paintEvent(self, event):
        # text = ('<b>Network Info Box</b><br>' +
        #        self._network_text + self._layer_text)
        # self.setText(text)
        super().paintEvent(event)


##############################################################################
#                                                                            #
#                               NetworkList                                  #
#                                                                            #
##############################################################################

from toolbox import Toolbox
from network import Network
from ..utils import QObserver, QDebug

from typing import Dict, Iterable

from .register import RegisterItemList, QRegisterItemComboBox

class NetworkList(RegisterItemList, QDebug, qobservables={
        Toolbox: {'networks_changed', 'network_changed'},
        Network: {'state_changed'}}):
    """
    There are different ways to use a :py:class:`NetworkList`:

    * Standalone: selectable :py:class:`Network`\ s have to be
      added and removed explicitly by calling
      :py:meth:`addNetwork` and :py:class:`removeNetwork`.
      Standalone can be enabled by calling the :py:meth:`setStandalone`
      method.

    * Register: directly observing the py:class:`Network` register. The
      :py:class:`NetworkList` will be updated when new networks
      are initialized.

    * Toolbox: A :py:class:`Toolbox` can be set with the
      :py:meth:`setToolbox` method or by providing the toolbox
      argument to the constructor. This allows to only select
      networks from the toolbox.


    FIXME[todo]:
    A :py:class:`NetworkList` can be configured to only show a
    sublist of the actual network list. This is useful when run
    in Register or Toolbox mode, but only a subset of the available
    networks shall be presented:
    * preparation: just show networks that are prepared
    * superclass: just show networks that are subclasses of a given
      superclass, e.g., only classifiers.
    * framework: just show networks using a specific framework like
      TensorFlow, Torch, or Caffe.

    Attributes
    ----------
    _toolbox: Toolbox
        A :py:class:`Toolbox` providing a list of
        :py:class:`Network`s. Can be set with :py:meth:`setToolbox`,
        making the :py:class:`QNetworkSelector` an observer of that
        :py:class:`Toolbox`, reacting to 'networks_changed' signals.


    Parameters
    ----------
    FIXME[todo]: we may also restrict the selectable networks
        (e.g. setting an Autoencoder will only allow to select
        autoencoders).

    standalone: bool
        If `True`, the :py:class:`QNetworkSelector` will be run
        in standalone mode.  Networks have to be added manually.
    """

    def __init__(self, toolbox: Toolbox = None,
                 standalone: bool = False, **kwargs) -> None:
        """Initialization of the :py:class:`QNetworkSelector`.

        """
        if standalone and toolbox is not None:
            raise ValueError("No toolbox can be set in standalone mode.")
        register = Network if toolbox is None and not standalone else None
        super().__init__(register=register, **kwargs)
        if toolbox is not None:
            self.setToolbox(toolbox)

    def addNetwork(self, network: Network) -> None:
        """Add the given network to this :py:class:`NetworkList`.
        """
        if not self.standalone():
            raise RuntimeError("Explicitly adding networks is only allowed "
                               "when running in standalone mode.")
        self._addItem(network)

    def removeNetwork(self, network: Network) -> None:
        """Remove a network from this :py:class:`NetworkList`.
        """
        if not self.standalone():
            raise RuntimeError("Explicit removal of networks is only allowed "
                               "when running in standalone mode.")
        self._addItem(network)

    def standalone(self) -> None:
        """A flag indicating that this :py:class:`NetworkList` is
        running in standalone mode.
        """
        return self._toolbox is None and self._register is None

    def setStandalone(self) -> None:
        """Run this :py:class:`NetworkList` in standalone mode.
        The list can only be changed by calling :py:meth:`addNetwork` and
        :py:meth:`removeNetwork`.
        Neither toolbox changes nor changes of the Network MetaRegister
        are reflected in this list.
        """
        self.setToolbox(None)
        self.setRegister(None)

    def setToolbox(self, toolbox: Toolbox) -> None:
        print(f"\nWarning: Setting Network Toolbox ({toolbox}) "
              "is currently disabled ...\n")
        # palette = QPalette();
        # palette.setColor(QPalette.Background, Qt.darkYellow)
        # self.setAutoFillBackground(True)
        # self.setPalette(palette)
        if toolbox is not None:
            if self._register is not None:
                self.setRegister(None)
            self._updateFromIterator(toolbox.networks)
        else:
            self.setRegister(Network)

    def toolbox_changed(self, toolbox: Toolbox,
                        change: Toolbox.Change) -> None:
        """React to changes in the Toolbox. We are only interested
        when the list of networks has changed, in which case we will
        update the content of the QComboBox to reflect the available
        Networks.
        """
        # FIXME[bug]: for some reason we receive Network.Change
        # (instead of Toolbox.Change) changes here
        print(f"QNetworkSelector.toolbox_changed({change})")
        # print(f"\nWarning: Network Toolbox changes ({toolbox}, {change}]) "
        #       "are currently disabled ...\n")
        # return
        if change.networks_changed:
            self._updateFromIterator(toolbox.networks)

    def setNetwork(self, network) -> None:
        """
        """
        self.setEnabled(network is not None)

    def network_changed(self, network: Network,
                        change: Network.Change) -> None:
        LOG.debug("QNetworkSelector.network_changed(%s, %s)", network, change)

    def currentNetwork(self) -> Network:
        """The currently selected :py:class:`Network`.
        This may be `None` if no network is selected.
        """
        return self.currentData()

    def debug(self) -> None:
        super().debug()
        LOG.debug("Debug {type(self)}")
        print(f"debug: NetworkList[{type(self).__name__}]:")
        if self._toolbox is None:
            print(f"debug:   * No Toolbox")
        else:
            print("debug:   * Toolbox networks:")
            for network in self._toolbox.networks:
                print("debug:    -", network)
        print(f"debug:   * standalone: {self.standalone()}")


##############################################################################
#                                                                            #
#                            QNetworkSelector                                #
#                                                                            #
##############################################################################

class QNetworkSelector(NetworkList, QRegisterItemComboBox):
    """A widget to select a :py:class:`network.Network` from a list of
    :py:class:`network.Network`s.


    The class provides the common :py:class:`QComboBox` signals, including

    activated[str/int]:
        An item was selected (no matter if the current item changed)

    currentIndexChanged[str/int]:
        An new item was selected.

    """
    def __init__(self, **kwargs) -> None:
        """Initialization of the :py:class:`QNetworkSelector`.

        Parameters
        ----------
        """
        super().__init__(**kwargs)
        self.activated[str].connect(self.onActivated)  # FIXME[debug]
        self.activated[int].connect(self.onActivated)  # FIXME[debug]

    @protect
    def onActivated(self, name) -> None:
        """A slot for the `activated` signal
        """
        LOG.debug("QNetworkSelector.activated(%s)", name)

    def setNetwork(self, network: Network = None) -> None:
        super().setNetwork(network)
        if network is None:
            self.setCurrentIndex(-1)

    def debug(self) -> None:
        super().debug()
        print(f"debug: QNetworkSelector[{type(self).__name__}]:")
        print(f"debug:   * Current index: {self.currentIndex()}")


##############################################################################
#                                                                            #
#                                 QLayerSelector                             #
#                                                                            #
##############################################################################


from network import Network
from tools.activation import Engine as ActivationEngine

from ..utils import QObserver, protect

from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel
from PyQt5.QtWidgets import QVBoxLayout, QGridLayout

class QLayerSelector(QWidget, QObserver, qobservables={
        Network: {'state_changed'},
        ActivationEngine: {'layer_changed'}}):
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
    _current_selected: int
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
        self._activation = None  # FIXME[old]
        self._current_selected = None  # FIXME[old]
        self._exclusive = exclusive
        self._layerButtons = {}

        # FIXME[hack]: make a nicer solution and then remove this!
        self._label_input = None
        self._label_output = None

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
        if self._label_input is not None:
            self._label_input.deleteLater()
            self._label_input = None
        if self._label_output is not None:
            self._label_output.deleteLater()
            self._label_output = None
        self._current_selected = None

    def _initButtonsFromNetwork(self, network: Network) -> None:
        """Update the buttons to reflect the layers of the
        underlying network.

        Arguments
        ---------
        network: Network
            The :py:class:`Network` from which this
            :py:class:`QLayerSelector` can select layers.
        """
        LOG.info(f"QLayerSelector._initButtonsFromNetwork(%s)", network)
        # remove the old network buttons
        self._clearButtons()

        layout = self._layerLayout
        if network is None:
            self._label_input = QLabel("No network")
            layout.addWidget(self._label_input)
        elif not network.prepared:
            self._label_input = QLabel(f"Unprepared network '{network.key}'")
            layout.addWidget(self._label_input)
        else:
            # add a label describing the network input
            self._label_input = \
                QLabel(f"input = {self._network.get_input_shape(False)}")
            layout.addWidget(self._label_input)

            # a column of buttons to select the network layer
            # FIXME[hack]: layer_id vs. layer.id
            #for layer_id in network.layer_dict.keys():
            for layer_id, layer in network.layer_dict.items():
                #button = QPushButton(layer_id, self)
                button = QPushButton(layer.id, self)
                #self._layerButtons[layer_id] = button
                self._layerButtons[layer.id] = button
                layout.addWidget(button)
                button.resize(button.sizeHint())
                button.clicked.connect(self.onLayerButtonClicked)

            # add a label describing the network output
            self._label_output = \
                QLabel(f"output = {self._network.get_output_shape(False)}")
            layout.addWidget(self._label_output)

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
                 "sender=%s, current_selected=%s, activation=%s",
                 checked, self.sender().text(),
                 self._current_selected, self._activation)

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

    def old(self):

        if self._current_selected is not None:
            if self._activation:
                self._activation.remove_layer(self._current_selected)
            self._markButton(self._current_selected, False)
            if self._current_selected == layer:
                layer = None

        if layer is not None:
            if self._activation:
                self._activation.add_layer(layer)
            self._markButton(layer, True)

        self._current_selected = layer

    #
    # ActivationEngine
    #

    def setActivationEngine(self, activations: ActivationEngine) -> None:
        """Set the ActivationEngine for this QLayerSelector.
        """
        LOG.info("QLayerSelector.setActivationEngine(activations=%s): "
                 "old activation=%s", activations, self._activation)
        self._networkInfo.setActivationEngine(activations)
        self._activation = activations  # FIXME[hack]: should be done by QObserver

    def activation_changed(self, activation: ActivationEngine,
                           info: ActivationEngine.Change) -> None:
        """The QLayerSelector is interested in network related changes, i.e.
        changes of the network itself or the current layer.
        """
        LOG.debug("QLayerSelector.activation_changed(%s, %s)",
                  activation, info)

        # FIXME[design]: The 'network_changed' message can mean that
        # either the current model has changed or that the list of
        # models was altered (or both).
        if False and info.network_changed:  # FIXME[old]: should be redundant with network.observable_changed
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

            layer_id = activation.layer_id
            try:
                layer_index = activation.idForLayer(layer_id)
            except ValueError as error:
                print(f"ERROR: {self.__class__.__name__}."
                      "activation_changed(): {error}")
                layer_index = None
            if layer_index != self._current_selected:

                # unmark the previously active layer
                if self._current_selected is not None:
                    oldItem = self._layerButtons[self._current_selected]
                    self._markButton(oldItem, False)

                self._current_selected = layer_index
                if self._current_selected is not None:
                    # and mark the new layer
                    newItem = self._layerButtons[self._current_selected]
                    self._markButton(newItem, True)

    #
    # Network
    #

    def setNetwork(self, network: Network) -> None:
        self._networkInfo.setNetwork(network)

    def network_changed(self, network: Network,
                        change: Network.Change) -> None:
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
