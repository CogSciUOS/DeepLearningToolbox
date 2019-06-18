###############################################################################
###                           QNetworkBox                                   ###
###############################################################################

from network import Network, View as NetworkView
from tools.activation import Engine as ActivationEngine

from ..utils import QObserver

from PyQt5.QtCore import QMetaObject, Q_ARG
from PyQt5.QtWidgets import QLabel


class QNetworkBox(QLabel, QObserver, Network.Observer,
                  ActivationEngine.Observer):
    '''
    .. class:: QNetworkBox

    A simple widget to display information on a network and a selected
    layer'''

    _networkView: NetworkView = None

    def __init__(self, network: NetworkView=None, parent=None):
        '''Create a new :py:class:``QNetworkInfoBox``'''
        super().__init__(parent)
        self.setWordWrap(True)
        self._network_text = ''
        self._layer_text = ''
        self.setNetworkView(network)

    def setNetworkView(self, network: NetworkView) -> None:
        self._exchangeView('_networkView', network,
                           Network.Change.all())  # FIXME[hack]: check what we are really interested in ...

    def activation_changed(self, model: ActivationEngine,
                           info: ActivationEngine.Change) -> None:
        # FIXME[old]
        if info.network_changed:
            self.network_changed(model._network, Network.Change.all())
        
        #
        # Set Layer info text
        #
        layer_text = ''
        layer_id = model._layer
        if info.layer_changed:
            layer_text += '<br>\n<br>\n'
            layer_text += '<b>Layer info:</b><br>\n'
            if layer_id:
                layer = network.layer_dict[layer_id]
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
        network_text += '<b>Network info:</b> '
        if network is not None:
            network_name = type(network).__name__
            network_text += 'FIXME[todo]: obtain network information ...'
            network_text += ('<br>\n<b>class:</b> '
                             f'{network.__class__.__name__}')
            network_text += f'<br>\n<b>name:</b> {network}'
            if not network.empty():
                network_text += ('<br>\n<b>input layer:</b> '
                                 f'{network.input_layer_id()}')
                network_text += ('<br>\n<b>output layer:</b> '
                                 f'{network.output_layer_id()}')
        else:
            network_text += "No network"
        self._network_text = network_text

        self.update()
        #QMetaObject.invokeMethod(self, "setText", Q_ARG(str, text))


    def paintEvent(self, event):
        text = ('<b>Network Info Box</b><br>' +
                self._network_text + self._layer_text)
        self.setText(text)
        super().paintEvent(event)
        

###############################################################################
###                           QNetworkSelector                              ###
###############################################################################

from toolbox import Toolbox, ToolboxView
from network import View as NetworkView
from tools.activation import Engine as ActivationEngine
from ..utils import QObserver

from typing import Dict, Iterable

from PyQt5.QtWidgets import QComboBox


class QNetworkSelector(QComboBox, QObserver, Toolbox.Observer,
                       Network.Observer, ActivationEngine.Observer):
    """A widget to select a :py:class:`Network` form a list of
    :py:class:`Network`s. The selected network can be made subject
    of an associated :py:class:`NetworkView`.
    
    Attributes
    ----------
    _network_map : Dict[str,Network]
        A dictionary mapping the strings displayed in the network selector
        dropdown to actual network objects.

    _toolboxView: ToolboxView
        View of a :py:class:`Toolbox` providing a list of
        :py:class:`Network`s. Can be set with :py:meth:`setToolboxView`,
        making the :py:class:`QNetworkSelector` an observer of that
        :py:class:`Toolbox`, reacting to 'networks_changed' signals.

    _networkView: NetworkView
        View of a :py:class:`Network`. The :py:class:`QNetworkSelector`
        is only interested in the :py:class:`Network` being viewed,
        not in its properties. This network will be made the current
        network of the :py:class:`QNetworkSelector`, and selecting
        another network will also change the viewed network accordingly.
        A :py:class:`NetworkView` can be set with
        :py:meth:`setNetworkView`, making the
        :py:class:`QNetworkSelector` an observer of that
        :py:class:`Toolbox`, reacting to 'networks_changed' signals.
        The networkView may also restrict the selectable networks
        (e.g. setting an AutoencoderView will only allow to select
        autoencoders).
    """
    _network_map = {}
    _toolboxView: ToolboxView = None
    _networkView: NetworkView = None
    
    def __init__(self, **kwargs) -> None:
        '''Initialization of the QNetworkView.

        Parameters
        ----------
        parent : QWidget
            The parent argument is sent to the QComboBox constructor.
        '''
        QComboBox.__init__(self, **kwargs)  # FIXME[hack]: call super() ...
        ActivationEngine.Observer.__init__(self)  # FIXME[hack]: call super() ...
        Toolbox.Observer.__init__(self)  # FIXME[hack]: call super() ...
        self.setToolboxView(None)   # FIXME[hack]: provide arguments ...
        self.setNetworkView(None)   # FIXME[hack]: provide arguments ...

        # make sure to connect the signal in the main thread (assuming
        # the QNetworkSelector is initialized in the main thread).
        def slot(name):
            if self._networkView is not None:
                self._networkView(self.currentData())
        self.activated[str].connect(slot)

    def setController(self, controller) -> None:
        super().setController(controller)
        self.setEnabled(controller is not None)

    def setToolboxView(self, toolbox: ToolboxView) -> None:
        self._exchangeView('_toolboxView', toolbox,
                           Toolbox.Change('networks_changed'))

    def setNetworkView(self, network: NetworkView) -> None:
        self._exchangeView('_networkView', network,
                           Network.Change('observable_changed'))

    def activation_changed(self, activation: ActivationEngine,
                           info: ActivationEngine.Change) -> None:
        """React to changes in the model.
        """
        
        # The 'network_changed' message can mean that the network has
        # changed.
        if info.network_changed:

            network = model.network
            if network is not None:
                new_index = self.findText(network.get_id())
                self.setCurrentIndex(new_index)

    def toolbox_changed(self, toolbox: Toolbox,
                        change: Toolbox.Change) -> None:
        """React to changes in the Toolbox. We are only interested
        when the list of networks has changed, in which case we will
        update the content of the QComboBox to reflect the available
        Networks.
        """
        self._update_networks(self._toolboxView.networks)

    def network_changed(self, network: Network,
                        change: Network.Change) -> None:
        """React to changes in the Network. We are only interested
        when the observable, i.e. the selected network has changed. In
        that case we will update the current item in the QComboBox to
        reflect display the selected Network.
        """
        if network is not None:
            new_index = self.findText(network.get_id())
            self.setCurrentIndex(new_index)
        else:
            self.setCurrentIndex(-1)           

    def _update_networks(self, networks: Iterable[Network]) -> None:
        """Update the list of networks to choose from.

        Arguments
        ---------
        networks: an Iterable providing available Networks.
        """
        networks = [n for n in networks]
        if len(networks) != self.count():
            self.clear()
            for network in networks:
                self.addItem(network.get_id(), network)

    @property
    def network(self) -> Network:
        """The currently selected Network.
        """
        return self.currentData()

##############################################################################
###                                QNetworkView                            ###
##############################################################################


from network import Network, View as NetworkView
from tools.activation import (Engine as ActivationEngine,
                              Controller as ActivationsController)

from ..utils import QObserver, protect

from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel
from PyQt5.QtWidgets import QVBoxLayout, QGridLayout


class QNetworkView(QWidget, QObserver, Network.Observer,
                   ActivationEngine.Observer):
    """A simple widget to display information on a network and select a
    layer.

    The Layers are presented as a stack of buttons. Clicking at a
    button will set the corresponding layer in the
    py:class:`ActivationEngine`.

    The :py:class:`QNetworkView` observes a py:class:`Network` and an
    py:class:`ActivationEngine`. If the network is changed, the
    new network architecture needs to be displayed. 

    Attributes
    ----------
    _network: NetworkView
        The network currently shown by this :py:class:`QNetworkView`
    _activations: ActivationsController
        Controller for this UI element
    _current_selected: int
        Layer id of the currently selected layer.
    """
    _activations: ActivationsController = None
    _network: NetworkView = None

    _current_selected: int = None

    # FIXME[hack]: make a nicer solution and then remove this!
    _label_input = None
    _label_output = None

    # Graphical elements
    _networkInfo: QNetworkBox = None

    # FIXME[old]: is this still used? remove ...
    _UPDATE_NETWORK = QEvent.registerEventType()

    def __init__(self, network: NetworkView=None, **kwargs):
        '''Initialization of the QNetworkView.

        Parameters
        ----------
        parent : QWidget
            The parent argument is sent to the QWidget constructor.
        '''
        super().__init__(**kwargs)
        self.initUI()
        self.layoutUI()
        self.setNetworkView(network)

    def initUI(self) -> None:
        '''Initialize the user interface.'''
        self._networkInfo = QNetworkBox(parent=self)
        self._networkInfo.setMinimumWidth(300)
        self._layer_buttons = []       

    def layoutUI(self) -> None:
        self._layer_layout = QVBoxLayout()
        
        info_layout = QVBoxLayout()
        info_layout.addWidget(self._networkInfo)

        layout = QVBoxLayout()
        layout.addLayout(self._layer_layout)
        layout.addLayout(info_layout)
        layout.addStretch(1)
        self.setLayout(layout)

    def setActivationsController(self, activations: ActivationsController):
        """Set the ActivationsController for this QNetworkView.
        """
        interests = ActivationEngine.Change('layer_changed')
        self._exchangeView('_activations', activations,
                           interests=interests)
        self._networkInfo.setController(activations)

    def activation_changed(self, activation: ActivationEngine,
                           info: ActivationEngine.Change) -> None:
        """The QNetworkView is interested in network related changes, i.e.
        changes of the network itself or the current layer.
        
        """

        # FIXME[design]: The 'network_changed' message can mean that
        # either the current model has changed or that the list of
        # models was altered (or both).
        if False and info.network_changed:  # FIXME[old]: should be redundant with network.observable_changed
            #
            # Respond to network change
            #
            
            # As we may add/remove QWidgets, we need to make sure that
            # this method is executed in the main thread
            event = QEvent(QNetworkView._UPDATE_NETWORK)
            event.activation = activation
            QCoreApplication.postEvent(self, event)

        if ((info.network_changed or info.layer_changed) and
            activation is not None):
            #
            # Respond to layer change
            #

            layer_id = activation.layer_id
            try:
                layer_index = activation.idForLayer(layer_id)
            except ValueError as error:
                print(f"ERROR: {self.__class__.__name__}.activation_changed(): "
                      "{error}")
                layer_index = None
            if layer_index != self._current_selected:
                
                # unmark the previously active layer
                if self._current_selected is not None:
                    oldItem = self._layer_buttons[self._current_selected]
                    self._markButton(oldItem, False)

                self._current_selected = layer_index
                if self._current_selected is not None:
                    # and mark the new layer
                    newItem = self._layer_buttons[self._current_selected]
                    self._markButton(newItem, True)

    def setNetworkView(self, network: NetworkView):
        self._exchangeView('_network', network,
                           interests=Network.Change('observable_changed'))
        self._networkInfo.setNetworkView(network)        

    def network_changed(self, network: Network,
                        change: Network.Change) -> None:
        if change.observable_changed:
            self._updateButtons(network)

    def _updateButtons(self, network: Network):
        """Update the buttons to reflect the layers of the
        underlying network.
        
        Arguments
        ---------
        network: the curent network.
        """
        #
        # Respond to network change
        #
        layout = self._layer_layout
        self._current_selected = None
        if network:
            #
            # remove the old network buttons
            #
                
            # FIXME[todo]: (still not sure what is the correct way to do:
            # widget.deleteLater(), widget.close(),
            # widget.setParent(None), or
            # layout.removeWidget(widget) + widget.setParent(None))
            for button in self._layer_buttons:
                button.deleteLater()
            self._layer_buttons = []
            if self._label_input is not None:
                self._label_input.deleteLater()
                self._label_input = None
            if self._label_output is not None:
                self._label_output.deleteLater()
                self._label_output = None                   

            self._label_input = QLabel("input = " + str(network.get_input_shape(False)))
            layout.addWidget(self._label_input)

            #
            # a column of buttons to select the network layer
            #
            for name in network.layer_dict.keys():
                button = QPushButton(name, self)
                self._layer_buttons.append(button)
                layout.addWidget(button)
                button.resize(button.sizeHint())
                button.clicked.connect(self.onLayerButtonClicked)

            self._label_output = QLabel("output = " + str(network.get_output_shape(False)))
            layout.addWidget(self._label_output)

            # choose the active layer for the new network
            # self.setActiveLayer(0)
            if self._activations:
                self._activations.set_layer(None)


    def _markButton(self, button, state:bool=True) -> None:
        """Mark a button as selected.

        Arguments
        ---------
        state: a flag indicating if the button is selected or deselected.
        """
        if button is not None:
            button.setStyleSheet('background-color: red' if state else '')
        

    @protect
    def onLayerButtonClicked(self, checked: bool):
        """Callback for clicking one of the layer buttons.
        If we have an ActivationsController set, we will inform it
        that 
        """
        if (self._current_selected is not None and
            self._layer_buttons[self._current_selected] == self.sender()):
            layer = None
        else:
            layer = self.sender().text()
        if self._activations:
            self._activations.set_layer(layer)

    # FIXME[old]
    def event(self, event: QEvent) -> bool:
        """React to the UPDATE_NETWORK event. 
        """
        if event.type() == QNetworkView._UPDATE_NETWORK:
            self._updateButtons(event.activation._network)
            return True
        return super().event(event)

