from base import Observable  # FIXME: should not be needed!
from base import View as BaseView, Controller as BaseController, run
from network import Network, Controller as NetworkController
from .engine import Engine

from . import logger

import sys
import numpy as np

# for type hints
from typing import Union, Tuple
import datasources

class View(BaseView, view_type=Engine):
    """Viewer for :py:class:`Engine`.

    Attributes
    ----------
    _engine: Engine
        The engine controlled by this Controller

    """

    def __init__(self, activation: Engine=None, **kwargs):
        super().__init__(observable=activation, **kwargs)

    def __str__(self) -> str:
        """String representation for the ActivationEngine.
        """
        self = self._engine  # FIXME[hack]
        result = f"Network = {self._network}"
        result += "\nLayers:"
        if self._layers is None:
            result += " None"
        else:
            for layer in self._layers:
                result += f" - {layer}"

        return result

class Controller(View, BaseController, Network.Observer):
    """Controller for :py:class:`qtgui.panels.ActivationsPanel`.
    This class contains callbacks for all kinds of events which are
    effected by the user in the ``ActivationsPanel``."""

    _network_controller: NetworkController = None

    def __init__(self, network: NetworkController=None, **kwargs) -> None:
        """
        Parameters
        ----------
        engine: Engine
        network: NetworkController
        """
        super().__init__(**kwargs)
        self.set_network_controller(network)

    def set_network_controller(self, network: NetworkController):
        self._network_controller = network
        if network is not None:
            interests = Network.Change('observable_changed')
            self._network_controller.add_observer(self, interests=interests)

    def set_network(self, network: Network) -> None:
        """Set the network to be used by this activation Controller. The
        activation Engine will use that network to compute activation
        values.
        """
        return self._network_controller(network)

    def network_changed(self, network: Network,
                        change: Network.Change) -> None:
        """Implementation of the Network.Observer interface. Whenever
        the network was changed, the new network will be assigned to
        the underlying activation Engine.
        """
        logger.debug(f"ActivationController: network is now {network} ({change})")
        if change.observable_changed:
            self._engine.set_network(network)

    def get_observable(self) -> Observable:
        """Get the Observable for the Controller.  This will be the
        object, that sends out notification in response to commands
        issued by this controller. Everyone interested in such
        notifications should register to this Observable:

        Result
        ------
        engine: Observable
            The engine controlled by this Controller.
        """
        return self._engine

    def onUnitSelected(self, unit: int, position: Tuple[int, int]) -> None:
        """(De)select a unit in the :py:class:`qtgui.widgets.QActivationView`.

        Parameters
        -----------
        unit: int
            Index of the unit in the layer
        position: Tuple[int, int]
            Position in the activation map of the current unit.
        """
        self._runner.runTask(self._engine.set_unit, unit, position)

    def setInputData(self, raw: np.ndarray=None, fitted: np.ndarray=None,
                     description: str=None) -> None:
        """Callback for setting a new input data set.

        Parameters
        ----------
        raw: np.ndarray
            Raw input data provided by the :py:class:`datasources.DataSource`
        fitted: np.ndarray
            Input data fit to the network input layer
        description: str
            Textual description of the data
        """
        pass

    @run
    def set_layer(self, layer: Union[int, str]):
        """Set the active layer.

        Parameters
        ----------
        layer: int or string
            The index or the name of the layer to activate.
        """
        self._engine.set_layer(layer)

    def onNetworkSelected(self, network_id: str):
        """Callback for selection of a new network.

        Parameters
        ----------
        network :   network.network.Network
                    The new network object
        """
        self._runner.runTask(self._engine.set_network, network_id)
        # self._runner.runTask(self._model.__class__.network.__set__,
        #                     self._model, network_id)

    def onNewInput(self, data: np.ndarray, target: int=None,
                   description: str = None) -> None:
        """Provide new input data.

        FIXME
        For a description of the parameters see Engine.set_input_data

        Parameters
        ----------
        data: np.ndarray
            The data array
        target: int
            The data label. None if no label is available.
        description: str
            A description of the input data.
        """        
        self._runner.runTask(self._engine.set_input_data,
                             data, target, description)

