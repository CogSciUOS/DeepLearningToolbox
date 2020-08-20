# standard imports
from typing import Union, Tuple
import sys

# third party imports
import numpy as np

# toolbox imports
from datasource import Datasource, Data
from base import Observable  # FIXME: should not be needed!
from base import View as BaseView, Controller as BaseController, run
from network import Network, Controller as NetworkController
from .engine import Engine

from . import logger


class View(BaseView, view_type=Engine):
    """Viewer for :py:class:`Engine`.

    Attributes
    ----------
    _engine: Engine
        The engine controlled by this Controller

    """

    def __init__(self, activation: Engine=None, **kwargs):
        super().__init__(observable=activation, **kwargs)


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

        Returns
        ------
        engine: Observable
            The engine controlled by this Controller.
        """
        return self._engine


