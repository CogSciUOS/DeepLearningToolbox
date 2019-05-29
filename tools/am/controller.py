from .engine import Engine
from base import View as BaseView, Controller as BaseController, run
from network import Network

import sys
import numpy as np


class View(BaseView, view_type=Engine):
    """Viewer for :py:class:`Engine`.

    Attributes
    ----------
    _engine: Engine
        The engine controlled by this Controller

    """

    def __init__(self, engine: Engine=None, **kwargs):
        super().__init__(observable=engine, **kwargs)


class Controller(View, BaseController):
    """Controller for :py:class:`Engine`.
    This class contains callbacks for all kinds of events which are
    effected by the user in the ``MaximizationPanel``.

    Attributes
    ----------
    _engine: Engine
        The engine controlled by this Controller
    """

    def __init__(self, engine: Engine, **kwargs) -> None:
        """
        Parameters
        ----------
        engine: Engine
        """
        super().__init__(engine=engine, **kwargs)
        self._network = None

    def get_engine(self) -> Engine:
        """Get the activation maximization Engine for this
        Controller.  This is the object that actually performs the
        maximization process and that sends out notification in
        response to commands issued by this Controller. Everyone
        interested in such notifications should register to this
        Observable.

        Result
        ------
        engine: Engine
            The engine controlled by this Controller.
        """
        return self._engine

    # FIXME[design]: this is only implemented to satisfy
    # Observer::set_controller()
    def get_observable(self) -> Engine:
        return self._engine

    def get_network(self) -> Network:
        return self._engine.network

    def set_network(self, network: Network) -> None:
        self._engine.network = network

    @run
    def start(self, reset: bool=False):
        """Run the activation maximization process.

        """
        self._engine.maximize_activation(reset=reset)

    def stop(self):
        """Run the activation maximization process.

        """
        self._engine.stop()

    def onConfigViewSelected(self, configView):
        """Set the engine configuration from a configuration view.

        """
        if configView.config is not None:
            self._engine.config.assign(configView.config)
