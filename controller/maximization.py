import sys
import numpy as np
from controller import BaseController
from network import Network
from tools.am import Engine

# for type hints
import PyQt5  # FIXME: no Qt here
import qtgui  # FIXME: no Qt here
from typing import Union
import datasources


class MaximizationController(BaseController):
    """Controller for :py:class:`qtgui.panels.Maximization`.
    This class contains callbacks for all kinds of events which are
    effected by the user in the ``MaximizationPanel``."""

    def __init__(self, engine: Engine, **kwargs) -> None:
        """
        Parameters
        ----------
        model: Model
        """
        super().__init__(**kwargs)
        self._engine = engine

    def get_engine(self) -> Engine:
        """Get the activation maximization Engine for this
        MaximizationController.  This is that actually performs the
        maximization process and that sends out notification in
        response to commands issued by this Controller. Everyone
        interested in such notifications should register to this
        Observable.

        Result
        ------
        engine: Engine
            The engine controlled by this MaximizationController.
        """
        return self._engine

    # FIXME[design]: this is only implemented to satisfy
    # Observer::set_controller()
    def get_observable(self) -> Engine:
        return self._engine


    def onMaximize(self, reset=False):
        """Run the activation maximization process.

        """
        self._runner.runTask(self._engine.maximize_activation, reset=reset)


    def onStop(self):
        """Run the activation maximization process.

        """
        self._engine.stop()

    def onConfigViewSelected(self, configView):
        """Set the engine configuration from a configuration view.

        """
        if configView.config is not None:
            self._engine.config.assign(configView.config)
