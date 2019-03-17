from base import Observable  # FIXME: should not be needed!
from base import View as BaseView, Controller as BaseController
from network import Network, Controller as NetworkController
from .engine import Engine

import sys
import numpy as np

# for type hints
import PyQt5.QtWidgets  # FIXME: no Qt here
from typing import Union
import datasources

class View(BaseView, view_type=Engine):
    """Viewer for :py:class:`Engine`.

    Attributes
    ----------
    _engine: Engine
        The engine controlled by this Controller

    """

    def __init__(self, engine: Engine=None, **kwargs):
        super().__init__(observable=engine, **kwargs)



class Controller(BaseController, Network.Observer):
    """Controller for :py:class:`qtgui.panels.ActivationsPanel`.
    This class contains callbacks for all kinds of events which are
    effected by the user in the ``ActivationsPanel``."""

    _network: NetworkController = None

    def __init__(self, engine: Engine=None, network: NetworkController=None,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        engine: Engine
        network: NetworkController
        """
        super().__init__(**kwargs)
        self._engine = engine
        self.set_network_controller(network)

    def set_network_controller(self, network: NetworkController):
        self._network = network
    
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

    def onUnitSelected(self, unit: int, sender: PyQt5.QtWidgets.QWidget):
        """(De)select a unit in the :py:class:`qtgui.widgets.QActivationView`.

        Parameters
        -----------
        unit    :   int
                    index of the unit in the layer
        sender  :   PyQt5.QtWidgets.QWidget
                    UI element triggering the callback
        """
        self._runner.runTask(self._engine.set_unit, unit)

    def onKeyPressed(self, sender: PyQt5.QtWidgets.QWidget):
        """Callback for handling keyboard events.

        Parameters
        ----------
        sender  :   PyQt5.QtWidgets.QWidget
                    UI element triggering the callback
        """
        pass

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

    def onLayerSelected(self, layer: Union[int, str]):
        """Set the active layer.

        Parameters
        ----------
        layer   :   int or string
                    The index or the name of the layer to activate.
        """
        self._runner.runTask(self._engine.set_layer, layer)

    def onNetworkSelected(self, network_id: str):
        """Callback for selection of a new network.

        Parameters
        ----------
        network :   network.network.Network
                    The new network object
        """
        print(f"activations.onNetworkSelected({network_id})")
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

    def getNetworkController(self):
        return self._network

    def set_network(self, network: Network):
        return self._network(network)

    def network_changed(self, network: Network,
                        change: Network.Change) -> None:
        if change.observable_changed:
            self._engine.set_network(network)
