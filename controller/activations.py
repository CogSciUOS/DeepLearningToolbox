import sys
import numpy as np
from controller import BaseController
from base.observer import Observable  # FIXME: should not be needed!
from network import Network
from model import Model

# for type hints
import PyQt5.QtWidgets  # FIXME: no Qt here
#import qtgui  # FIXME: no Qt here
from typing import Union
import datasources


class ActivationsController(BaseController):
    """Controller for :py:class:`qtgui.panels.ActivationsPanel`.
    This class contains callbacks for all kinds of events which are
    effected by the user in the ``ActivationsPanel``."""

    def __init__(self, model: Model, **kwargs) -> None:
        """
        Parameters
        ----------
        model: Model
        """
        super().__init__(**kwargs)
        self._model = model

    def get_observable(self) -> Observable:
        """Get the Observable for the ActivationsController.  This will be the
        object, that sends out notification in response to commands
        issued by this controller. Everyone interested in such
        notifications should register to this Observable:

        Result
        ------
        model: Observable
            The model controlled by this ActivationsController.
        """
        return self._model

    def onUnitSelected(self, unit: int, sender: PyQt5.QtWidgets.QWidget):
        """(De)select a unit in the :py:class:`qtgui.widgets.QActivationView`.

        Parameters
        -----------
        unit    :   int
                    index of the unit in the layer
        sender  :   PyQt5.QtWidgets.QWidget
                    UI element triggering the callback
        """
        self._runner.runTask(self._model.set_unit, unit)

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
        self._runner.runTask(self._model.set_layer, layer)

    def onNetworkSelected(self, network_id: str):
        """Callback for selection of a new network.

        Parameters
        ----------
        network :   network.network.Network
                    The new network object
        """
        print(f"activations.onNetworkSelected({network_id})")
        self._runner.runTask(self._model.set_network, network_id)
        # self._runner.runTask(self._model.__class__.network.__set__,
        #                     self._model, network_id)

    def onNewInput(self, data: np.ndarray, target: int=None,
                   description: str = None) -> None:
        """Provide new input data.

        FIXME
        For a description of the parameters see Model.set_input_data

        Parameters
        ----------
        data: np.ndarray
            The data array
        target: int
            The data label. None if no label is available.
        description: str
            A description of the input data.
        """        
        self._runner.runTask(self._model.set_input_data,
                             data, target, description)
