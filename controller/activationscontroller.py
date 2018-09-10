import numpy as np
import sys
from controller import InputController
from network import Network
import model

# for type hints
import PyQt5, qtgui
from typing import Union
import datasources


class ActivationsController(InputController):
    """Controller for :py:class:`qtgui.panels.ActivationsPanel`.
    This class contains callbacks for all kinds of events which are
    effected by the user in the ``ActivationsPanel``."""

    def __init__(self, model: 'model.Model'):
        """
        Parameters
        ----------
        model   :   model.Model
        """
        super().__init__(model)

    def onUnitSelected(self, unit: int, sender: PyQt5.QtWidgets.QWidget):
        """(De)select a unit in the :py:class:`qtgui.widgets.QActivationView`.

        Parameters
        -----------
        unit    :   int
                    index of the unit in the layer
        sender  :   PyQt5.QtWidgets.QWidget
                    UI element triggering the callback
        """
        if self._model._current_activation is not None and unit is not None:
            self._runner.runTask(self._model.setUnit, unit)

    def onKeyPressed(self, sender: PyQt5.QtWidgets.QWidget):
        """Callback for handling keyboard events.

        Parameters
        ----------
        sender  :   PyQt5.QtWidgets.QWidget
                    UI element triggering the callback
        """
        pass

    def setInputData(self, raw: np.ndarray=None, fitted: np.ndarray=None,
                       description: str=None):
        """Callback for setting a new input data set.

        Parameters
        ----------
        raw :   np.ndarray
                Raw input data provided by the :py:class:`datasources.DataSource`
        fitted  :   np.ndarray
                    Input data fit to the network input layer
        description :   str
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
        print(f"ActivationsController.onLayerSelected('{layer}'): runTask(setLayer)")
        self._runner.runTask(self._model.setLayer, layer)

    def onSourceSelected(self, source: datasources.DataSource):
        """Set a new :py:class:`datasources.DataSource`.

        Parameters
        ----------
        source  :   datasources.DataSource
        """
        self._runner.runTask(self._model.setDataSource, source)

    def onNetworkSelected(self, network: Network, force_update: bool=False):
        """Callback for selection of a new network.

        Parameters
        ----------
        network :   network.network.Network
                    The new network object
        force_update    :   bool
                            Cause the model to broadcast an update regardless of whether the state
                            actually changed
        """
        self._runner.runTask(self._model.setNetwork, network, force_update)
