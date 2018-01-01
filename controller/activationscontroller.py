import numpy as np
from .networkcontroller import NetworkController


class ActivationsController(NetworkController):
    '''Controller for ``ActivationsPanel``'''

    _parent = None
    _selectedUnit: int = None

    def __init__(self, model):
        self._model = model

    def on_unit_selected(self, unit: int, sender):
        '''(De)select a unit in this QActivationView.

        Parameters
        -----------
        unit    :   int
                    index of the unit in the layer

        '''
        if self._model._current_activation is None:
            unit = None
        elif unit is not None and (unit < 0 or unit >= self._model._activation.shape[0]):
            unit = None
        if self._selectedUnit != unit:
            self._selectedUnit = unit
            sender.update()

    def on_key_pressed(self, sender):
        '''Callback for handling keyboard events.

        Parameters
        ----------
        sender  :   QWidget
                    Widget receiving the event
        '''
        pass

    def set_input_data(self, raw: np.ndarray=None, fitted: np.ndarray=None,
                       description: str=None):
        pass

    def on_layer_selected(self, layer):
        '''Set the active layer.

        Parameters
        ----------
        layer : int or string
            The index or the name of the layer to activate.
        '''
        self._model.setLayer(layer)

    def on_network_selected(self, network_str):
        self._model.setNetwork(network_str)

    def source_selected(self, source):
        self._model.setDataSource(source)

    def onNetworkSelected(self, index):
        print(f'Selecting network {index}')
