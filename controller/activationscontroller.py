import numpy as np
from controller import NetworkController
from network import Network


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

    def source_selected(self, source):
        self._model.setDataSource(source)

    def on_network_selected(self, network):
        if isinstance(network, Network):
            raise ValueError('Setting network via object is not supported.')
        if isinstance(network, int) or isinstance(network, str):
            print(f'Selecting network {network}')
            self._model.setNetwork(network)

    def on_open_button_clicked(self, sender=None):
        from qtgui.widgets.inputselector import DataDirectory, DataFile
        try:
            source = self._model._current_source
            mode = self._model._current_mode
            if mode == 'array':
                if not isinstance(source, DataFile):
                    source = DataFile()
                source.selectFile(sender)
            elif mode == 'dir':
                if not isinstance(source, DataDirectory):
                    source = DataDirectory()
                source.selectDirectory(sender)
            self.source_selected(source)
        except FileNotFoundError:
            pass
