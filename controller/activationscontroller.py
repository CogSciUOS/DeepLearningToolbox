import numpy as np
import sys
from controller import InputController
from network import Network
import model

# for type hints
import PyQt5, qtgui
from typing import Union


class ActivationsController(InputController):
    '''Controller for :py:class:``qtgui.panels.ActivationsPanel``.
    This class contains callbacks for all kinds of events which are effected by the user in the
    ``ActivationsPanel``.'''

    def __init__(self, model: 'model.Model'):
        '''
        Parameters
        ----------
        model   :   model.Model
        '''
        super().__init__(model)

    def on_unit_selected(self, unit: int, sender: PyQt5.QtWidgets.QWidget):
        '''(De)select a unit in the :py:class:``qtgui.widgets.QActivationView``.

        Parameters
        -----------
        unit    :   int
                    index of the unit in the layer
        sender  :   PyQt5.QtWidgets.QWidget
                    UI element triggering the callback
        '''
        if self._model._current_activation is not None and unit is not None:
            self._runner.run_task(self._model.setUnit, unit)

    def on_key_pressed(self, sender: PyQt5.QtWidgets.QWidget):
        '''Callback for handling keyboard events.

        Parameters
        ----------
        sender  :   PyQt5.QtWidgets.QWidget
                    UI element triggering the callback
        '''
        pass

    def set_input_data(self, raw: np.ndarray=None, fitted: np.ndarray=None,
                       description: str=None):
        '''Callback for setting a new input data set.

        Parameters
        ----------
        raw :   np.ndarray
        Raw input data provided by the :py:class:``qtgui.datasources.DataSource``
        fitted  :   np.ndarray
                    Input data fit to the network input layer
        description :   str
                        Textual description of the data
        '''
        pass

    def on_layer_selected(self, layer: Union[int, str]):
        '''Set the active layer.

        Parameters
        ----------
        layer   :   int or string
                    The index or the name of the layer to activate.
        '''
        self._runner.run_task(self._model.setLayer, layer)

    def source_selected(self, source: qtgui.datasources.DataSource):
        '''Set a new :py:class:``qtgui.datasources.DataSource``.

        Parameters
        ----------
        source  :   qtgui.datasources.DataSource
        '''
        self._runner.run_task(self._model.setDataSource, source)

    def on_network_selected(self, network: Network, force_update: bool=False):
        '''Callback for selection of a new network.

        Parameters
        ----------
        network :   Network
                    The new network object
        force_update    :   bool
                            Cause the model to broadcast an update regardless of whether the state
                            actually changed
        '''
        self._runner.run_task(self._model.setNetwork, network, force_update)

    def on_open_button_clicked(self, sender: PyQt5.QtWidgets.QWidget=None):
        '''Helper callback for handling the click on the ``Open`` button. Unfortunately, this cannot
        be handled directly in the GUI layer since we need to coordinate with the model's current
        mode.

        Parameters
        ----------
        sender  :   PyQt5.QtWidgets.QWidget
                    GUI element receiving the click.
        '''
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
            # TODO: Inform user via GUI
            print('Could not open file.', file=sys.stderr)
