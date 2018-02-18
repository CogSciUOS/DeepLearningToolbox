import numpy as np
from controller import NetworkController
from network import Network
import model

from concurrent.futures import ThreadPoolExecutor, Future
from PyQt5.QtCore import QObject, pyqtSignal

class AsyncRunner(object):

    def __init__(self, model):
        super().__init__()
        self._model = model
        self._executor = ThreadPoolExecutor(max_workers=1)

    def run_task(self, fn, *args, **kwargs):
        future = self._executor.submit(fn, *args, **kwargs)
        future.add_done_callback(self.on_completion)

    def on_completion(self, result):
        pass

class QTAsyncRunner(AsyncRunner, QObject):

    _completion_signal = pyqtSignal(object)

    def __init__(self, model):
        super().__init__(model)
        self._completion_signal.connect(lambda info: model.notifyObservers(info))

    def on_completion(self, future):
        self._completion_signal.emit(future.result())


class ActivationsController(NetworkController):
    '''Controller for ``ActivationsPanel``'''

    def __init__(self, model : 'model.Model'):
        '''
        Parameters
        ----------
        model   :   model.Model
        '''
        self._model = model
        self._runner = QTAsyncRunner(model)

    def on_unit_selected(self, unit : int, sender):
        '''(De)select a unit in the ``QActivationView``.

        Parameters
        -----------
        unit    :   int
                    index of the unit in the layer

        '''
        if self._model._current_activation is not None and unit is not None:
            self._runner.run_task(self._model.setUnit, unit)

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
        '''Callback for setting a new input data set.

        Parameters
        ----------
        raw :   np.ndarray
                Raw input data provided by the ``DataSource``
        fitted  :   np.ndarray
                    Input data fit to the network input layer
        description :   str
                        Textual description of the data
        '''
        pass

    def on_layer_selected(self, layer):
        '''Set the active layer.

        Parameters
        ----------
        layer   :   int or string
                    The index or the name of the layer to activate.
        '''
        self._runner.run_task(self._model.setLayer, layer)

    def source_selected(self, source):
        '''Set a new ``DataSource``

        Parameters
        ----------
        source  :   DataSource
        '''
        self._runner.run_task(self._model.setDataSource, source)

    def on_network_selected(self, network, force_update=False):
        self._runner.run_task(self._model.setNetwork, network, force_update)

    def on_open_button_clicked(self, sender=None):
        '''Helper callback for handling the click on the ``Open`` button. Unfortunately, this cannot
        be handled directly in the GUI layer since we need to coordinate with the model\'s current
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
            print('Could not open file.')
