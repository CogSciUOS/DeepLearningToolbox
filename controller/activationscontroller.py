import numpy as np
from PyQt5.QtCore import Qt

from controller import NetworkController

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
        if self._model._activation is None:
            unit = None
        elif unit is not None and (unit < 0 or unit >= self._model._activation.shape[0]):
            unit = None
        if self._selectedUnit != unit:
            self._selectedUnit = unit
            sender.update()

    def on_key_pressed(self, sender):
        pass
