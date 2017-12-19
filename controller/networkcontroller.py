import numpy as np
from random import randint


class NetworkController(object):
    '''Base controller backed by a network.'''

    _parent = None
    _data:   np.ndarray = None

    def __init__(self, model):
        self._model = model

    def setParent(self, controller):
        '''Set the parent controller.'''
        self._parent = controller

    def random(self):
        n_elems = len(self._model)
        index = randint(0, n_elems)
        self._model.editIndex(index)

    def advance(self):
        n_elems = len(self._model)
        self._model.editIndex(n_elems - 1)

    def advance_one(self):
        current_index = self._model._current_index
        self._model.editIndex(current_index + 1)

    def rewind(self):
        self._model.editIndex(0)

    def rewind_one(self):
        current_index = self._model._current_index
        self._model.editIndex(current_index - 1)

    def editIndex(self, index):
        self._model.editIndex(index)

    def mode_changed(self, mode):
        self._model.setMode(mode)
