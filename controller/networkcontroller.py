import numpy as np
from random import randint


class NetworkController(object):
    '''Base controller backed by a network. Contains functionality for manipulating input index.'''

    _parent = None
    _data:   np.ndarray = None

    def __init__(self, model):
        self._model = model

    def setParent(self, controller):
        '''Set the parent controller.

        Parameters
        ----------
        controller  :   controller.NetworkController
                        Parent controller to which stuff will be delegated.
        '''
        self._parent = controller

    def random(self):
        '''Select a random index into the dataset.'''
        n_elems = len(self._model)
        index = randint(0, n_elems)
        self._model.editIndex(index)

    def advance(self):
        '''Advance data index to end.'''
        n_elems = len(self._model)
        self._model.editIndex(n_elems - 1)

    def advance_one(self):
        '''Advance data index by one.'''
        current_index = self._model._current_index
        self._model.editIndex(current_index + 1)

    def rewind(self):
        '''Reset data index to zero.'''
        self._model.editIndex(0)

    def rewind_one(self):
        '''Rewind data index by one.'''
        current_index = self._model._current_index
        self._model.editIndex(current_index - 1)

    def editIndex(self, index):
        '''Set data index to specified values.

        Parameters
        ----------
        '''
        self._model.editIndex(index)

    def mode_changed(self, mode):
        '''Change the model\'s data mode.

        Parameters
        ----------
        mode    :   str
                    One of 'dir' or 'array'
        '''
        self._model.setMode(mode)
