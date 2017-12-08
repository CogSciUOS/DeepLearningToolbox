

class NetworkController(object):
    '''Base controller backed by a network.'''

    _parent = None
    _data   :   np.ndarray = None

    def __init__(self, model):
        self._model = model

    def setParent(self, controller):
        '''Set the parent controller.'''
        self._parent = controller

    def setInputData(self, data: np.ndarray=None, description: str=None):
        self._model.setInputData(data, description)
        self._model.notifyObservers()
