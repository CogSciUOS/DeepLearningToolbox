class Observer(object):

    '''Mixin for inheriting observer functionality.'''

    def __init__(self, **kwargs):
        self._model = kwargs.get('model', None)

    def observe(self, obj):
        obj.add_observer(self)

    def modelChanged(self, model=None):
        pass

    def setController(self, controller):
        self._controller = controller
        self.observe(controller._model)
