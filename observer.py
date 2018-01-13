class Observer(object):
    '''Mixin for inheriting observer functionality.'''

    def __init__(self, **kwargs):
        '''Respected kwargs:

        Parameters
        ----------
        model   :   model.Model
                    Model to observe
        '''
        self._model = kwargs.get('model', None)

    def observe(self, obj):
        '''Add self to observer list of `obj`'''
        obj.add_observer(self)

    def modelChanged(self, model=None, info=None):
        '''Respond to change in the model.

        Parameters
        ----------
        model   :   model.Model
                    Model which changed (since we could observer multiple ones)
        info    :   model.ModelChange
                    Object for communicating which parts of the model changed
        '''
        pass

    def setController(self, controller):
        '''Set the controller for this observer. Will trigger observation of the controller\'s
        model'''
        self._controller = controller
        self.observe(controller._model)
