'''
.. moduleauthor:: Rasmus Diederichsen

.. module:: observer

This module contains definitions for observer functionality
'''
import model
import controller

class Observer(object):
    '''Mixin for inheriting observer functionality. An observer registers
    itself to a class which notifies its observers in case something
    changes. Every observer has an associated controller object for
    dispatching changes to the observed object.

    '''

    def __init__(self, **kwargs):
        '''Respected kwargs:

        Parameters
        ----------
        model   :   model.Model
                    Model to observe
        '''
        self._model = kwargs.get('model', None)

    def observe(self, obj):
        '''Add self to observer list of ``obj``'''
        obj.addObserver(self)

    def modelChanged(self, model: 'model.Model'=None, info: 'model.ModelChange'=None):
        '''Respond to change in the model.

        Parameters
        ----------
        model   :   model.Model
                    Model which changed (since we could observer multiple ones)
        info    :   model.ModelChange
                    Object for communicating which parts of the model changed
        '''
        pass

    def setController(self, controller : 'controller.InputController'):
        '''Set the controller for this observer. Will trigger observation of the controller's
        model

        Parameters
        ----------
        controller  :   controller.InputController
                        Controller for mediating communication with the observer object
        '''
        self._controller = controller
        self.observe(controller._model)
