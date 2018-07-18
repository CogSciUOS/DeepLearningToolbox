import numpy as np
from random import randint
from model import Model

from concurrent.futures import ThreadPoolExecutor, Future
from PyQt5.QtCore import QObject, pyqtSignal

class AsyncRunner(object):
    '''Base class for runner objects which must be provided for each
    controller/user interface.

    .. note:: *Why is this necessary?*

    When working with Qt for instance, it is not possible to update
    the gui from a non-main thread.  So we cannot simply run a thread
    to do expensive computations and have it call
    :py:meth:`model.Model.notifyObservers` since that call would
    invoke methods on :py:class:`PyQt5.QWidget`s. We would need some
    way to call back into the main thread, but that one is busy running
    Qt's main loop so the app can respond to further user action. Qt's
    way of resolving this is with signals. But we do not want to make
    this a fixed solution since the app should be as independent of Qt
    as possible. I have thus decided that a dedicated runner must be
    provided for each kind of user interface which knows how to handle
    asynchronous updates.

    Attributes
    ----------
    _model   :   model.Model
                Model whose updates should be broadcast asynchronously
    _executor   :   ThreadPoolExecutor
                    Singular reusable thread to run computations in.

    '''

    def __init__(self, model: Model):
        super().__init__()
        self._model = model
        self._executor = ThreadPoolExecutor(max_workers=1)

    def runTask(self, fn, *args, **kwargs):
        '''Schedule the execution of a function. This is equivalent to running::

            fn(*args, **kwargs)

        asynchronously.

        Parameters
        ----------
        fn  :   function
                Function to run asynchronously
        args    :   list
                    Non-keyword args to ``fn``
        kwargs  :   dict
                    keyword args to ``fn``
        '''
        future = self._executor.submit(fn, *args, **kwargs)
        future.add_done_callback(self.onCompletion)

    def onCompletion(self, result):
        '''Callback exectuted on completion of a running task. This method must be implemented.
        By subclasses and - together with any necessary initialisations in  :py:meth:`__init__` -
        should lead to calling :py:meth:`model.Model.notifyObservers` on the main thread.

        Parameters
        ----------
        result  :   object or model.ModelChange
                    Object returned by the asynchronously run method.
        '''
        raise NotImplementedError('This abstract base class should not be used directly.')

class QTAsyncRunner(AsyncRunner, QObject):
    ''':py:class:`AsyncRunner` subclass which knows how to update Qt widgets.

    Attributes
    ----------
    _completion_signal  :   pyqtSignal
                            Signal emitted once the computation is done. The signal is connected to
                            :py:meth:`onCompletion` which will be run on the main thread by the
                            qt magic.
    '''

    # signals must be declared outside the constructor, for some weird reason
    _completion_signal = pyqtSignal(object)

    def __init__(self, model):
        '''Connect a signal to :py:meth:`model.Model.notifyObservers`.'''
        super().__init__(model)
        self._completion_signal.connect(lambda info: model.notifyObservers(info))

    def onCompletion(self, future):
        '''Emit the sompletion signal to have :py:meth:`model.Model.notifyObservers` called.'''
        self._completion_signal.emit(future.result())



class InputController(object):
    '''Base controller backed by a network. Contains functionality for
    manipulating input index.

    Attributes
    ----------
    _parent :   InputController
                Parent controller to bubble up events to (currently unused)

    '''

    _parent: 'InputController' = None

    def __init__(self, model):
        self._model = model
        self._runner = QTAsyncRunner(model)

    def setParent(self, controller: 'InputController'):
        '''Set the parent controller.

        Parameters
        ----------
        controller  :   controller.InputController
                        Parent controller to which stuff will be delegated.
        '''
        self._parent = controller

    def random(self):
        '''Select a random index into the dataset.'''
        n_elems = len(self._model)
        index = randint(0, n_elems)
        self._runner.runTask(self._model.editIndex, index)

    def advance(self):
        '''Advance data index to end.'''
        n_elems = len(self._model)
        self._runner.runTask(self._model.editIndex, n_elems - 1)

    def advance_one(self):
        '''Advance data index by one.'''
        current_index = self._model._current_index
        self._runner.runTask(self._model.editIndex, current_index + 1)

    def rewind(self):
        '''Reset data index to zero.'''
        self._runner.runTask(self._model.editIndex, 0)

    def rewind_one(self):
        '''Rewind data index by one.'''
        current_index = self._model._current_index
        self._runner.runTask(self._model.editIndex, current_index - 1)

    def editIndex(self, index: int):
        '''Set data index to specified value.

        Parameters
        ----------
        index   :   int
        '''
        self._runner.runTask(self._model.editIndex, index)
