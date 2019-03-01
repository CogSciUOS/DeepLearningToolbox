
from base.observer import Observable, change
#from asyncio import Semaphore
from threading import Semaphore

class Toolbox(Semaphore, Observable,
              changes=['lock_changed'],
              default='lock_changed',
              method='toolboxChanged'):

    def __init__(self):
        Semaphore.__init__(self, 1)
        Observable.__init__(self)

    def acquire(self):
        result = super().acquire()
        self.change('lock_changed')
        return result

    def release(self):
        super().release()
        self.change('lock_changed')

    def locked(self):
        return (self._value == 0)

toolbox = Toolbox()
