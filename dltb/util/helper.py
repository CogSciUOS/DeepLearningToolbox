"""Several commonly used general purpose functions, classes, decorators,
and context managers, only based on Python standard libraries.
"""

# standard imports
from contextlib import contextmanager
from threading import Lock


class classproperty1(property):
    """A docorator to mark a method as classproperty.

    This only allows for read access, it does not prevent overwriting.
    """

    def __get__(self, cls, owner):
        # cls=None, owner=class of decorated method
        return classmethod(self.fget).__get__(None, owner)()


class classproperty2(staticmethod):
    """Another way to define a classproperty docorator.

    This also only allows for read access, it does not prevent overwriting.
    """
    def __get__(self, cls, owner):
        # cls=None, owner=class of decorated method
        return self.__func__(owner)


classproperty = classproperty1


@contextmanager
def nonblocking(lock: Lock):
    """A contextmanager that does something with the given `Lock`,
    but only if it is not currently locked.

        lock = threading.Lock()
        with nonblocking(lock):
            ...

    """
    locked = lock.acquire(blocking=False)
    # if not locked:
    #     raise RuntimeError("Lock cannot be acquired.")
    try:
        yield locked
    finally:
        if locked:
            lock.release()
