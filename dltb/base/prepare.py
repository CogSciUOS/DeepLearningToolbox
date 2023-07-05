"""
.. moduleauthor:: Ulf Krumnack

.. module:: base.prepare

This module contains abstract definitions of the :py:class:`Preparable`
interface.  The idea of preparation is to split the initialization of an
object into two steps:

(1) Setting up an object: create a stable state (initializing all
attributes to default values, usually None). This is usually
considered to be a qick operation which should not raise any exception
and is performed by the :py:meth:`__init__` method of the object.  /

(2) Preparing the object: set the object in a useful state. This may
include the import of additional modules, loading data, initializing
hardware, etc. This operation may be complex, taking some time to
complete and hence should in certain settings be run in a background
Thread. It may also fail if resources are not available.

Classes that realize the idea of preparation should inherit from
:py:class:`Preparable` and overwrite the relevant methods.
"""
# standard imports
from abc import ABC
import logging

# toolbox imports
from ..config import config
from .busy import BusyObservable, busy
from .meta import Postinitializable

# logging
LOG = logging.getLogger(__name__)


class Preparable(BusyObservable, Postinitializable, # ABC,
                 method='preparable_changed', changes={'state_changed'}):
    """The :py:class:`Preparable` implements this idea providing three public
    methods:

    :py:meth:`prepare`:
        Prepare the object by allocating all resources required
        to use the object.

    :py:meth:`unprepare`:
        Release all resource acquired by :py:meth:`prepare`.

    :py:meth:`prepared`:
    A boolean property indicating if the the object was prepared
    (successfully).

    Classes that realize the idea preparation should inherit from
    :py:class:`Preparable` and reimplement the corresponding private
    methods :py:meth:`_prepare`, :py:meth:`_unprepare` and
    :py:meth:`_prepared`.

    """

    def __init__(self, prepare: bool = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if prepare is not None:
            self._prepare_on_init = prepare

    def __post_init__(self) -> None:
        super().__post_init__()
        if getattr(self, '_prepare_on_init', config.prepare_on_init):
            self.prepare()

    def __del__(self) -> None:
        """Before deleting an object make sure it is unprepared.
        Unpreparing frees all resources that may be aquired by
        this object.
        """
        self._unprepare()
        super().__del__()

    def __enter__(self) -> None:
        """The context manager for :py:class:`Preparable` objects allows
        to use such objects in a prepared state.
        """
        # FIXME[todo]: check if super().__enter__() has to be called ...
        result = self  # super().__enter__()
        if not hasattr(self, '_prepare_entered') and not self.prepared:
            self._prepare_entered = True
            self.prepare()
        return result

    def __exit__(self, _exception_type, _exception_value, _traceback) -> None:
        """Exit the runtime context related to this :py:class:`Preparable`.
        Only if the object was prepared when entering the context, it
        will be unprepared here, otherwise it will stay prepared.
        """
        if hasattr(self, '_prepare_entered'):
            self.unprepare()
            delattr(self, '_prepare_entered')

    @property
    def prepared(self) -> bool:
        """Report if this :py:class:`Preparable` is prepared for use.
        Subclasses with their own preparation implementation should
        not overwrite this property but rather the private method
        :py:meth:`_prepared`.
        """
        return self._prepared()

    def _prepared(self) -> bool:
        """Implementation of the :py:class:`prepared` property.
        The default implementation will always return `True`.
        Subclasses should overwrite this method to adapt to their
        individual implementation. When doing so, they may form
        the conjunction of `super()._prepared()` and their own result
        to allow for deep class hierarchies and multiple inheritance.
        """
        return True  # to be implemented by subclasses

    # FIXME[old]: @change
    # FIXME[todo]: a better implementation would be to only
    # become busy (which may include starting a new Thread) if
    # we are not already prepared ...
    @busy("preparing")
    def prepare(self, force: bool = False) -> None:
        """Prepare this :py:class:`Preparable` for use.
        Preparation will allocate all resources required to use the
        object. This may take some time and is also prone to errors.

        Subclasses that want to provide their own preparation
        mechanism should not overwrite this method directly but
        rather the private method :py:meth:`_prepare`.
        """
        if self.prepared:
            return  # nothing to do ...

        if not self.preparable and not force:
            raise RuntimeError(f"{self} ({type(self)}) is not preparable.")

        try:
            with self.failure_manager(cleanup=self._unprepare, catch=True):
                # FIXME[todo]: catch only if running in own thread
                self._prepare()
                self._post_prepare()
        finally:
            self.change('state_changed')

    def _prepare(self) -> None:
        """Implemenation of the :py:meth:`prepare` method.  When called, it is
        guaranteed that this object is not in the prepared state. The
        method may be run asynchronously in a background thread.

        The default implementation does nothing. Subclasses that
        provide their own preparation mechanism should overwrite this
        method (and also provide corresponding :py:meth:`_unprepare`
        and :py:meth:`_prepared` implementations). When doing so, the
        first step should be to call `super()._prepare()` to allow for
        deep class hierarchies and multiple inheritance.

        """
        # To be implemented by subclasses

    def _post_prepare(self) -> None:
        """An additional action that shall be performed upon successful
        preparation. The default implementation does nothing, but
        subclasses may overwrite this method to perform some initial
        action with the freshly prepared object.
        """
        # To be implemented by subclasses

    @busy("unpreparing")
    def unprepare(self) -> None:
        """Free resources allocated by the preparation procedure.  After
        preparation, the object is no longer in a useful state, but it
        should be possible to revive it by invoking :py:meth:`prepare`
        again.

        If the object was not prepared, nothing will be done.

        Subclasses that want to provide their own unpreparation
        mechanism should not overwrite this method directly but
        rather the private method :py:meth:`_unprepare`.
        """
        if not self.prepared:
            return

        self._pre_unprepare()
        self._unprepare()
        self.change('state_changed')

    def _pre_unprepare(self) -> None:
        """An additional action that shall be performed before
        unpreparation. The default implementation does nothing, but
        subclasses may overwrite this method to perform some
        action before the resources are released. The method should
        not release any resources itself.
        """

    def _unprepare(self) -> None:
        """Implementation of the :py:meth:`unprepare` operation. This will
        free resources allocated by this :py:class:`Preparable`. When
        called, it is not guaranteed that this object is in a prepared
        or even consistent state - it may also find some inconsistent
        state caused by a failed preparation. It should also be
        considered that this method may be run asynchronously in a
        background thread.

        The default implementation does nothing. Subclasses that
        provide their own preparation mechanism should overwrite this
        method. When doing so, the last step should be to call
        `super()._unprepare()` to allow for deep class hierarchies and
        multiple inheritance.

        """
        # To be implemented by subclasses

    @property
    def preparable(self) -> bool:
        """Check if this :py:class:`Preparable` can be prepared. This
        is intended to be a first sanity check - the actual preparation
        may still fail even if this method returns `True`.
        """
        return self.prepared or self._preparable()

    def _preparable(self) -> bool:
        """The actual implementation of :py:meth:`preparable`. The default
        is `True` and subclasses with deviating behaviour should overwrite
        this method (combining their state `with super._preparable()`).
        """
        return True
