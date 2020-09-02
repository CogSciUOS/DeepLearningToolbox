# standard imports
from typing import Callable, Iterable
import logging

# Qt imports
from PyQt5.QtCore import QObject, pyqtSignal, QThreadPool, QRunnable
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QWidget

# toolbox imports
from toolbox import Toolbox
from base import Observable, AsyncRunner
import util
from util.error import protect, handle_exception

# logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


class QtAsyncRunner(AsyncRunner, QObject):
    """:py:class:`AsyncRunner` subclass which knows how to update Qt widgets.

    The important thing about Qt is that you must work with Qt GUI
    only from the GUI thread (that is the main thread). The proper way
    to do this is to notify the main thread from worker and the code
    in the main thread will actually update the GUI.

    Another point is that there seems to be a difference between
    python threads and Qt threads (QThread). When interacting with Qt,
    always use QThreads.

    Attributes
    ----------
    _completion_signal: pyqtSignal
        Signal emitted once the computation is done.
        The signal is connected to :py:meth:`onCompletion` which will be run
        in the main thread by the Qt magic.
    """

    # signals must be declared outside the constructor, for some weird reason
    _completion_signal = pyqtSignal(Observable, object)

    def __init__(self):
        """Connect a signal to :py:meth:`Observable.notifyObservers`."""
        # FIXME[question]: can we use real python multiple inheritance here?
        # (that is just super().__init__(*args, **kwargs))
        AsyncRunner.__init__(self)
        QObject.__init__(self)
        self._completion_signal.connect(self._notifyObservers)

    def onCompletion(self, future):
        """Emit the completion signal to have
        :py:meth:`Observable.notifyObservers` called.

        This method is still executed in the runner Thread. It will
        emit a pyqtSignal that is received in the main Thread and
        therefore can notify the Qt GUI.
        """
        self._completed += 1
        try:
            result = future.result()
            if result is not None:
                observable, info = result
                import threading
                me = threading.current_thread().name
                LOG.debug(f"{self.__class__.__name__}.onCompletion():{info}")
                if isinstance(info, Observable.Change):
                    self._completion_signal.emit(observable, info)
        except BaseException as exception:
            handle_exception(exception)

    def _notifyObservers(self, observable, info):
        """The method is intended as a receiver of the pyqtSignal.
        It will be run in the main Thread and hence can notify
        the Qt GUI.
        """
        observable.notifyObservers(info)


class QDebug:  # FIXME[todo]: derive from (QWidget)
    """The :py:class:`QDebug` is a base class that classes can inherit
    from to provide some debug functionality.  Such classes should
    then implement a :py:meth:`debug` method, which will be called to
    output debug information.

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @protect
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Process key events. The :py:class:`QImageView` supports
        the following keys:

        ?: debug
        t: debug toolbox
        """
        key, text = event.key(), event.text()
        print(f"debug: QDebug[{type(self).__name__}].keyPressEvent:"
              f" key={key}, text={text}")
        if text == '?':
            self.debug()
        elif key == Qt.Key_T:  # Debug Toolbox
            Toolbox.debug_register()
        elif hasattr(super(), 'keyPressEvent'):
            super().keyPressEvent(event)

    def debug(self) -> None:
        print(f"debug: QDebug[{type(self).__name__}]. MRO:")
        print("\n".join([f"debug:   - {str(cls)}"
                         for cls in type(self).__mro__]))


from PyQt5.QtCore import QObject, QMetaObject, pyqtSlot, Q_ARG
from PyQt5.QtGui import QHideEvent, QShowEvent
from base.observer import Observable
from base import MetaRegister
import threading


class QAttribute(QDebug):
    """A :py:class:`QAttribute` can store attributes of specified types.
    It can also propagate changes of these attributes to interested
    parties, usually child widgets of this :py:class:`QAttribute`.

    Inheriting from :py:class:`QAttribute` allows to use a new class
    attribute `qattributes:dict ={}`. Keys of this dictionary are
    types of the attributes to be added to the class and the
    corresponding value describes how the attribute can be accessed:
    `False` means that the attribute can not be accessed (there is no
    getter), but it can be changed (there is a setter). Having an
    attribute that cannot be read may seem pointless, but makes sense
    in combination with attribute propagation (see below).
    The name of the attribute will be derived from the ClassName:
    the getter will be called className() and the setter is called
    setClassName(). There is also a private attribute _className
    for storing the attribute value.
    FIXME[todo]: we may allow to pass a name for the attribute

    Attribute propagation allows to propagate changes of the attribute
    to other objects (provided they implement an appropriate setter).
    The intended use is for containers, with elements interested in
    the same type of attribute.

    Examples
    --------

    Create a Widget with a `Toolbox` attribute (getter and setter):
    >>> class QMyWidget(QWidget, QObserver, qattributes={Toolbox: True})

    Create a Widget with a `Toolbox` write only attribute (only setter):
    >>> class QViewWidget(QWidget, QObserver, qattributes={Toolbox: False})
    """

    @classmethod
    def _qAttributeGetter(cls, name):
        attribute_name, getter_name = \
            cls._qAttributeNameFor(name, 'attribute', 'getter')
        LOG.debug("QAttribute: creating new attribute to %s.%s",
                  cls, attribute_name)
        setattr(cls, attribute_name, None)
        getter = lambda self, observable: getattr(self, attribute_name, None)
        LOG.debug("QAttribute: creating new getter to  %s.%s",
                  cls, getter_name)
        setattr(cls, getter_name, getter)

    @classmethod
    def _qAttributeSetter(cls, name):
        setter_name, orig_setter_name = \
            cls._qAttributeNameFor(name, 'setter', 'orig_setter')
        LOG.debug("QAttribute: creating new setter %s.%s",
                  cls, setter_name)
        if hasattr(cls, setter_name):
            setattr(cls, orig_setter_name, getattr(cls, setter_name))
            LOG.debug("QAttribute: moving old setter to %s.%s",
                      cls, orig_setter_name)
        setter = lambda self, *args, **kwargs: \
            self._qAttributeSetAttribute(name, *args, **kwargs)
        setattr(cls, setter_name, setter)

    @staticmethod
    def _qAttributeName(cls: type, *args) -> str:
        """Provide name(s) for an attribute of a given type.
        This will be the name provided by the class method
        :py:meth:`observable_name`, which defaults to the class name.
        """
        name = cls.observable_name()
        LOG.debug("QAttribute: Using attribute name '%s' for class '%s'",
                  name, cls.__name__)
        return QAttribute._qAttributeNameFor(name, *args)

    @staticmethod
    def _qAttributeNameFor(name: str, *args) -> Iterable[str]:
        if not args:
            return name
        return iter('_' + name[0].lower() + name[1:]
                    if what == 'attribute' else
                    name.lower() if what == 'getter' else
                    'set' + name if what == 'setter' else
                    '_qAttribute' + name if what == 'propagation' else
                    '_qAttributeSet' + name if what == 'orig_setter' else
                    name for what in args)

    def __init_subclass__(cls: type, qattributes={}):
        for attributeClass, createAttribute in qattributes.items():
            name = cls._qAttributeName(attributeClass)
            LOG.debug(f"  {cls.__name__}: "
                      f"Adding{' getter and' if createAttribute else ''}"
                      f" setter for name ({attributeClass}): ")
            cls._qAttributeSetter(name)
            if createAttribute:
                cls._qAttributeGetter(name)

    def _qAttributeSetAttribute(self, name: str, obj, *args, **kwargs):
        LOG.debug(f"_qAttributeSetAttribute({self.__class__.__name__}, "
                  f"'{name}', {type(obj)}, {args}, {kwargs}")

        attribute, setter, propagation, orig_setter = \
            self._qAttributeNameFor(name, 'attribute', 'setter',
                                    'propagation', 'orig_setter')

        # (0) check if the attribute has changed
        changed = (not hasattr(self, attribute) or
                   getattr(self, attribute) is not obj)
        LOG.debug("%s.%s(%s): self.%s: old={%s}, "
                  "changed=%s, propagations: %d, orig_setter: %s",
                  type(self).__name__, setter, obj, attribute,
                  getattr(self, attribute, '??'), changed,
                  len(getattr(self, propagation, ())),
                  hasattr(self, orig_setter))

        # (1) update the attribute if changed
        if hasattr(self, attribute) and changed:
            setattr(self, attribute, obj)

        # (2) propagate new attribute value (we do this even if the
        # attribute has not changed, as it may have changed for some
        # of the propagates)
        for target in getattr(self, propagation, ()):
            LOG.debug(f"{type(self).__name__}: "
                      f"propagate change of attribute '{name}' to "
                      f"{type(target).__name__}.{setter}({obj})")
            getattr(target, setter)(obj, *args, **kwargs)

        # (3) finally call the original setter (only if the attribute
        # has really changed)
        if hasattr(self, orig_setter) and changed:
            LOG.debug(f"{type(self).__name__}: "
                      f"calling original setter for attribute '{name}' as "
                      f"{type(self).__name__}.{orig_setter}({obj})")
            getattr(self, orig_setter)(obj, *args, **kwargs)

    def addAttributePropagation(self, cls: type, obj):
        propagation, = self._qAttributeName(cls, 'propagation')
        if hasattr(self, propagation):
            getattr(self, propagation).append(obj)
        else:
            setattr(self, propagation, [obj])

    def removeAttributePropagation(self, cls: type, obj):
        propagation, = self._qAttributeName(cls, 'propagation')
        if hasattr(self, propagation):
            getattr(self, propagation).remove(obj)


class QObserver(QAttribute):
    """This as a base class for all QWidgets that shall act as
    Observers in the toolbox. It implements support for asynchronous
    message passing to Qt's main event loop.

    This class provides a convenience method :py:meth:`observe`, which
    should be used to observe some :py:class:`Observable` (instead
    of calling :py:meth:`Observable.add_observer` directly). This will
    set up the required magic.

    Deriving from :py:class:`QObserver` will automatically define
    the following properties and methods for each observable
    listed in `qobservables`:

    * a property `observable()` providing a reference to the
      observable.

    * a method `setObervable(observable, observe=True)` allowing to
      set and observe that observable.

    * patched versions of :py:meth:`observe` and :py:meth:`unobserve`
      that allow for asynchronous observation in the Qt main event loop.


    Examples
    --------
    >>> class QMyWidget(QWidget, QObserver, qobservables={
    ...         Toolbox: {'network_changed'},
    ...         Network: {'state_changed'}}):
    ...
    ...     def __init__(self, toolbox: Toolbox = None, **kwargs):
    ...         super().__init__(**kwargs)
    ...         self.setToolbox(toolbox)
    ...
    ...     def toolboxChanged(self, toolbox: Toolbox, change: Toolbox.Change):
    ...         if change.network_changed:
    ...             self.setNetwork(toolbox.network)
    """

    # FIXME[concept]: we need some concept to assign observables to helpers
    #_qObserverHelpers: dict = None

    @classmethod
    def _qObserverSetter(cls, name, interests) -> None:
        # FIXME[hack]: here we replace the qAttributeSetter.
        # It would be nicer to find a way to set the correct setter
        # right from the start ...
        setter_name, = cls._qAttributeNameFor(name, 'setter')
        if cls.__name__ == 'QIndexControls':  # FIXME[debug]
            print(f" - {name}: {setter_name} - {interests}")
        setter = lambda self, observable, *args, **kwargs: \
            self._qObserverSetAttribute(name, observable, interests,
                                        *args, **kwargs)
        setattr(cls, setter_name, setter)

    def __init_subclass__(cls: type, qobservables={}, qattributes={}):
        if cls.__name__ == 'QIndexControls':  # FIXME[debug] 
            print(f"FIXME[old]: QObserver[{cls.__name__}] .__init_subclass__({cls}, {qobservables}, {qattributes}): {cls.__mro__}")
        if not qobservables:
            # FIXME[hack]: we can not really detect if observables
            # were registered in superclasses!  FIXME[todo]: but we
            # should - otherwise we could call our own super()-setter
            # as orig_setter, probably not what we want to do ...  as
            # an example use the Datasource buttons (QPrepareButton,
            # QLoopButton, ...), which form a hierarchy derived from
            # the same base class.
            print(f"FIXME[old]: QObserver.__init_subclass__({cls}, without qobservables")
            LOG.warning(f"FIXME[old]: QObserver.__init_subclass__({cls}, without qobservables")

        # Add attributes for all observable classes
        new_qattributes = dict(qattributes)
        for observableClass in qobservables.keys():
            new_qattributes[observableClass] = True

        # Treat all observables as attributes.
        super().__init_subclass__(qattributes=new_qattributes)

        # Now adjust the setter to be informed of the changes of interest
        for observableClass, interests in qobservables.items():
            change = observableClass.Change(interests)
            name = cls._qAttributeName(observableClass)
            cls._qObserverSetter(name, change)
            LOG.debug("Updated setter for %s (%s)", observableClass, change)

    def _qObservableHelper(self, observableClass: type):  # -> QObserverHelper:
        return self._qObservableHelpers.get(observableClass, None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._qObserverHelpers = {}

    def __del__(self):
        self.setVisible(False)
        #super().__del__()

    def helperForObservable(self, observable: Observable, interests=None,
                            notify: Callable = None, create: bool = False,
                            remove: bool = False) -> Observable.Observer:
        if self._qObserverHelpers is None:
            raise RuntimeError("It seems QObserver's constructor was not "
                               "called (QObserverHelpers is not set)")
        key = type(observable).Change
        if key not in self._qObserverHelpers:
            if not create:
                raise RuntimeError("No QObserverHelper for observing "
                                   f"{type(observable)} in {self}")
            self._qObserverHelpers[key] = \
                QObserver.QObserverHelper(self, observable, interests, notify)
        elif remove:
            return self._qObserverHelpers.pop(key)
        return self._qObserverHelpers[key]

    @protect
    def hideEvent(self, event: QHideEvent) -> None:
        """Hiding the widget will stop all observations.
        """
        for qObserverHelper in self._qObserverHelpers.values():
            qObserverHelper.stopObservation()

    @protect
    def showEvent(self, event: QShowEvent) -> None:
        """Showing a widget will resume all stopped observations.
        """
        # FIXME[bug]: here the following error can occur:
        # RuntimeError: dictionary changed size during iteration
        for qObserverHelper in self._qObserverHelpers.values():
            qObserverHelper.startObservation()

    def observe(self, observable: Observable, interests=None,
                notify: Callable = None) -> None:
        qObserverHelper = self.helperForObservable(observable, create=True)
        qObserverHelper.startObservation(interests=interests, notify=notify)

    def unobserve(self, observable, remove: bool = True):
        qObserverHelper = self.helperForObservable(observable, remove=True)
        qObserverHelper.stopObservation()

    def _registerView(self, name, interests=None):
        # FIXME[question]: do we have to call this from the main thread
        # if not threading.current_thread() is threading.main_thread():
        #     raise RuntimeError("QObserver._registerView must be called from"
        #                        "the main thread, not " +
        #                        threading.current_thread().name + ".")
        if interests is not None:
            key = interests.__class__
            if key not in self._qObserverHelpers:
                # print(f"INFO: creating a new QObserverHelper for {interests}: {interests.__class__}!")
                self._qObserverHelpers[key] = \
                    QObserver.QObserverHelper(self, interests)
        else:
            print(f"WARNING: no interest during registration of {name} for {self} - ignoring registration!")

    def _qObserverSetAttribute(self, name: str, new_observable, interests=None,
                               *args, **kwargs):
        """Exchange an observable. This implies stopping observing the
        old observable and starting to observe the new one.
        """
        LOG.debug(f"_qObserverSetAttribute({self.__class__.__name__}, "
                  f"'{name}', {type(new_observable)}, {interests}, {args}, "
                  f"{kwargs}")

        # FIXME[debug]: "setter". It seems that this setter is called a
        #     bit too often, sometimes with identical arguments (button
        #     interestingly with different interests).
        # debug_class = 'QIndexControls'
        # if type(self).__name__ == debug_class:  # FIXME[debug]
        #     print(f"[setter] 1: _qObserverSetAttribute('{name}', "
        #           f"{type(new_observable).__name__}[at {id(new_observable)}]): {interests}")
        #     import traceback
        #     traceback.print_stack()
        #     print("-------------------------------------------------")

        attribute, = self._qAttributeNameFor(name, 'attribute')

        old_observable = getattr(self, attribute, None)
        if new_observable is old_observable:
            # FIXME[debug]: "setter". see above.
            # if type(self).__name__ == debug_class:
            #     print("[setter] 2a: _qObserverSetAttribute: "
            #           "new observable is old observable")
            return  # nothing to do ...

        # FIXME[debug]: "setter". see above.
        # if type(self).__name__ == debug_class:
        #     print(f"[setter] 2b: observe {interests}")

        if old_observable is not None:
            self.unobserve(old_observable)
        self._qAttributeSetAttribute(name, new_observable, *args, **kwargs)
        if new_observable is not None:
            self.observe(new_observable, interests=interests)

    def _exchangeView(self, name: str, new_view, interests=None):
        """Exchange the object (View or Controller) being observed.
        This will store the object as an attribute of this QObserver
        and add this QObserver as an oberver to our object of interest.
        If we have observed another object before, we will stop
        observing that object.

        name: str
            The name of the attribute of this object that holds the
            reference.
        new_view: View
            The object to be observed. None means to not observe
            anything.
        interests:
            The changes that we are interested in.
        """
        old_view = getattr(self, name)
        if new_view is old_view:
            return  # nothing to do
        if old_view is not None:
            old_view.remove_observer(self)
            # -> this will call self.unobserve(old_view)
        setattr(self, name, new_view)
        if new_view is not None:
            new_view.add_observer(self, interests)
            # -> this will call self.observe(new_view, interests)

    def debug(self) -> None:
        super().debug()
        print(f"debug: QObserver[{type(self).__name__}]: "
              f"{len(self._qObserverHelpers)} observations:")
        for i, (key, helper) in enumerate(self._qObserverHelpers.items()):
            print(f"debug:   ({i}) {key.__name__.split('.')[-2]}: "
                  f"{helper}")

    # FIXME[bug]:
    #   File "base/observer.py", line 314, in __del__
    #      AttributeError: 'QObserverHelper' object has no attribute 'unobserve'
    class QObserverHelper(QObject):
        """A helper class for the :py:class:`QObserver`.

        We define the functionality in an extra class to avoid
        problems with multiple inheritance (:py:class:`QObserver` is
        intended to be inherited from in addition to some
        :py:class:`QWidget`): The main problem with QWidgets and
        multiple inheritance is, that signals and slots are only
        inherited from the first super class (which will usually be
        QWidget or one of its subclasses), but we have to define our
        own pyqtSlot here to make asynchronous notification work.

        This class has to inherit from QObject, as it defines an
        pyqtSlot.

        This class has the following attributes:

        _observer: Observer
            The actuall Observer that finally should receive the
            notifications.
        _interests: Change
            The changes, the Observer is interested in.
        _change: Change
            The unprocessed changes accumulated so far.
        """

        def __init__(self, observer: Observable.Observer,
                     observable: Observable, interests=None,
                     notify: Callable=None, **kwargs) -> None:
            """Initialization of the QObserverHelper.
            """
            super().__init__(**kwargs)
            self._observer = observer
            self._observable = observable
            self._observing = False
            self._interests = interests
            self._change = None  # accumulated changes
            self._kwargs = {}  # additional arguments provided on notification
            self._notify = notify

        def startObservation(self, interests=None,
                             notify: Callable=None) -> None:
            if self._observing:
                return  # The observation was already started
            self._observing = True
            if interests is not None:
                self._interests = interests
            if notify is not None:
                self._notify = notify
            # FIXME[hack/bug]: this should be: notify=self._qNotify ...
            self._observable.add_observer(self, notify=
                                          QObserver.QObserverHelper._qNotify,
                                          interests=self._interests)
            self.__class__._qNotify(self._observable, self,
                                    self._interests)

        def stopObservation(self) -> None:
            if not self._observing:
                return  # No ongoing observation to stop.
            self._observing = False
            self._observable.remove_observer(self)

        # FIXME[hack/bug]: make this a method, take self as first argument
        # something like 'observable_changed(self, observable, change)'
        # Then QObserverHelper could be made an Observer,
        # e.g. called QObservableWrapper
        def _qNotify(observable, self, change, **kwargs):
            """Notify the Observer about some change. This will
            initiate an asynchronous notification by queueing an
            invocation of :py:meth:_qAsyncNotify in the Qt main event
            loop. If there are already pending notifications, instead
            of adding more notifications, the changes are simply
            accumulated.

            Arguments
            ---------
            observable: Observable
                The observable that triggered the notification.
            self: QObserverHelper
                This QObserverHelper.
            change:
                The change that caused this notification.
            """
            if self._change is None:
                # Currently, there is no change event pending for this
                # object. So we will queue a new one and remember the
                # change:
                self._change = change
                self._kwargs = kwargs
                # This will use Qt.AutoConnection: the member is invoked
                # synchronously if obj lives in the same thread as the caller;
                # otherwise it will invoke the member asynchronously.
                QMetaObject.invokeMethod(self, '_qAsyncNotify',
                                         Q_ARG("PyQt_PyObject", observable))
                #     Q_ARG("PyQt_PyObject", change))
                #   RuntimeError: QMetaObject.invokeMethod() call failed
            else:
                # There is already one change pending in the event loop.
                # We will just update the change, but not queue another
                # event.
                self._change |= change

        @pyqtSlot(object)
        def _qAsyncNotify(self, observable):
            """Process the actual notification. This method is to be invoked in
            the Qt main event loop, where it can trigger changes to
            the graphical user interface.

            This method will simply call the :py:meth:`notify` method
            of the :py:class:`Observable` to notify the actual
            observer.
            """
            change = self._change
            kwargs = self._kwargs
            if change is not None:
                try:
                    self._change = None
                    self._kwargs = {}
                    if self._notify is None:
                        observable.notify(self._observer, change, **kwargs)
                    else:
                        self._notify(observable, change, **kwargs)
                except Exception as ex:
                    util.error.handle_exception(ex)

        def __str__(self) -> str:
            #f"QObserver({self._observer};"
            #f"interests={self._interests}, change={self._change})"
            return (f"observable={self._observable}, "
                    f"observing={self._observing}, "
                    f"notify={self._notify}")


class QThreadedUpdate(QWidget):
    """A class supporting the implementation of threaded update methods
    using the `@qupdate` decorator.

    Methods decorated with `@qupdate` will be executed in a separate
    thread if currently no other update operation is running. Otherwise,
    it is queued for later execution. If another update is issued in
    the meantime, the former is dropped and only the last update
    is kept in the queue. The ratio is, that only the most recent update
    is relevant, as it will invalidate all previous updates.
    """
    # FIXME[todo]: we may think if a locking mechanism makes sense here

    class QUpdater(QRunnable):
        """
        """

        def __init__(self, target, obj) -> None:
            super().__init__()
            self._target = target
            self._object = obj

        @pyqtSlot()
        def run(self):
            self._target(self._object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._updating = False
        self._nextUpdate = None

    def update(self) -> None:
        super().update()
        if self._nextUpdate is None:
            self._updating = False
        else:
            self._updating = True
            QThreadPool.globalInstance().start(self._nextUpdate)
            self._nextUpdate = None


def pyqtThreadedUpdate(method):
    """A decorator for threaded update functions. It can be used to
    decorate update methods in subclases of
    :py:class:`QThreadedUpdate`. An update method is a method that
    does some (complex) updates to a widget (e.g. prepare some graphics)
    and then calls self.update() to trigger a repaint event that
    than can display the updated content. When decorated with this
    decorator, the update method will be executed in a background
    thread, hopefully resulting in a smoother user experience, especially
    if the update method is called from the main event loop (e.g. by
    an event handler).
    """

    def closure(self) -> None:
        nextUpdate = QThreadedUpdate.QUpdater(method, self)
        if self._updating:
            self._nextUpdate = nextUpdate
        else:
            self._updating = True
            QThreadPool.globalInstance().start(nextUpdate)

    return closure


import os
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QMovie

from base import BusyObservable


class QBusyWidget(QLabel, QObserver, qobservables={
        BusyObservable: {'busy_changed'}}):
    """A widget indicating a that some component is busy.
    """
    _label: QLabel = None
    _movie: QMovie = None
    _busy = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Animated gifs can be obtained from
        # http://www.ajaxload.info/#preview
        self._movie = QMovie(os.path.join('assets', 'busy.gif'))
        self.setMovie(self._movie)
        self.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        #self._movie.start()

    def setView(self, busyView):
        interests = BusyObservable.Change('busy_changed')
        self._exchangeView('_busy', busyView, interests=interests)

    def detector_changed(self, busyBody: BusyObservable,
                         change: BusyObservable.Change) -> None:
        """FIXME[hack]: we should have a callback busy_changed!
        """
        #print("QBusyWidget.detector_changed")
        #self._movie.setPaused(not busyBody.busy)
        #self.setVisible(busyBody.busy)

    def __del__(self):
        self._movie.stop()
        del self._movie
