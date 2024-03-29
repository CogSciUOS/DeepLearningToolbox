# standard imports
from typing import Callable, Iterable, Dict, Optional
import os
import logging

# Qt imports
from PyQt5.QtCore import Qt, QObject, QEvent, QThread, QThreadPool, QRunnable
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QMetaObject, Q_ARG
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QHideEvent, QShowEvent
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QApplication
import sip

# toolbox imports
from dltb.base.observer import Observable
from dltb.base.busy import BusyObservable
from dltb.base.prepare import Preparable
from dltb.util.error import protect, handle_exception
from toolbox import Toolbox
from base import AsyncRunner

# logging
LOG = logging.getLogger(__name__)


def qtName(name: str, lower: bool = True) -> str:
    """Transform a name into the Qt naming style (camelCase).
    """
    first, rest = name.split('_', maxsplit=1)
    if lower and not first[0].islower():
        first = first[0].lower() + first[1:]
    elif not lower and first[0].islower():
        first = first[0].upper() + first[1:]
    return (first if not rest else
            first + (''.join(s.capitalize() or '_' for s in rest.split('_'))))


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
        """Connect a signal to :py:meth:`Observable.notify_observers`."""
        # FIXME[question]: can we use real python multiple inheritance here?
        # (that is just super().__init__(*args, **kwargs))
        AsyncRunner.__init__(self)
        QObject.__init__(self)
        self._completion_signal.connect(self._notify_observers)

    def onCompletion(self, future):
        """Emit the completion signal to have
        :py:meth:`Observable.notify_observers` called.

        This method is still executed in the runner Thread. It will
        emit a pyqtSignal that is received in the main Thread and
        therefore can notify the Qt GUI.
        """
        self._completed += 1
        try:
            result = future.result()
            # FIXME[what]: what is info, what is observable?
            if result is not None:
                LOG.debug(f"{self.__class__.__name__}.onCompletion():{info}")
                if isinstance(info, Observable.Change):
                    self._completion_signal.emit(observable, info)
        except BaseException as exception:
            handle_exception(exception)

    def _notify_observers(self, observable, info):
        """The method is intended as a receiver of the pyqtSignal.
        It will be run in the main Thread and hence can notify
        the Qt GUI.
        """
        observable.notify_observers(info)


class QDebug:
    """The :py:class:`QDebug` is a base class that classes can inherit
    from to provide some debug functionality.  Such classes should
    then implement a :py:meth:`debug` method, which will be called to
    output debug information.

    Note: when deriving from this class, this class should be put
    before and :py:class:`QWidget` in the list of base classes as
    otherwise the `QWidget` will catch away all events.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.installEventFilter(self)

    @protect
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Process key events. The :py:class:`QImageView` supports
        the following keys:

        ?: debug
        t: debug toolbox
        """
        key, text = event.key(), event.text()
        print(f"debug: QDebug[{type(self).__name__}].keyPressEvent: "
              f"key={key}, text={text}")
        if text == '?':
            self.debug()
        elif key == Qt.Key_T:  # Debug Toolbox
            Toolbox.debug_register()
        elif hasattr(super(), 'keyPressEvent'):
            super().keyPressEvent(event)
        else:
            event.ignore()

    @protect
    def mousePressEventFIXME(self, event: QMouseEvent) -> None:
        """
        """
        button = event.button()
        modifiers = event.modifiers()
        print(f"debug: QDebug[{type(self).__name__}].mousePressEvent: "
              f"button={button}, modifiers={modifiers}")
        event.ignore()

    @protect
    def eventFilter(self, object: object, event: QEvent) -> bool:
        """
        """
        # print(f"debug: QDebug[{type(self).__name__}].eventFilter: "
        #       f"event={type(event)}, object={type(object)}, "
        #       f"type={event.type()}")
        if event.type() == QEvent.MouseButtonPress:
            # event is of type QMouseEvent
            modifiers = event.modifiers()
            if modifiers & Qt.ControlModifier:
                button = event.button()
                print(f"debug: QDebug[{type(self).__name__}].eventFilter: "
                      f"object={type(object).__name__}, button={button}, "
                      f"shift={bool(modifiers & Qt.ShiftModifier)}, "
                      f"control={bool(modifiers & Qt.ControlModifier)}"
                      f"alt={bool(modifiers & Qt.AltModifier)}")
                self.debug()
        return False

    def debug(self) -> None:
        """Output debug information for this :py:class:`QDebug`.
        """
        print(f"debug: QDebug[{type(self).__name__}]. MRO:")
        print("\n".join([f"debug:   - {str(cls)}"
                         for cls in type(self).__mro__]))


class QAttribute:
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

    _qattributes = {}

    @classmethod
    def _qAttributeGetter(cls, name: str) -> None:
        """Add a (private) attribute to the given class and create
        (public) getter that allows to read out the attribute
        value.

        Arguments
        ---------
        name: str
            The name for the attribute.
        """
        attributeName, getterName = \
            cls._qAttributeNameFor(name, 'attribute', 'getter')
        LOG.debug("QAttribute: creating new attribute to %s.%s",
                  cls, attributeName)

        # add the attribue to the class with default value None.
        setattr(cls, attributeName, None)

        # create a new getter and add it to the class.
        getter = lambda self: getattr(self, attributeName, None)
        LOG.debug("QAttribute: creating new getter to  %s.%s",
                  cls, getterName)
        setattr(cls, getterName, getter)

    @classmethod
    def _qAttributeSetter(cls, name):
        """Create a setter for an attribute an add it to the class.
        This setter will just call :py:meth:`self._qAttributeSetAttribute`
        with appropriate arguments.

        If there is already some setter for this attribute defined
        in the class, it will be renamed. The new setter will call
        this original setter after it has done its internal work.

        Arguments
        ---------
        name: str
            The name for the attribute.
        """
        setter_name, orig_setter_name = \
            cls._qAttributeNameFor(name, 'setter', 'orig_setter')
        LOG.debug("QAttribute: creating new setter %s.%s",
                  cls, setter_name)
        if hasattr(cls, setter_name):
            setattr(cls, orig_setter_name, getattr(cls, setter_name))
            LOG.debug("QAttribute: moving old setter to %s.%s",
                      cls, orig_setter_name)
        setattr(cls, setter_name, lambda self, *args, **kwargs:
                self._qAttributeSetAttribute(name, *args, **kwargs))

    @staticmethod
    def _qAttributeName(cls: type, *args) -> str:
        """Provide name(s) for an attribute of a given type.
        This will be the name of the class.
        """
        name = cls.__name__
        LOG.debug("QAttribute: Using attribute name '%s' for class '%s'",
                  name, cls)
        return QAttribute._qAttributeNameFor(name, *args)

    @staticmethod
    def _qAttributeNameFor(name: str, *args) -> Iterable[str]:
        if not args:
            return name
        lower = name[0].lower() + name[1:]
        return iter('_' + lower
                    if what == 'attribute' else
                    lower if what == 'getter' else
                    'set' + name if what == 'setter' else
                    '_qAttribute' + name if what == 'propagation' else
                    '_qAttributeSet' + name if what == 'orig_setter' else
                    name for what in args)

    @classmethod
    def _addQAttribute(cls, attributeClass: type, full: bool = True) -> None:
        """Add an attribute this class.

        This method is intended to be called upon class creation, e.g.
        from :py:meth:`__init_subclass__`. Later invocation is not
        recommended.

        Arguments
        ---------
        attributeClass: type
            The type of the attribute to be added. The name of the
            attribute is derived from the class name.
        full: bool
            A boolean value that indicates if this is a full attribute
            (`True`: getter and setter), or only a forwarder
            (`False`: just a setter).

        """
        name = cls._qAttributeName(attributeClass)
        if attributeClass in cls._qattributes:
            if full < cls._qattributes[attributeClass]:
                raise ValueError("Cannot add '{name}' as forwarder "
                                 "attribute as it is already registered "
                                 "as full attribute.")
            if full == cls._qattributes[attributeClass]:
                return  # nothing to do

        LOG.info("%s: Adding %s for %s.",
                 cls, 'getter and setter' if full else 'setter', name)
        if attributeClass not in cls._qattributes:
            cls._qAttributeSetter(name)
        if full:
            cls._qAttributeGetter(name)

        # add the new qattribute to the class' qattribute register
        if '_qattributes' not in cls.__dict__:  # make sure to use own register
            cls._qattributes = dict(cls._qattributes)
        cls._qattributes[attributeClass] = full

    def __init_subclass__(cls, qattributes: Dict[type, bool] = None,
                          **kwargs) -> None:
        """Intialized a new subclasses of :py:class:`QAttribute`.
        This will add attributes getters and setters to the new class.

        Arguments
        ---------
        qattributes: Dict[type, bool]
            A dictionary, have the the types of the attributes to
            be added as keys and boolean value indicating if this
            is a full attribute. This values are passed to
            :py:meth:`_addQAttribute` for creating the
            attributes.
        """
        super().__init_subclass__(**kwargs)
        if qattributes is not None:
            for attributeClass, full in qattributes.items():
                cls._addQAttribute(attributeClass, full)

    def _qAttributeSetAttribute(self, name: str, obj, *args, **kwargs):
        """A generic attribute setter which is used to realize
        the setters for attributes added by :py:meth:`_addQAttribute`.
        """
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
        if changed and hasattr(self, orig_setter):
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

    _qobservables = {}
    # FIXME[concept]: we need some concept to assign observables to helpers
    # _qObserverHelpers: dict = None

    @classmethod
    def _qObserverSetter(cls, name, interests, observableClass) -> None:
        # FIXME[hack]: here we replace the qAttributeSetter.
        # It would be nicer to find a way to set the correct setter
        # right from the start ...
        setter_name, = cls._qAttributeNameFor(name, 'setter')
        setter = lambda self, observable, *args, **kwargs: \
            self._qObserverSetAttribute(name, observable, interests,
                                        notify=observableClass.
                                        notify_method(self),
                                        *args, **kwargs)
        setattr(cls, setter_name, setter)

    @classmethod
    def _addQObservable(cls, observableClass: type, interests) -> None:
        # FIXME[todo]: does nothing, only performs sanity checks an
        # updates the register ...
        if observableClass in cls._qobservables:
            if interests < cls._qobservables[observableClass]:
                raise ValueError("Illegal attempt to to reduce the "
                                 f"interests for observable {observableClass} "
                                 f"in class {cls} from "
                                 f"{cls._qobservables[observableClass]} "
                                 f"to {interests}")
            if interests == cls._qobservables[observableClass]:
                return  # nothing to do

        # add the new qobservable to the class' qobservable register
        if '_qobservables' not in cls.__dict__:  # ensure using own register
            cls._qobservables = dict(cls._qobservables)
        cls._qobservables[observableClass] = interests

    def __init_subclass__(cls, qobservables: dict = None,
                          qattributes: dict = None, **kwargs) -> None:
        """Initialization of subclasses of :py:class:`QObserver`.

        Arguments
        ---------
        qobservables: dict
            :py:class:`Observable` classes that can be observed
            by instances of this py:class:`QObserver`.  Associated
            values are the changes that the observer is interested in.
            Each observable is added as a full qattribute, meaning
            that a private property and a getter/setter pair are
            automatically added.
        """
        # FIXME[bug]: interference of qattributes and qobservables:
        # if a class has qattributes and qobservables, both will be
        # treated as qobservables ()
        
        if qobservables is not None:
            for observableClass, interests in qobservables.items():
                cls._addQObservable(observableClass, interests)

        if not cls._qobservables:
            LOG.warning("Subclassing QObserver to %s without providing any "
                        "qobservables", cls)

        # Add attributes for all observable classes
        new_qattributes = {} if qattributes is None else dict(qattributes)
        if qobservables is None:
            qobservables = {}
        for observableClass in qobservables.keys():
            new_qattributes[observableClass] = True

        # Treat all observables as attributes.
        super().__init_subclass__(qattributes=new_qattributes, **kwargs)

        # Now adjust the setter to be informed of the changes of interest
        for observableClass, interests in qobservables.items():
            change = observableClass.Change(interests)
            name = cls._qAttributeName(observableClass)
            cls._qObserverSetter(name, change, observableClass)
            LOG.debug("Updated setter for %s (%s)", observableClass, change)

    def _qObservableHelper(self, observableClass: type):  # -> QObserverHelper:
        """Obtain the :py:class:`QObserverHelper` for the given type
        of :py:class:`Obervable`.
        """
        return self._qObservableHelpers.get(observableClass, None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._qObserverHelpers = {}

    def __del__(self):
        # FIXME[bug]: it may happen that this is called even after the
        # wrapped C/C++ object has already be deleted.
        # self.setVisible(False)
        pass
        # super().__del__()  # FIXME[todo]: check how to del QWidgets ...

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
            helper = QObserver.QObserverHelper(self, observable,
                                               interests, notify)
            # make sure that event processing for the helper object
            # is done in th main thread:
            mainThread = QApplication.instance().thread()
            if mainThread != QThread.currentThread():
                LOG.warning("Moving observer helper for %s to main Thread",
                            observable)
                helper.moveToThread(mainThread)
            self._qObserverHelpers[key] = helper
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

    def observe(self, observable: Observable,
                interests: Observable.Change = None,
                notify: Callable = None) -> None:
        """Observe an :py:class:`Observable`.

        Note: The type of the observable (currently) has to be
        suitable for this class, i.e. it is not necessary to declare
        it with `qobservables` arguments at class definition, but
        that may change in future.

        Arguments
        ---------
        observable: Observable
            The object to observer.
        interests:
            Notification we are interested in.
        notify: Callable
            The method to invoke on notification. If none is provided,
            the default notifiaction method of the observable is used.

        Raises
        ------
        TypeError:
            The observable is of inapropriate type.
        """
        if not isinstance(observable, Observable):
            raise TypeError(f"{type(self).__name__} is trying to observe "
                            f"a non-observable ({type(observable)})")
        mainThread = QApplication.instance().thread()
        currentThread = QThread.currentThread()
        qObserverHelper = \
            self.helperForObservable(observable, create=True,
                                     interests=interests, notify=notify)
        qObserverHelper.startObservation(interests=interests, notify=notify)

    def unobserve(self, observable, remove: bool = True):
        """Stop an ongoing observation.

        Arguments
        ---------
        observable: Observable
            The observable the should no longer be observed.
        remove: bool
            A flag indicating if the internal oberservation structure
            for that observable should be removed. Removal is the
            default behahviour, but if it is likely that the observable
            will be observed again, it can be worth keep that structure.
        """
        qObserverHelper = self.helperForObservable(observable, remove=True)
        qObserverHelper.stopObservation()

    def _registerView(self, name, interests=None):
        # FIXME[question]: do we have to call this from the main thread
        # if not threading.current_thread() is threading.main_thread():
        #     raise RuntimeError("QObserver._registerView must be called from"
        #                        "the main thread, not " +
        #                        threading.current_thread().name + ".")
        print("FIXME[old]: _registerView - should not be used anymore")
        if interests is not None:
            key = interests.__class__
            if key not in self._qObserverHelpers:
                # print(f"INFO: creating a new QObserverHelper for {interests}: {interests.__class__}!")
                self._qObserverHelpers[key] = \
                    QObserver.QObserverHelper(self, interests)
        else:
            print(f"WARNING: no interest during registration of {name} for {self} - ignoring registration!")

    def _qObserverSetAttribute(self, name: str, new_observable, interests=None,
                               notify=None, *args, **kwargs):
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
            return  # nothing to do ...

        if old_observable is not None:
            self.unobserve(old_observable)
        self._qAttributeSetAttribute(name, new_observable, *args, **kwargs)
        if new_observable is not None:
            self.observe(new_observable, interests=interests, notify=notify)

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
        if hasattr(super(), 'debug'):
            super().debug()
        print(f"debug: QObserver[{type(self).__name__}]: "
              f"{len(self._qObserverHelpers)} observations:")
        for i, (key, helper) in enumerate(self._qObserverHelpers.items()):
            print(f"debug:   ({i}) {key.__name__.split('.')[-2]}: "
                  f"{helper}")

    class QObserverHelper(QObject, Observable.Observer):
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

        _observer: QObserver
            The actuall Observer that finally should receive the
            notifications.
        _interests: Change
            The changes, the observer is interested in. This will be
            used as the `interests` argument when adding this
            :py:class:`Observer` to an :py:class:`Observable` using
            the :py:meth:`Observable.add_observer` method.  A value
            of `None` means that the :py:class:`Observer` is interested
            in all changes.
        _change: Change
            The unprocessed changes accumulated so far.
        _notify: Callable[[Observable, Change], None]
            The method to call upon notification. This will
            usually be the specific notification method of the
            observer.
        """

        def __init__(self, observer: Observable.Observer,
                     observable: Observable, interests=None,
                     notify: Callable = None, **kwargs) -> None:
            """Initialization of the QObserverHelper.
            """
            super().__init__(**kwargs)
            self._observer = observer
            self._observable = observable
            self._observing = False
            self._interests = interests
            self._change = None  # accumulated changes
            self._kwargs = {}  # additional arguments provided on notification
            self._notify = (observable.notify_method(observer)
                            if notify is None else notify)

        def observe(self, observable: Observable) -> None:
            """This method is implemented to comply with the Observer
            interface, but it is not intended to be used!
            If you want to observe an Observable in the Qt GUI, use
            the QObserver and call its :py:meth:`QObserver.observe` method.
            That will automatically create an :py:class:`QObserverHelper`
            """
            raise ValueError("This method is not intended to be called.")

        def unobserve(self, observable: Observable) -> None:
            """This method is implemented to comply with the Observer
            interface. It is equivalent to calling
            self.observer.unobserve(self.observable)
            """
            self._observer.unobserve(observable)

        def startObservation(self, interests=None,
                             notify: Callable = None) -> None:
            """Stat the observation.
            """
            if self._observing:
                return  # The observation was already started
            self._observing = True
            if interests is not None:
                self._interests = interests
            if notify is not None:
                self._notify = notify

            # Add this QObserverHelper as an observer to the observable
            self._observable.add_observer(self, notify=self._qNotify,
                                          interests=self._interests)
            # Send out a notifcation on all changes to update the observer
            self._qNotify(self._observable, self._interests)

        def stopObservation(self) -> None:
            if not self._observing:
                return  # No ongoing observation to stop.
            self._observing = False
            self._observable.remove_observer(self)

        def _qNotify(self, observable, change, **kwargs):
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
            # LOG.debug("%s._qNotify: change=%s [%s]",
            #           self._observer, change, self._change)
            if self._change is None:
                # Currently, there is no change event pending for this
                # object. So we will queue a new one and remember the
                # change:
                self._change = change
                self._kwargs = kwargs
                # This will use Qt.AutoConnection: the member is invoked
                # synchronously if obj lives in the same thread as the caller;
                # otherwise it will invoke the member asynchronously.
                try:
                    # LOG.debug("%s._qNotify: invokeMethod(_qAsyncNotify, %s)"
                    #           "in QThread %s",
                    #           self._observer, observable, self.thread())
                    argument = Q_ARG("PyQt_PyObject", observable)
                    QMetaObject.invokeMethod(self, '_qAsyncNotify', argument)
                    #     Q_ARG("PyQt_PyObject", change))
                    #   RuntimeError: QMetaObject.invokeMethod() call failed
                except Exception as ex:
                    handle_exception(ex)
            else:
                # There is already one change pending in the event loop.
                # We will just update the change, but not queue another
                # event.
                self._change |= change

        @pyqtSlot(object)
        def _qAsyncNotify(self, observable: Observable):
            """Process the actual notification.  This method is
            to be invoked in the Qt main event loop, where it can
            trigger changes to the graphical user interface.

            This method will simply call the :py:meth:`notify` method
            of the :py:class:`Observable` to notify the actual
            observer.

            """
            # print(f"%s._qAsyncNotify: notify=%s change=%s in thread %s",
            #       self._observer, self._notify, self._change, self.thread())
            change = self._change
            kwargs = self._kwargs
            if change is not None:
                self._change = None
                self._kwargs = {}
                try:
                    # make sure the wrapped C++ object has not
                    # been deleted in the meantime
                    self._observer.parent()
                except RuntimeError:
                    return
                try:
                    self._notify(observable, change, **kwargs)
                except Exception as ex:
                    handle_exception(ex)

        def __str__(self) -> str:
            return (f"QObserver[{self._observer}]: "
                    f"observable={self._observable}, "
                    f"interests={self._interests} [change={self._change}], "
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


class QBusyWidget(QLabel, QObserver, qobservables={
        BusyObservable: {'busy_changed'}}):
    """A widget indicating a that some component is busy.
    """
    _label: QLabel = None
    _movie: QMovie = None
    _busy: bool = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Animated gifs can be obtained from
        # http://www.ajaxload.info/#preview
        self._busy = None
        self._movie = QMovie(os.path.join('assets', 'busy.gif'))
        self.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

    def busy_changed(self, busyBody: BusyObservable,
                     change: BusyObservable.Change) -> None:
        """React to a change of business.
        """
        LOG.debug("QBusyWidget[%s].busy_changed(%s): %s",
                  busyBody, change, busyBody.busy)
        self.setBusy(busyBody.busy)

    def busy(self) -> bool:
        return self._busy

    def setBusy(self, busy: bool) -> None:
        if busy is not self._busy:
            self._busy = busy
            self.update()

    def update(self) -> None:
        self._movie.setPaused(not self._busy)
        if self._busy:
            # Setting the text clears any previous content (like
            # text, picture, etc.)
            self.setMovie(self._movie)
        else:
            # Setting the text clears any previous content (like
            # picture, movie, etc.)
            self.setText("Not busy")
        super().update()

    def __del__(self):
        """Free resources used by this :py:class:`QBusyWidget`.
        """
        # Make sure the movie stops playing.
        self._movie.stop()
        del self._movie


class QPrepareButton(QPushButton, QObserver, QDebug, qobservables={
        Preparable: {'busy_changed', 'state_changed'}}):
    """A Button to control a :py:class:`Preparable`, allowing to prepare
    and unprepare it.

    The :py:class:`QPrepareButton` can observe a
    :py:class:`Preparable` and adapt its appearance and function based
    on the state of the :py:class:`Preparable`.

    """
    # FIXME[todo]: This could be made a 'QPreaparableObserver'
    _unpreparable: bool = True

    def __init__(self, text: str = 'Prepare', **kwargs) -> None:
        """Initialize the :py:class:`QPrepareButton`.

        Arguments
        ---------
        text: str
            The button label.
        """
        super().__init__(text, **kwargs)
        self.setCheckable(True)
        self._text = text

        # We only want to react to button activation by user, not to
        # change of state via click() slot activation, or because
        # setChecked() is called. Hence we use the 'clicked' signal,
        # not 'toggled'!
        self.clicked.connect(self.onClicked)
        self.updateState()

    def preparable_changed(self, preparable: Preparable,
                           info: Preparable.Change) -> None:
        if info.state_changed:
            self.updateState()

    def setPreparable(self, preparable: Optional[Preparable]) -> None:
        # FIXME[hack/todo]: setting the preparable should actually
        # sent notification
        self.updateState()

    def setUnpreparable(self, unpreparable: bool = True) -> None:
        self._unpreparable = unpreparable
        self.updateState()
        
    @protect
    def onClicked(self, checked: bool) -> None:
        """React to a button activation by the user. This will
        adapt the preparation state of the :py:class:`Preparable`
        based on the state of the button.

        Arguments
        ---------
        checked: bool
            The new state of the button.
        """
        if isinstance(self._preparable, Preparable):
            if checked and not self._preparable.prepared:
                self._preparable.prepare()
            elif not checked and self._preparable.prepared:
                self._preparable.unprepare()

    def updateState(self) -> None:
        """Update this :py:class:`QPrepareButton` based on the state of the
        :py:class:`Preparable`.
        """
        enabled = (isinstance(self._preparable, Preparable) and
                   self._preparable.preparable and
                   not self._preparable.busy)
        checked = enabled and self._preparable.prepared
        if checked and not self._unpreparable:
            enabled = False
        if isinstance(self._preparable, Preparable) and self._preparable.busy:
            if self._preparable.prepared:
                self.setText("Unpreparing")
            else:
                self.setText("Preparing")
        else:
            self.setText(self._text)
        self.setEnabled(enabled)
        self.setChecked(checked)

    def debug(self) -> None:
        if isinstance(self, QDebug):
            super().debug()
        print(f"debug: QPrepareButton[{type(self).__name__}]: "
              f"enabled={self.isEnabled()}, checked={self.isChecked()}")
        print(f"debug:  - preparable: {type(self._preparable)}" +
              ("" if self._preparable is None else
               f"self._preparable.prepared"))
