"""Qt widgets to access a :py:class:`MetaRegister`.

"""

# standard imports
from typing import Iterator, Union, Callable
import logging

# Qt imports
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QBrush, QKeyEvent

from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel,
                             QListWidget, QListWidgetItem,
                             QHBoxLayout, QVBoxLayout)
from PyQt5.QtWidgets import QComboBox

# toolbox imports
from base import (Observable, MetaRegister, Registrable,
                  InstanceRegisterItem)
from toolbox import Toolbox

# GUI imports
from ..utils import QObserver, QDebug, protect

# logging
LOG = logging.getLogger(__name__)


class RegisterItemList(QObserver, QDebug,
                       qobservables={MetaRegister: {'key_changed'}}):
    """A list of register items.
    Register items are objects registered in a :py:class:`MetaRegister`.
    Those objects have :py:attr:`key` property providing
    a unique string allowing to refer to the object in the register.

    The :py:class:`RegisterItemList` can observe the
    :py:class:`MetaRegister` and react to changes like adding
    or removing keys, or initialization or deletion of objects.

    Subclasses of this class have to implement the methods
    :py:meth:`_keys`, :py:meth:`_addItem`, :py:meth:`_removeItem`,
    and :py:meth:`_updateItems`.

    FIXME[todo]: introduce a flag `initialized` allowing to switch
    between listing (False) all keys, (True) just initiaized items.
    Currently only initialized items can be listed.
    """

    def __init__(self, register: MetaRegister = None,
                 onlyInitialized: bool = True, **kwargs) -> None:
        """Initialization of the :py:class:`QNetworkSelector`.

        Parameters
        ----------
        """
        self._onlyInitialized = onlyInitialized
        super().__init__(**kwargs)
        self.setRegister(register)

    def register(self) -> MetaRegister:
        """The :py:class:`MetaRegister` used by this
        :py:class:`RegisterItemList`.
        """
        return self._register

    def setRegister(self, register: MetaRegister) -> None:
        """Set a new :py:class:`MetaRegister` for this
        :py:class:`RegisterItemList`. The entries of this list
        will be updated from the register.

        Arguments
        ---------
        register: MetaRegister
            The :py:class:`MetaRegister` from which the list
            will be updated. If `None` the list will be cleared.
        """
        self._updateFromRegister()

    def onlyInitialized(self) -> bool:
        """A flag indicating if only initialized items (`True`)
        or all item (`False`) are listed.
        """
        return self._onlyInitialized

    def setOnlyInitialized(self, onlyInitialized: bool = True) -> None:
        """Specify if only initialized items (`True`) or all item (`False`)
        shall be listed.
        """
        if onlyInitialized != self._onlyInitialized:
            self._onlyInitialized = onlyInitialized
            self._updateFromRegister()

    def register_changed(self, register: MetaRegister,
                         change: MetaRegister.Change, key: str = None) -> None:
        """Called upon a change in the :py:class:`MetaRegister`.

        Arguments
        ---------
        register: MetaRegister
            The :py:class:`MetaRegister` that was changed.
        change: MetaRegister.Change
        key: str
            The key that was changed.
        """
        LOG.info("%s.register_changed: %s [%r], key=%s",
                 type(self).__name__, register.__name__, change, key)
        # FIXME[concept]: key may be None, if the notification is
        # received uppon showing the widget after it was hidden.
        # This means that we can in fact not rely on key having
        # a meaningful value in the GUI - if we want to change this,
        # we would have to make the notification upon show more sophisticated!
        if change.key_changed:  # and key
            self._updateFromRegister()

    def _updateFromIterator(self, iterator: Iterator) -> None:
        """Update the items in this list from an iterator.
        Items from the iterator, that are not yet contained in the
        list are added, while elements from the list, that are not
        listed by the iterator, are removed from the list.
        """
        # 1. Create a set of keys of items already contained
        #    in this list (this is used for bookkeeping).
        keys = {key for key in self._keys()}

        # 2. Iterate over items from the iterator and add items
        #    missing in this list.
        for item in iterator:
            if item.key in keys:
                keys.remove(item.key)
            else:
                self._addItem(item)

        # 3. Remove items from this list that are no longer present
        for key in keys:
            index = self.findText(key)  # FIXME[hack]: combobox specific
            if index >= 0:
                item = self.itemData(index)
                self._removeItem(item)

        self._updateItems()

    def _updateFromRegister(self) -> None:
        """Update this :py:class:`RegisterItemList` to reflect the
        the current state of the register, taken the display flag
        `onlyInitialized` into account.
        """
        if self._register is None:
            self._updateFromIterator(iter(()))
        else:
            self._updateFromIterator(self._register.
                                     register_instances(self._onlyInitialized))

    def _update(self) -> None:
        self._updateFromRegister()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Process key events. The :py:class:`RegisterItemList` supports
        the following keys:

        R: update list from register
        U: update display of the items of this :py:class:`RegisterItemList`,
        I: toggle the `onlyInitialized` flag,
        """
        key = event.key()
        if key == Qt.Key_U:  # Update
            self._updateItems()
        elif key == Qt.Key_R:  # Update from register
            if self._register is not None:
                self._updateFromRegister()
        elif key == Qt.Key_I:  # Toggle onlyInitialized
            self.setOnlyInitialized(not self.onlyInitialized())
        else:
            super().keyPressEvent(event)

    def debug(self) -> None:
        super().debug()
        print(f"debug: RegisterItemList[{type(self).__name__}]:")
        print(f"debug:   * register: {self._register}")
        print(f"debug:   * onlyInitialized: {self.onlyInitialized()}")
        if self._register is not None:
            print(f"debug:   * register keys:")
            for key in self._register.register_keys():
                print(f"debug:     - {key} "
                      f"[{self._register.key_is_initialized(key)}]")
        print(f"debug:   * list keys:")
        for key in self._keys():
            LOG.debug(f"debug:  - key: {key}")

    def currentItem(self) -> Registrable:
        """Get the currently selected item.
        This may be `None` if no item is selected.
        """
        return self._currentItem()

    def setCurrentItem(self, item: Registrable) -> None:
        """Select the given item in this :py:class:`RegisterItemList`.

        Arguments
        ---------
        item: Registrable
            The item to become the currently selected item
            in this list. `None` will deselect the current element.
        
        Raises
        ------
        ValueError:
            The given item is not an element of this
            :py:class:`RegisterItemList`.
        """
        # FIXME[bug]: contains is not provided by this class!
        # if not self.contains(item):
        #    raise ValueError("Item is not contained in RegisterItemList")
        self._setCurrentItem(item)

    #
    # methods to be implemented by subclasses
    #

    def _keys(self) -> Iterator[str]:
        """An iterator for the keys of all items in this
        :py:class:`RegisterItemList`.
        """
        raise NotImplementedError("A 'RegisterItemList' has to implement "
                                  "the _keys() method")

    def _addItem(self, item: Registrable) -> None:
        """Add an item to this :py:class:`RegisterItemList`.
        It is assumed that the item is not yet contained in this
        :py:class:`RegisterItemList`.
        """
        raise NotImplementedError("A 'RegisterItemList' has to implement "
                                  "the _addItem() method")

    def _removeItem(self, item: Registrable) -> None:
        """Remove an item from this :py:class:`RegisterItemList`.  It is
        assumed that the item is contained in this
        :py:class:`RegisterItemList`, otherwise a
        :py:class:`ValueError` is raised.

        """
        raise NotImplementedError("A 'RegisterItemList' has to implement "
                                  "the _removeItem() method")

    def _updateItems(self) -> None:
        """Update the display of the list items. This may be implemented by
        subclasses that would like to adapt the style of display
        depending on the state of the item (e.g., initialized/prepared/busy).

        This method will be called when the list has been updated
        (e.g. by directly adding or removing elements, or by filling
        the list from some iterator), but subclasses may also call this
        method in response to `state_changed` and other notifications.
        """
        raise NotImplementedError("A 'RegisterItemList' has to implement "
                                  "the _updateItems() method")

    def _currentItem(self) -> Registrable:
        """Get the currently selected item.
        This may be `None` if no item is selected.
        """
        raise NotImplementedError("A 'RegisterItemList' has to implement "
                                  "the _currentItem() method")

    def _setCurrentItem(self, item: Registrable) -> None:
        """Select the given item in this :py:class:`RegisterItemList`.

        Arguments
        ---------
        item: Registrable
            The item to become the currently selected item
            in this list (the item is guaranteed to be an element
            of this list). `None` will deselect the current element.        
        """
        raise NotImplementedError("A 'RegisterItemList' has to implement "
                                  "the _setCurrentItem() method")


class QRegisterItemComboBox(RegisterItemList, QComboBox):
    """A :py:class:`QRegisterItemComboBox` is a :py:class:`QComboBox`
    allowing to select items from a :py:class:`MetaRegister`.

    This class extends the :py:class:`RegisterItemList` by providing
    methods for adding and removing items to/from a :py:class:`QComboBox`.

    The `QComboBox` will show the keys of the items and it will have
    the actual items stored as associated data (using Qt's `itemData`
    mechanism with the default role `Qt.UserRole`).
    """

    def __init__(self, **kwargs) -> None:
        """Initialization of the :py:class:`QNetworkSelector`.

        Parameters
        ----------
        """
        super().__init__(**kwargs)
        self.setCurrentIndex(-1)

    def _keys(self) -> Iterator[str]:
        """An iterator providing keys for all items in this
        :py:class:`QRegisterItemComboBox`.
        """
        for index in range(self.count()):
            item = self.itemData(index)
            yield item.key

    def _addItem(self, item: Registrable) -> None:
        """Add an item to this :py:class:`QRegisterItemComboBox`.
        It is assumed that the item is not yet contained in this
        :py:class:`QRegisterItemComboBox`.
        """
        self.addItem(item.key, item)

    def _removeItem(self, item: Registrable) -> None:
        """Remove an item from this :py:class:`QRegisterItemComboBox`.  It is
        assumed that the item is contained in this
        :py:class:`QRegisterItemComboBox`, otherwise a
        :py:class:`ValueError` is raised.

        """
        index = self.findText(item.key)
        if index < 0:
            raise ValueError(f"Item '{item.key}' is not contained in this "
                             f"QRegisterItemComboBox")
        self.removeItem(index)

    def _updateItems(self) -> None:
        """Update all items of this :py:class:`QRegisterItemComboBox`
        to reflect their current state.
        Currently only preparation of an item is indicated
        (red = unprepared, green = prepared).
        """
        for index in range(self.count()):
            item = self.itemData(index)
            initialized = not isinstance(item, InstanceRegisterItem)
            self.setItemData(index,
                             QBrush(Qt.gray if not initialized else
                                    Qt.green if item.prepared else Qt.red),
                             Qt.ForegroundRole)  # Qt.BackgroundRole
        self.update()

    def _currentItem(self) -> Registrable:
        """Get the currently selected item.
        This may be `None` if no item is selected.
        """
        return self.currentData()

    def _setCurrentItem(self, item: Registrable) -> None:
        """Select the given item in this :py:class:`RegisterItemList`.

        Arguments
        ---------
        item: Registrable
            The item to become the currently selected item
            in this list (the item is guaranteed to be an element
            of this list). `None` will deselect the current element.        
        """
        # For an empty combo box or a combo box in which no current item
        # is set, the index is -1 (which is also returned by findText if
        # the item is not found).
        self.setCurrentIndex(self.findText(item.key))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Process key events. The :py:class:`RegisterItemList` supports
        the following keys:

        c: clear the currently selected entry

        Note the this event is only received if the QComboBox is closed
        (not while currently selecting an item).
        """
        key = event.key()
        LOG.debug("debug: QRegisterItemComboBox.keyPressEvent: key=%d", key)
        if key == Qt.Key_C:  # Clear
            self.setCurrentIndex(-1)
        else:
            super().keyPressEvent(event)


class QRegisterListItem(QListWidgetItem, QObserver, qobservables={
        Observable: {'state_changed'}}):
    """
    **Observer:**
    If representing an observable object (that is an initialized key),
    the `QRegisterListItem` will observe it for 'state_changed' events,
    and it will reflect such changes automatically.
    However, if the item represents something else (an unititialized
    key or a class), the item has to be explicitly asked to
    reflect this change by invoking its :py:meth:`update` method.
    """
    
    KeyType = QListWidgetItem.UserType + 1
    ClassType = QListWidgetItem.UserType + 2

    UserKeyRole = Qt.UserRole + 1
    UserDataRole = Qt.UserRole + 2
 
    def __init__(self, key: str=None, data=None, **kwargs) -> None:
        """
        Arguments
        ---------
        data:
            The data behind this :py:class:`QRegisterListItem`.
            This can either be (a) an Observable, (b) a class,
            (c) a fully qualified class name, or (d) None, meaning
            the text is a key for initializing a registered item.
        """
        super().__init__(**kwargs)
        self._initializableHack = False  
        self.setData(self.UserKeyRole, key)
        self.updateData(data)

        if self.isKey():
            self.setText(self.key())
        elif self.isClass():
            self.setText(f"[{self.className(full=False)}]")

    def __del__(self):
        # Note: when calling __del__(), the wrapped C/C++ object of
        # may already have been deleted. Hence no access to the
        # internal state via self.text() or self.data() is possible.
        # Note 2: there is no parent destructor super().__del__()
        LOG.info("QRegisterListItem.__del__")
        self.updateData(None)  # remove observer if registered

    def isKey(self):
        """Check if this :py:class:`QRegisterListItem` is the key of
        an (initialized or uninitialized) instance.
        """
        return self.type() == self.KeyType

    def isClass(self):
        """Check if this :py:class:`QRegisterListItem` is the name of
        an (initialized or not yet initialized) class.
        """
        return self.type() == self.ClassType

    def key(self) -> str:
        """The key, that is a string uniquely identifying the (initialized or
        uninitialized) object or class represented by this
        :py:class:`QRegisterListItem`. In case of an object, this is
        the key by which the object is registered at the register meta
        class. In case of a class, it is the fully qualified class name.
        """
        return self.data(self.UserKeyRole)

    def className(self, full: bool=True) -> str:
        if self.type() != self.ClassType:
            raise TypeError("Item of type {self.type()} has no class name.")
        name = self.data(self.UserKeyRole)
        return name if full else name.split('.')[-1]

    def isInitialized(self) -> bool:
        return self.data(self.UserDataRole) is not None

    # FIXME[hack]: it would be better if this could be determined automatically ...
    def setInitializableHack(self, initializable: bool) -> None:
        self._initializableHack = initializable
        self.updateFormat()

    def isInitializable(self) -> bool:
        return self._initializableHack

    def updateData(self, data) -> None:
        oldData = self.data(self.UserDataRole)
        if oldData != data:
            if isinstance(oldData, Observable):
                self.unobserve(oldData)
            self.setData(self.UserDataRole, data)
            if isinstance(data, Observable):
                # we are interested in the following information:
                #  - data.prepared: 'state_changed'
                self.observe(data, notify=self.observableChanged)
        self.updateFormat()

    def updateFormat(self):
        """Update the format of this :py:class:`QRegisterListItem` based
        on the state of the object represented by this item.
        This method is intended to be invoked by methods that invoke
        """
        if self.isKey():
            if self.isInitialized():
                data = self.data(self.UserDataRole)
                if not data.prepared:
                    self.setForeground(Qt.blue)
                elif data.busy:
                    self.setForeground(Qt.red)
                else:
                    self.setForeground(Qt.green)
            else:
                self.setForeground(Qt.black)
        elif self.isClass():
            self.setForeground(Qt.black)
        LOG.debug("key='%s': initialized=%r, initializable=%r",
                  self.key(), self.isInitialized(), self.isInitializable())
        self.setBackground(Qt.white if self.isInitialized() else
                           Qt.lightGray if self.isInitializable() else
                           QColor(Qt.red).lighter())

    def observableChanged(self, observable, change) -> None:
        """React to state_changed notifications of the instantiated
        observables represented in this :py:class:`QRegisterListItem`.
        """
        # FIXME[todo]: currently we also recieve change notification
        # even if the list item or even the complete list is not visible.
        # This could be improved to only receive notifications if necessary.
        LOG.debug("QRegisterListItem.observableChanged(%r, %r)",
                  observable, change)
        self.updateFormat()


class QRegisterList(QListWidget, QObserver, qobservables={
        MetaRegister: MetaRegister.Change.all(),
        Toolbox: set()}):
    """A list displaying the observables listed by a
    :py:class:`Toolbox` or a register, or classes listed in a register.


    Attributes
    ----------

    **Signals:**

    A :py:class:`QRegisterList` provides different signals that
    correspond to the different types of entries that can be selected:
    * keySelected(str): The value is the selected key.
    * classSelected(str): The value is the fully qualified class name.
    """

    keySelected = pyqtSignal(str)
    classSelected = pyqtSignal(str)

    def __init__(self, toolbox: Toolbox=None, interests: Toolbox.Change=None,
                 register: MetaRegister=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.itemClicked.connect(self.onItemClicked)
        self._toolboxInterests = interests
        self._showKeys = True
        self._showClasses = False
        self._showUninitialized = True  # show uninitialized items
        self._showRegister = True       # show items from the register
        self.setToolbox(toolbox)
        self.setRegister(register)

    def showKeys(self) -> bool:
        """Check if key items should be shown.
        """
        return self._showKeys

    def setShowKeys(self, flag: bool) -> None:
        """Set if key items should be shown.
        """
        if self._showKeys != flag:
            self._showKeys = flag
            self.update()

    def showClasses(self) -> bool:
        """Check if class items are shown.
        """
        return self._showClasses

    def setShowClasses(self, flag: bool) -> None:
        """Set if class items should be shown.
        """
        if self._showClasses != flag:
            self._showClasses = flag
            self.update()

    def showUninitialized(self) -> bool:
        """Check if uninitialized items are shown.
        """
        return self._showUninitialized

    def setShowUninitialized(self, flag: bool) -> None:
        """Set if uninitialized items should be shown.
        """
        if self._showUninitialized != flag:
            self._showUninitialized = flag
            self.update()

    def showRegister(self) -> bool:
        """Check if items from the register are shown.
        """
        return self._showRegister

    def setShowRegister(self, flag: bool) -> None:
        """Set if register items should be shown.
        """
        if self._showRegister != flag:
            self._showRegister = flag
            self.update()

    def itemForKey(self, key: str) -> QRegisterListItem:
        for i in range(self.count()):
            item = self.item(i)
            if item.isKey() and item.key() == key:
                return item
        return None

    def itemForClass(self, name: str) -> QRegisterListItem:
        for i in range(self.count()):
            item = self.item(i)
            if item.isClass() and item.key() == name:
                return item
        return None

    def toolbox_changed(self, toolbox: Toolbox,
                        change: Toolbox.Change) -> None:
        # FIXME[bug]: I receive uninteresting changes here (e.g.'input_changed')
        # although I only want to see 'datasources_changed', e.g. when
        # selecting items in QRegisterList
        LOG.debug("QRegisterList.toolbox_changed(%r, %r, %r)",
                  self, toolbox, change)
        if self._toolboxInterests is None or (self._toolboxInterests & change):
            self._updateList(toolbox=toolbox)

    def setRegister(self, register) -> None:
        self._updateList(register=register)

    def register_changed(self, register, change, key: str = None) -> None:
        LOG.info("%s.register_changed: %s [%r], key=%s",
                 type(self).__name__, register.__name__, change, key)
        if change.key_changed and key and key in register.register_keys():
            item = self.itemForKey(key)
            if register.key_is_initialized(key):
                item.updateData(register[key])
            else:
                item.updateFormat()
        elif change.class_changed and key and key in register.classes():
            item = self.itemForClass(key)
            if register.class_is_initialized(key):
                item.updateData(register.register_get_class[key])
            else:
                item.updateFormat()
        else:
            self._updateList(register=register)

    def _updateKeys(self, itemType: int, iterator: Iterator,
                    remove: bool = False) -> None:
        """Update list items of a given type from an iterator.

        Arguments
        ---------
        itemType: int
            The type of the items to be updated.
        iterator: Iterator[Tuple[str,object]]
        """
        # 1. Create list of key items already contained
        #    in this QRegisterList. This is used for bookkeeping.
        items = {item.key(): item for item in
                 [self.item(i) for i in range(self.count())]
                 if item.type() == itemType}

        # 2. Iterate over the keys in the register and
        #    add items for those that are missing
        for key, data in iterator:
            if key in items:
                items[key].updateData(data)
                del items[key]
            else:
                item = QRegisterListItem(key, data=data, type=itemType)
                if itemType == item.KeyType and data is None:
                    initializable = self._register.key_is_initializable(key)
                    item.setInitializableHack(initializable)
                self.addItem(item)

        # 3. Remove items from this list that are no longer present
        # FIXME[concept]: what do we really want here: if initialized
        # from a register, here may be more items than in the toolbox.
        # Do we want to remove those, or just hide ...
        if remove:
            for text, item in items.items():
                # FIXME[todo]: check if to use removeItem() or
                # takeItem() should be used -> also check if the item
                # is deleted (stops observing)...
                #self.removeItem(item)
                self.takeItem(self.row(item))
                del item

    def _updateList(self, toolbox: Toolbox = None, register=None) -> None:
        """Update the item list, from a toolbox, a register, or both.
        """
        if toolbox is not None:
            self._updateKeys(QRegisterListItem.KeyType,
                             self._toolboxIterator())

        if register is not None:
            iterator = ((key, register[key]
                         if register.key_is_initialized(key) else None)
                        for key in register.register_keys())
            self._updateKeys(QRegisterListItem.KeyType, iterator, remove=True)
            iterator = ((name, register.register_get_class(name)
                         if register.class_is_initialized(name) else None)
                        for name in register.classes(abstract=False))
            self._updateKeys(QRegisterListItem.ClassType, iterator)

    def update(self):
        """Update this :py:class:`QRegisterList` to only show the
        items indicated by the show* flags.
        """

        if not self._showRegister:
            toolbox_keys = set((key for key, _ in self._toolboxIterator()))

        for i in range(self.count()):
            item = self.item(i)
            hidden = ((not self._showUninitialized and
                       not item.isInitialized()) or
                      (not self._showKeys and item.isKey()) or
                      (not self._showClasses and item.isClass()) or
                      (not self._showRegister and
                       (not item.isKey() or item.key() not in toolbox_keys)))
            item.setHidden(hidden)
        super().update()

    def _toolboxIterator(self) -> Iterator:
        """An iterator of pairs (key, observable) of the currently selected
        Toolbox. Will be an empty iterator if no Toolbox is set.
        """
        return iter(())  # to be implemented by subclasses

    @protect
    def onItemClicked(self, item: QListWidgetItem):
        """Respond to a click on an item in this :py:class:`QRegisterList`.
        Depending on the type of the item, this will emit a
        `keySelected` or a `classSelected` signal.

        Arguments
        ---------
        item: QListWidgetItem
            The list item that triggered the event. Should be a
            :py:class:`QRegisterListItem`, either of type
            `QRegisterListItem.KeyType` or of type
            `QRegisterListItem.ClassType`.

        Raises
        ------
        TypeError:
            If the type of item is neither `QRegisterListItem.KeyType`
            nor `QRegisterListItem.ClassType`.
        """
        itemType = item.type()
        if itemType == QRegisterListItem.KeyType:
            self.keySelected.emit(item.key())
        elif itemType == QRegisterListItem.ClassType:
            self.classSelected.emit(item.className())
        else:
            raise TypeError("Invalid type associed with clicked item "
                            f"'{item.text()}': {itemType}")

    @protect
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Process key events. The :py:class:`QImageView` supports
        the following keys:

        r: toggle the keepAspectRatio flag
        """
        key = event.key()
        LOG.debug("debug: QRegisterList.keyPressEvent: key=%d", key)

        if key == Qt.Key_I:  # Initialized
            self.setShowUninitialized(not self.showUninitialized())
        elif key == Qt.Key_K:  # Keys
            self.setShowKeys(not self.showKeys())
        elif key == Qt.Key_C:  # Classes
            self.setShowClasses(not self.showClasses())
        elif key == Qt.Key_R:  # Register
            self.setShowRegister(not self.showRegister())
        elif key == Qt.Key_U:  # Update
            self._updateList(toolbox=self._toolbox, register=self._register)
        elif key == Qt.Key_D:  # Debug
            self.debug()
        elif key == Qt.Key_T:  # Debug Toolbox
            Toolbox.debug_register()
        else:
            super().keyPressEvent(event)

    #
    # FIXME[debug]
    #

    def removeItemWidget(self, item) -> None:
        # FIXME[debug]
        print(f"QListWidget.removeItemWidget({item})")
        super().removeItemWidget(item)

    def takeItem(self, row: int) -> QListWidgetItem:
        # FIXME[debug]
        print(f"QListWidget.takeItem(row)")
        return super().takeItem(row)

    def debug(self) -> None:
        super().debug()
        print(f"debug: QRegisterList[{type(self).__name__}]:"
              f"debug:  * register: {getattr(self,'_register',None)}"
              f"debug:  * toolbox: {getattr(self,'_toolbox',None)}")


class QRegisterController(QWidget, QObserver, qobservables={
        MetaRegister: {'key_changed', 'class_changed'}}):
    """The :py:class:`QRegisterController` can control general
    a register. This includes:
    * initialization of register keys
    * prepare/unprepare of initialized register items

    Attributes
    ----------

    **Signals:**

    A :py:class:`QRegisterController` emits different signals
    corresponding to the different actions that can be initiated:
    * initializeKeyClicked(str): The value is the key.
    * initializeClassClicked(str): The value is the fully qualified class name.

    """
    # FIXME[todo]: make this three different controls:
    # 1. a key initialization Widget
    # 2. a class initialization widget
    # 3. a prepartion panel

    initializeKeyClicked = pyqtSignal(str)
    initializeClassClicked = pyqtSignal(str)
    keyInitialized = pyqtSignal(str)

    def __init__(self, register, **kwargs) -> None:
        """Initialization of the :py:class:`QRegisterController`.
        """
        super().__init__(**kwargs)
        self._register = register
        self._key = None  # the register key
        self._object = None  # the actual object (if initialized)
        self._name = None  # the class name
        self._initUI()
        self._layoutUI()

    def _initUI(self) -> None:
        """Initialize the user interface for this
        :py:class:`QRegisterController`.
        """
        self._button = QPushButton()
        self._button.clicked.connect(self._onButtonClicked)
        self._button.setEnabled(False)
        self._keyLabel = QLabel()
        self._stateLabel = QLabel()

        self._errorLabel = QLabel()
        self._errorLabel.setWordWrap(True)
        # FIXME[todo]: find the best way (portable, theme aware, ...)
        # to set QLabel style and apply this globally (to all QLabels
        # in the user interface)
        #self._errorLabel.setStyleSheet("QLabel { color : red; }")
        palette = self._errorLabel.palette()
        palette.setColor(self._errorLabel.foregroundRole(), Qt.red)
        self._errorLabel.setPalette(palette)

        self._descriptionLabel = QLabel()
        self._descriptionLabel.setWordWrap(True)

    def _layoutUI(self) -> None:
        """Layout the user interface for this
        :py:class:`QRegisterController`.
        """
        layout = QVBoxLayout()

        rows = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(QLabel('Key: '))
        row.addWidget(self._keyLabel)
        row.addStretch()
        rows.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel('State: '))
        row.addWidget(self._stateLabel)
        row.addStretch()
        rows.addLayout(row)

        row = QHBoxLayout()
        row.addLayout(rows)
        row.addStretch()
        row.addWidget(self._button)

        layout.addLayout(row)
        layout.addWidget(self._errorLabel)
        layout.addStretch()
        layout.addWidget(self._descriptionLabel)
        self.setLayout(layout)

    def setObject(self, obj: Union[object, str]) -> None:
        """Set the object to be controlled by this
        :py:class:`QRegisterController`.

        Arguments
        ---------
        obj: Union[object, str]
            A :py:class:`object` or the key of a
            :py:class:`object` to be controlled by this
            :py:class:`QRegisterController`. If a key is given
            the actual :py:class:`object` will be obtained from the
            register class.  If not initialized
            yet, the :py:class:`QRegisterController` will present
            an option to initialize the :py:class:`object`.
        """

        if obj is None:
            key = None
        elif isinstance(obj, str):
            # An object key is provided
            key = obj
            obj = (self._register[key] if self._register.key_is_initialized(key)
                   else self._register)
        elif isinstance(obj, self._register):
            # An object is provided
            key = obj.key
        else:
            raise TypeError("Argument object has invalid type: "
                            f"{type(obj)}")

        if key == self._key:
            if obj is self._object:
                return  # nothing changed, hence ther is nothing to do ...
            print(f"\n*** New Object was initialized [{key}]: {obj}\n")

        self.setClass(None)
        
        if self._object is not None:
            if self._object is self._register:
                self.unobserve(self._object)
            else:
                self._object.remove_observer(self)

        self._key, self._object = key, obj

        self._keyLabel.setText(key or "None")
        if not key:
            self._descriptionLabel.setText("") 
            self._stateLabel.setText("")
        elif obj is self._register:
            if self._register.key_is_initializable(key):
                self._descriptionLabel.setText("Uninitialized object")
            else:
                self._descriptionLabel.setText("Uninitializable object")
            self.observe(obj)
        else:
            insterests = obj.Change('busy_changed', 'state_changed')
            obj.add_observer(self, insterests, self.observableChanged)
            description = getattr(obj, 'description', "No description")
            self._descriptionLabel.setText(description)

        self.update()

    def setClass(self, cls: Union[type, str]) -> None:
        """

        Arguments
        ---------
        cls: Union[type, str]
            Either a subclass of the register class or the
            (fully qualified) name of such a class.
        """
        if cls is None or isinstance(cls, str):
            # A class name is provided
            name = cls
        elif issubclass(cls, self._register):
            name = cls.__module__ + '.' + cls.__name__
        else:
            raise TypeError("Argument class has invalid type: "
                            f"{type(cls)}")

        if name == self._name:
            return  # nothing to do ...

        self.setObject(None)
        self._name = name
        
        self._keyLabel.setText(f"[{name}]" if name else "None")
        if not name:
            self._description = None
            self._stateLabel.setText("")
        elif not self._register.class_is_initialized(name):
            self._descriptionLabel.setText("Uninitialized class")
        else:
            self._descriptionLabel.setText("Initialized class")

        self.update()

    def update(self) -> None:
        """Update the display to reflect the current state of the object or
        class.

        """
        failure_description = ''
        import threading
        if self._key is not None:
            enableButton = bool(self._key)
            if self._register.key_is_initialized(self._key):
                obj = self._register[self._key]
                if obj.failed:
                    self._button.setText('Clean')
                    self._button.setCheckable(False)
                else:
                    self._button.setText('Prepare')
                    self._button.setCheckable(True)
                    self._button.setChecked(obj.prepared)
                if obj.busy:
                    enableButton = False
                    stateText = f"busy ({obj.busy_message})"
                elif obj.prepared:
                    stateText = "prepared"
                else:
                    stateText = "unprepared"
                failure_description = obj.failure_description
            else:
                self._button.setText('Initialize')
                self._button.setCheckable(False)
                if not self._register.key_is_initializable(self._key):
                    enableButton = False
                if self._register.busy:
                    enableButton = False
                    stateText = f"busy ({self._register.busy_message})"
                else:
                    stateText = f"uninitialized"
            self._stateLabel.setText(stateText)
            self._button.setEnabled(enableButton)
        elif self._name is not None:
            initialized = self._register.class_is_initialized(self._name)
            self._button.setText('Import')
            self._button.setCheckable(False)
            self._button.setEnabled(not initialized)
        else:
            self._button.setEnabled(False)
        self._errorLabel.setText(failure_description)
        super().update()

    def observableChanged(self, obj: object, self2,
                          info: Observable.Change) -> None:
        #Unhandled exception (TypeError): datasourceChanged() takes 3 positional arguments but 4 were given
        #  File "/home/ulf/projects/github/DeepLearningToolbox/util/error.py", line 93, in closure
        #    function(self, *args, **kwargs)
        #  File "/home/ulf/projects/github/DeepLearningToolbox/qtgui/widgets/datasource.py", line 986, in _onButtonClicked
        #    datasource.unprepare()
        #  File "/home/ulf/projects/github/DeepLearningToolbox/datasource/datasource.py", line 120, in unprepare
        #    self.change('state_changed', 'data_changed')
        #  File "/home/ulf/projects/github/DeepLearningToolbox/base/observer.py", line 323, in change
        #    self.notify_observers(self.Change(*args, **kwargs))
        #  File "/home/ulf/projects/github/DeepLearningToolbox/base/observer.py", line 377, in notify_observers
        self.update()

    def observable_changed(self, sender, change) -> None:
        self.setObject(self._key)
        self.update()

    def register_changed(self, register, change, key: str=None) -> None:
        print(f"QRegisterController.register_changed({self.__class__}, {register}, {change}, '{key}')")
        #self.update()
        if change.key_changed and key == self._key:
            if register.key_is_initialized(key):
                print(f"QRegisterController.register_changed: "
                      f"emit key_initialized({key})")
                self.keyInitialized.emit(key)

    @protect
    def _onButtonClicked(self, checked: bool) -> None:
        LOG.info("%s.buttonClicked(checked=%r): text=%s, key=%s",
                 type(self).__name__, checked, self._button.text(), self._key)
        if self._key is None:
            return  # nothing to do (should not happen) ...

        if self._register.key_is_initialized(self._key):
            obj = self._register[self._key]
            LOG.info("%s.buttonClicked(checked=%r): "
                     "failed=%r, busy=%r, prepared=%r",
                     type(self).__name__, checked,
                     obj.failed, obj.busy,obj.prepared)
            if obj.failed:
                obj.clean_failure()
            elif checked:
                obj.prepare()
            else:
                obj.unprepare()
        else:
            # initialize the object for key
            self._register.register_initialize_key(self._key)

    @protect
    def onKeySelected(self, key: str) -> None:
        """
        """
        #if isinstance(self._toolbox, Toolbox):
        #    self._toolbox.add_datasource(key)
        self.setObject(key)

    @protect
    def onClassSelected(self, name: str) -> None:
        """
        """
        #if isinstance(self._toolbox, Toolbox):
        #    self._toolbox.add_datasource(key)
        self.setClass(name)

from base import Preparable

class QPrepareButton(QPushButton, QObserver, qobservables={
        Preparable: {'busy_changed', 'state_changed'}}):
    """A Button to control a :py:class:`Preparable`, allowing to prepare
    and unprepare it.

    The :py:class:`QPrepareButton` can observe a
    :py:class:`Preparable` and adapt its appearance and function based
    on the state of the datasource.

    """
    # FIXME[todo]: This could be made a 'QPreaparableObserver'

    def __init__(self, text: str='Prepare', **kwargs) -> None:
        """Initialize the :py:class:`QPrepareButton`.

        Arguments
        ---------
        text: str
            The button label.
        
        """
        super().__init__(text, **kwargs)
        self.setCheckable(True)

        # We only want to react to button activation by user, not to
        # change of state via click() slot activation, or because
        # setChecked() is called. Hence we use the 'clicked' signal,
        # not 'toggled'!
        self.clicked.connect(self.onClicked)
        self.update()

    def preparable_changed(self, preparable: Preparable,
                           info: Preparable.Change) -> None:
        if info.state_changed:
            self.update()

    def setPreparable(self, preparable: Preparable) -> None:
        # FIXME[hack/todo]: setting the preparable should actually
        # sent notification
        self.update()

    @protect
    def onClicked(self, checked: bool) -> None:
        """React to a button activation by the user. This will
        adapt the preparation state of the :py:class:`Datasource`
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

    def update(self) -> None:
        """Update this :py:class:`QPrepareButton` based on the state of the
        :py:class:`Datasource`.
        """
        enabled = (isinstance(self._preparable, Preparable) and
                   not self._preparable.busy)
        self.setEnabled(enabled)
        self.setChecked(enabled and self._preparable.prepared)
