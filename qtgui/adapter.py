"""This module provides different adapter classes that allow
for a smoother combination of Qt and the Deep Learning ToolBox.

"""

# standard imports
from typing import Iterator, Iterable, Any, Callable
import logging

# Qt imports
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QComboBox, QListWidget, QListWidgetItem

# GUI imports
from .utils import qtName, protect, QDebug

# logging
LOG = logging.getLogger(__name__)


class ItemAdapter(QDebug):
    """This class provides functionality that can be used by QWidgets that
    allow to choose from lists of items, like `QComboBox` and
    `QListWidget`. It acts as a translator mapping between the data
    structures used in the Deep Learning ToolBox and the Qt widgets.

    The QWidgets allow to store items and associated data in different
    ways:

    * The `QListWidget` uses `QListWidgetItem`s to represent the list items.
      Such an item is not a QWidget, but holds some information specifying
      display properties (like foreground and background color or icons),
      the text value of the item and it allows to store additional
      associated user date by introducing specific roles.

    * The `QComboBox` does not use an explict class to represent
      list items, but it also allows to set display properties and
      to store associated information for each item using roles.

    Both Widgets have the following comonalities:
    * New items can be registered with
      `QComboBox.addItem(text, [icon], [userData])` and
      `QListWidget.addItem(label=text)`

    * Items can be accessed by index:
      `QComboBox.itemText(index)` and `QListWidget.item(row).text()`

    * Items can be accessed by text:
      `QComboBox.findText(text)` gives a single index while
      `QList.findItems(text)` returns a list of item objects.

    * Items can be removed:
      `QComboBox.removeItem(index)` and
      `QListWidget.takeItem(QListWidget.item(index))

    * There may be a current item (selected item). The numerical index
      can be obtained by
      `QComboBox.currentIndex()` and `QListWidget.currentRow()`

    * The text of the current item can be obtained by
      `QComboBox.currentText()` and `QListWidget.currentItem().text()`

    * data associated with the current item can be obtained by
      `QComboBox.currentData(role)` and `QListWidget.currentItem().data(role)`
    """

    _itemToText: Callable[[Any], str] = str

    def __init_subclass__(cls, itemType: type = None,
                          itemToText: Callable[[Any], str] = None,
                          **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if itemType is not None:
            setattr(cls, qtName(itemType.__name__), cls._currentItem)
            setattr(cls, qtName('set_' + itemType.__name__), cls._currentItem)
        if itemToText is not None:
            cls._itemToText = staticmethod(itemToText)
        print(f"DEBUG1[{cls.__name__}]: itemToText:",
              itemToText, cls._itemToText)

    def __init__(self, itemToText: Callable[[Any], str] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.setItemToText(itemToText)

    #
    # methods to be implemented by subclasses
    #

    def _items(self) -> Iterator[Any]:
        """An iterator for the items in this
        :py:class:`ItemAdapter`.
        """
        raise NotImplementedError("A 'ItemAdapter' has to implement "
                                  "the _items() method")

    def _addItem(self, item: Any) -> None:
        """Add an item to this :py:class:`ItemAdapter`.
        It is assumed that the item is not yet contained in this
        :py:class:`ItemAdapter`.
        """
        raise NotImplementedError("A 'ItemAdapter' has to implement "
                                  "the _addItem() method")

    def _removeItem(self, item: Any) -> None:
        """Remove an item from this :py:class:`ItemAdapter`.  It is
        assumed that the item is contained in this
        :py:class:`ItemAdapter`, otherwise a
        :py:class:`ValueError` is raised.
        """
        raise NotImplementedError("A 'ItemAdapter' has to implement "
                                  "the _removeElement() method")

    def _currentItem(self) -> Any:
        """Get the currently selected item.
        This may be `None` if no itm is selected.
        """
        raise NotImplementedError("A 'ItemAdapter' has to implement "
                                  "the _currentItem() method")

    def _setCurrentItem(self, item: Any) -> None:
        """Select the given entry in this :py:class:`ItemAdapter`.

        Arguments
        ---------
        item: Any
            The item to become the current item. If the item is not
            contained in this :py:class:`ItemAdapter` (e.g. if
            `item` is `None`), the current will be set to `None`.
        """
        raise NotImplementedError("A 'ItemAdapter' has to implement "
                                  "the _setCurrentItem() method")

    #
    # Implemented methods
    #

    def _countItems(self) -> int:
        """Get the number of items in this :py:class:`ItemAdapter`.
        """
        return sum(1 for _ in self._items())

    def _textForItem(self, item: Any) -> str:
        """Get the text to be display from a given item.
        """
        return self._itemToText(item)

    def _formatItem(self, item: Any) -> None:
        """May be implemented by a subclass to format an item.

        This method is only called if the item is currently displayed
        by this :py:class:`ItemAdapter` (has been added and was not removed),
        but it may be called several times for the same item (to
        trigger an update of this item).

        The base implementation does nothing, but derived classes may
        overwrite this method to allow for fancy formating.
        """

    def _getItemAt(self, index: int) -> Any:
        """

        Raises
        ------
        IndexError:
            The index provided is invalid.
        """
        try:
            return next((x for i, x in enumerate(self._items()) if i == index))
        except StopIteration:
            raise IndexError(f"Index {i} beyond end of items.")

    def _getTextAt(self, index: int) -> str:
        """
        Raises
        ------
        IndexError:
            The index provided is invalid.
        """
        return self._textForItem(self._getItemAt(index))

    def _indexOfItem(self, item: Any) -> int:
        """

        Raises
        ------
        LookupError:
            The given item is not found in this :py:class:`ItemAdapter`.
        """
        try:
            return next(i for i, x in enumerate(self._items()) if x == item)
        except StopIteration:
            raise LookupError(f"Item {item} not found.")

    def _indexOfText(self, text: str) -> int:
        """
        Raises
        ------
        LookupError:
            The given text is not found in this :py:class:`ItemAdapter`.
        """
        try:
            return next(i for i, t in enumerate(self._texts()) if t == text)
        except StopIteration:
            raise LookupError(f"Item with text '{text}' not found")

    def _findItem(self, text: str) -> Any:
        """
        Raises
        ------
        LookupError:
            The given text is not found in this :py:class:`ItemAdapter`.
        """
        try:
            return next(item for item in self._items()
                        if self._textForItem(item) == text)
        except StopIteration:
            raise LookupError(f"Item with text '{text}' not found.")

    def _setCurrentText(self, text: str) -> None:
        """
        """
        self._setCurrentItem(self._findItem(text))

    def _texts(self) -> Iterator[str]:
        """An iterator for the texts presented by this
        :py:class:`ItemAdapter`.
        """
        for item in self._items():
            yield self._textForItem(item)

    def _removeText(self, text: str) -> None:
        """Remove the item with the given text. This may be
        overwritten by subclasses when a more efficient implementation
        is possible.
        """
        self._removeItem(self._findItem(text))

    def _removeItemAt(self, index: int) -> None:
        """Remove the item at the given index.

        Raises
        ------
        IndexError:
            The index provided is invalid.
        """
        self._removeItem(self._getItemAt(index))

    def _removeAllItems(self) -> None:
        """Remove all items in this :py:class:`ItemAdapter`.
        """
        try:
            self._removeItemAt(0)
        except IndexError:
            pass  # no item left to remove

    def _formatAllItems(self) -> None:
        """
        """
        for item in self._items():
            self._formatItem(item)

    def _updateAllItems(self) -> None:
        """Update the display of the list elements. This may be implemented by
        subclasses that would like to adapt the style of display
        depending on the state of the element.

        This method will be called when the list has been updated
        (e.g. by directly adding or removing elements, or by filling
        the list from some iterable), but subclasses may also call this
        method proactively in repsonse to notifications.
        """

    #
    # public interface
    #

    def setFromIterable(self, iterable: Iterable) -> None:
        """Set the items in this :py:class:`ItemAdapter` from an
        iterable. This will first remove the old items and then
        add the new items.
        """
        self._removeAllItems()
        for item in iterable:
            self._addItem(item)

    def updateFromIterable(self, iterable: Iterable) -> None:
        """Update the items in this :py:class:`ItemAdapter` from an iterable.
        Items from the iterable, that are not yet contained in the
        list are added, while items originally contained in this
        :py:class:`ItemAdapter`, that are not iterated by the
        iterable, are removed.
        """
        # 1. Create a set containing the texts for items already contained
        #    in this list (this is used for bookkeeping).
        bookkeeping = set(self._texts())

        # 2. Iterate over entries from the iterable and add entries
        #    missing in this list.
        for item in iterable:
            text = self._textForItem(item)
            if text in bookkeeping:
                bookkeeping.remove(text)
            else:
                self._addItem(item)

        # 3. Remove items from this list that are no longer present
        for text in bookkeeping:
            self._removeText(text)

    def setItemToText(self, itemToText: Callable[[Any], str]) -> None:
        """Set the function to be used when converting items
        to their textual presentation.
        """
        if itemToText is None:
            self.__dict__.pop('_itemToText', None)
        else:
            self._itemToText = itemToText
        self._formatAllItems()

    @protect
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Process key events. The :py:class:`ItemAdapter` supports
        the following keys:

        C: clear the currently selected entry

        Note: in a :py:class:`QComboBox` this event is only received
        if the combobox is closed (not while currently selecting an entry).
        """
        key = event.key()
        LOG.debug("ItemAdapter[%s].keyPressEvent: key=%d",
                  type(self).__name__, key)
        if key == Qt.Key_C:  # clear
            self._setCurrentItem(None)
        elif key == Qt.Key_Y:  # no itemToText function (inherit from super)
            self.setItemToText(None)
        elif key == Qt.Key_Z:  # simple str() as itemToText function (debug)
            self.setItemToText(str)
        elif hasattr(super(), 'keyPressEvent'):
            super().keyPressEvent(event)
        else:
            event.ignore()

    def debug(self) -> None:
        """Ouput debug information for this :py:class:`ItemAdapter`.
        """
        if hasattr(super(), 'debug'):
            super().debug()
        print(f"debug: ItemAdapter[{type(self).__name__}]: "
              f"with {self._countItems()} entries:")
        for index, item in enumerate(self._items()):
            print(f"debug:{'**' if item is self._currentItem() else '  '}"
                  f"({index+1}) {self._textForItem(item)} "
                  f"[{repr(item)}]")


class QAdaptedComboBox(ItemAdapter, QComboBox):
    """A :py:class:`QComboBox` implementing the
    :py:class:`ItemAdapter` interface.
    """

    #
    # methods to be implemented by subclasses
    #

    def _countItems(self) -> int:
        """Get the number of items in this :py:class:`QAdaptedComboBox`.
        """
        return self.count()

    def _items(self) -> Iterator[Any]:
        """An iterator for the items in this
        :py:class:`QAdaptedComboBox`.
        """
        for index in range(self.count()):
            yield self.itemData(index)

    def _texts(self) -> Iterator[str]:
        """An iterator for the texts presented by this
        :py:class:`QAdaptedComboBox`.
        """
        for index in range(self.count()):
            yield self.itemText(index)

    def _addItem(self, item: Any) -> None:
        """Add an item to this :py:class:`QAdaptedComboBox`.
        It is assumed that the item is not yet contained in this
        :py:class:`QAdaptedComboBox`.
        """
        self.addItem(self._textForItem(item), item)
        self._formatItem(item)

    def _removeItem(self, item: Any) -> None:
        """Remove an item from this :py:class:`QAdaptedComboBox`.
        It is assumed that the item is contained in this
        :py:class:`QAdaptedComboBox`, otherwise a
        :py:class:`ValueError` is raised.
        """
        self._removeItemAt(self._indexOfItem(item))

    def _removeItemAt(self, index: int) -> None:
        """Remove the item at the given index.
        """
        self.removeItem(index)

    def _removeText(self, text: str) -> None:
        """Remove the item with the given text. This may be
        overwritten by subclasses when a more efficient implementation
        is possible.
        """
        self._removeItemAt(self._indexOfText(text))

    def _formatItemAt(self, index: int) -> None:
        """Format the item at the given index to reflect
        the state of the underlying item.

        This method may be extended by subclasses.
        """
        self.setItemText(index, self._textForItem(self.itemData(index)))

    def _formatItem(self, item: Any) -> None:
        """Update the format of the item's presentation
        in this :py:class:`QAdaptedComboBox`
        to reflect its state.
        """
        self._formatItemAt(self._indexOfItem(item))

    def _formatAllItems(self) -> None:
        """Format all items in this :py:class:`QAdaptedComboBox`.
        """
        for index in range(self.count()):
            self._formatItemAt(index)

    def _currentItem(self) -> Any:
        """Get the currently selected item.
        This may be `None` if no itm is selected.
        """
        return self.currentData()

    def _setCurrentItem(self, item: Any) -> None:
        """Select the given entry in this :py:class:`QAdaptedComboBox`.

        Arguments
        ---------
        item: Any
            The item to become the current item. If the item is not
            contained in this :py:class:`QAdaptedComboBox` (e.g. if
            `item` is `None`), the current will be set to `None`.
        """
        try:
            self.setCurrentIndex(self._indexOfItem(item))
        except LookupError:
            # For an empty QComboBox or a QComboBox in which no
            # current entry is set, the index is -1 (which is also
            # returned by QComboBox.findText if the entry is not found).
            self.setCurrentIndex(-1)


class QAdaptedListWidget(ItemAdapter, QListWidget):
    """A :py:class:`QListWidget` implementing the
    :py:class:`ItemAdapter` interface.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._formater = None

    def setListWidgetItemFormater(self, formater:
                                  Callable[[QListWidgetItem], None]) -> None:
        """Set a formater for the list items.
        """
        self._formater = formater
        self._formatAllItems()

    def updateFormat(self) -> None:
        """Update the format of all items in this
        :py:class:`QAdaptedListWidget`.
        """
        self._formatAllItems()

    #
    # methods to be implemented by subclasses
    #

    def _countItems(self) -> int:
        """Get the number of items in this :py:class:`QAdaptedListWidget`.
        """
        return self.count()

    def _qitem(self, item: Any) -> QListWidgetItem:
        """Get the :py:class:`QListWidgetItem` that holds the given
        item.
        """
        return next((qitem for qitem in self._qitems()
                     if qitem.data(Qt.UserRole) is item), None)

    def _qitems(self) -> Iterator[QListWidgetItem]:
        """An :py:class:`Iterator` for the :py:class:`QListWidgetItem`
        in this :py:class:`QAdaptedListWidget`.
        """
        for index in range(self.count()):
            yield self.item(index)

    def _formatQItem(self, qitem: QListWidgetItem) -> None:
        """Format the given :py:class:`QListWidgetItem` to reflect
        the state of the underlying item.

        This method may be extended by subclasses.
        """
        qitem.setText(self._textForItem(qitem.data(Qt.UserRole)))
        if self._formater is not None:
            self._formater(qitem)

    def _items(self) -> Iterator[Any]:
        """An iterator for the items in this
        :py:class:`QAdaptedComboBox`.
        """
        for qitem in self._qitems():
            yield qitem.data(Qt.UserRole)

    def _texts(self) -> Iterator[str]:
        """An iterator for the texts presented by this
        :py:class:`QAdaptedListWidget`.
        """
        for qitem in self._qitems():
            yield qitem.text()

    def _addItem(self, item: Any) -> None:
        """Add an item to this :py:class:`QAdaptedComboBox`.
        It is assumed that the item is not yet contained in this
        :py:class:`QAdaptedListWidget`.
        """
        qitem = QListWidgetItem(self._textForItem(item))
        qitem.setData(Qt.UserRole, item)
        self.addItem(qitem)
        self._formatQItem(qitem)

    def _formatItem(self, item: Any) -> None:
        """Update the format of the item's presentation
        in this :py:class:`QAdaptedListWidget`
        to reflect its state.
        """
        self._formatQItem(self._qitem(item))

    def _formatAllItems(self) -> None:
        """Format all items in this :py:class:`QAdaptedListWidget`.
        """
        for qitem in self._qitems():
            self._formatQItem(qitem)

    def _removeItem(self, item: Any) -> None:
        """Remove an item from this :py:class:`QAdaptedListWidget`.
        It is assumed that the item is contained in this
        :py:class:`QAdaptedComboBox`, otherwise a
        :py:class:`ValueError` is raised.
        """
        qitem = self.takeItem(self._indexOfItem(item))
        del qitem

    def _currentItem(self) -> Any:
        """Get the currently selected item.
        This may be `None` if no itm is selected.
        """
        qitem = self.currentItem()
        return None if qitem is None else qitem.data(Qt.UserRole)

    def _setCurrentItem(self, item: Any) -> None:
        """Select the given entry in this :py:class:`QAdaptedListWidget`.

        Arguments
        ---------
        item: Any
            The item to become the current item. If the item is not
            contained in this :py:class:`QAdaptedListWidget` (e.g. if
            `item` is `None`), the current will be set to `None`.
        """
        try:
            self.setCurrentRow(self._indexOfItem(item))
        except LookupError:
            self.setCurrentRow(-1)
