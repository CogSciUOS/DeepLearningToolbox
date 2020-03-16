from typing import Iterable

from base import Observable
from toolbox import Toolbox, View as ToolboxView
from base import View
from qtgui.utils import QObserver, protect

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QListWidget, QListWidgetItem

class QToolboxViewList(QListWidget, QObserver, Toolbox.Observer):
    """A list displaying the observables listed by a
    :py:class:`Toolbox`.

    By providing an appropriate View, the list becomes clickable, and
    selecting an object from the list will set the observed
    object of the View, and vice versa, i.e. changing the
    observed object will change the current item in the list.

    The list will show the names of the observables. In addition
    it will for each item store a reference to the observable and
    its id (by calling item.setData(Qt.UserRole, observable)).
    """
    _toolbox: ToolboxView=None
    _toolboxInterest: Toolbox.Change=None

    def __init__(self, toolbox: ToolboxView=None,
                 interest: str=None, parent=None) -> None:
        super().__init__(parent)
        self._toolboxInterest = interest
        self._viewObserver = self.ViewObserver(self)
        self._initUI()
        self.setToolboxView(toolbox)

    @protect
    def onItemClicked(self, item: QListWidgetItem):
        pass  # to be implemented by subclasses

    def _initUI(self):
        self.itemClicked.connect(self.onItemClicked)

    def setToolboxView(self, toolbox: ToolboxView) -> None:
        interests = Toolbox.Change(self._toolboxInterest)
        # FIXME[bug]: see below
        #print(f"QToolboxViewList: {interests})") 
        self._exchangeView('_toolbox', toolbox, interests=interests)

    def toolbox_changed(self, toolbox: Toolbox,
                        change: Toolbox.Change) -> None:
        # FIXME[bug]: I receive uninteresting changes here (e.g.'input_changed')
        # although I only want to see 'datasources_changed', e.g. when
        # selecting items in QDatasourceList.
        #print(f"QToolboxViewList.toolbox_changed({self}, {toolbox}, {change})")
        if self._toolboxInterest in change:
            self._updateListFromToolbox(toolbox)

    def addItem(self, item: QListWidgetItem) -> None:
        """Inserts the item at the end of the list widget.

        This adapts the :py:meth:`addItem` method of the superclass
        by taking care that observables are observed.
        """
        data = item.data(Qt.UserRole)
        if isinstance(data, Observable):
            data.add_observer(self._viewObserver,
                              interests=self._viewObserver.interests)
        super().addItem(item)

    def clear(self) -> None:
        """Removes all items and selections in the view.

        This adapts the :py:meth:`clear` method of the superclass
        by taking care that observers are removed from items.
        """
        for i in range(self.count()):
            data = self.item(i).data(Qt.UserRole)
            print(f"del: {self.item(i).text()}, data is of type {type(data)}.")
            if isinstance(data, Observable):
                data.remove_observer(self._viewObserver)
        super().clear()

    def _updateListFromToolbox(self, toolbox: Toolbox) -> None:
        """Update the items displayed in the list.  The new list of items to
        be displayed is provided by a :py:class:`Toolbox`.

        """
        self._updateList(toolbox=toolbox)

    def _updateListFromRegister(self, register) -> None:
        """Update the items displayed in the list.  The new list of items to
        be displayed is obtained from a :py:class:`RegisterMetaclass`.
        """
        self._updateList(register=register)

    def _updateList(self, toolbox: Toolbox=None, register=None) -> None:
        # First remove all items from the list ...
        self.clear()

        if toolbox is not None:
            for observable in self._viewObserver.data(self._toolbox): # FIXME[hack]: we need a ToolboxView, not just a Toolbox here
                item = QListWidgetItem(str(observable))
                item.setData(Qt.UserRole, observable)
                self._viewObserver.formatItem(item)
                self.addItem(item)

        if register is not None:
            for key in register.keys():
                item = QListWidgetItem(key)
                if not register.key_is_initialized(key):
                    item.setData(Qt.UserRole, key)
                    item.setForeground(Qt.gray)
                elif toolbox is None:  # avoid listing items twice
                    item.setData(Qt.UserRole, register[key])
                    self._viewObserver.formatItem(item)
                self.addItem(item)

    def _updateItem(self, observable: Observable) -> None:
        item = self._itemForObservable(observable)
        self._viewObserver.formatItem(item)

    def _updateCurrent(self, observable: Observable) -> None:
        item = self._itemForObservable(observable)
        self.setCurrentItem(item)                

    def _itemForObservable(self, observable: Observable) -> QListWidgetItem:
        for i in range(self.count()):
            item = self.item(i)
            if item.data(Qt.UserRole) == observable:
                return item
        return None

    class ViewObserver(QObserver):
        """This auxiliary class is provided to avoid problems due to
        multiple observation of an observable by an QToolboxViewList
        object. These problems arise, as all list entries are observed,
        and in addition the viewed object (which can be one of the
        list entries) is observed for an `observable_change` event.
        The observable does not count the observers, and then multiple
        removals cause a KeyError.
        """
        interests = None

        def __init__(self, listWidget):
            self._listWidget = listWidget

        def data(self, toolbox: ToolboxView) -> Iterable[Observable]:
            """Return an iterator for the data to be listed.
            
            The intended use is that the item provided by this
            iterator are included in the list, using its
            stringifictaion (via `str`) as label and the observable
            as `Qt.UserRole` data.

            Result
            ------
            An iterator. If no data are available, this should be
            an empty iterator, not None.
            """
            return []  # to be implemented by subclasses

        def formatItem(self, item:QListWidgetItem) -> None:
            pass  # to be implemented by subclasses
