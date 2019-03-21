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
            self._updateList()

    def _updateList(self):
        # First remove all items from the list ...
        for i in range(self.count()):
            self.item(i).data(Qt.UserRole).remove_observer(self._viewObserver)
        self.clear()

        # ... and then rebuild the list
        if self._toolbox:
            interests = self._viewObserver.interests
            for observable in self._viewObserver.data(self._toolbox):
                item = QListWidgetItem(str(observable))
                item.setData(Qt.UserRole, observable)
                self._viewObserver.formatItem(item)
                self.addItem(item)
                observable.add_observer(self._viewObserver,
                                        interests=interests)

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
        object. These problems arise, as all list entry is observed,
        and in addition the viewed object (which can be one of the
        list entries) is observed for an `observable_change` event.
        The observable does not count the observers, and then multiple
        removals cause a KeyError.
        """
        interests = None

        def __init__(self, listWidget):
            self._listWidget = listWidget

        def data(self, toolbox: ToolboxView):
            return []  # to be implemented by subclasses

        def formatItem(self, item:QListWidgetItem) -> None:
            pass  # to be implemented by subclasses
