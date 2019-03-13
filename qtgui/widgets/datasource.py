from toolbox import Toolbox, View as ToolboxView
from datasources import Datasource, View as DatasourceView
from qtgui.utils import QObserver, protect
from .helper import QToolboxViewList

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QListWidgetItem

class QDatasourceList(QToolboxViewList, Datasource.Observer):
    """A list displaying the Datasources of a Toolbox.

    By providing a DatasourceView, the list becomes clickable, and
    selecting a Datasource from the list will set the observed
    Datasource, and vice versa, i.e. changing the observed datasource
    will change the current item in the list.
    """
    _datasource: DatasourceView=None

    def __init__(self, datasource: DatasourceView=None, **kwargs) -> None:
        """
        Arguments
        ---------
        toolbox: ToolboxView
        datasource: DatasourceView
        parent: QWidget
        """
        super().__init__(interest='datasources_changed', **kwargs)
        self._initUI()
        if datasource != self._datasource:
            self.setDatasourceView(datasource)

    def setToolboxView(self, toolbox: ToolboxView,
                       datasource: DatasourceView=None) -> None:
        super().setToolboxView(toolbox)
        if datasource is None and toolbox is not None:
            datasource = toolbox.datasource_controller
        self.setDatasourceView(datasource)

    def setDatasourceView(self, datasource: DatasourceView=None) -> None:
        interests = Datasource.Change('observable_changed')
        self._exchangeView('_datasource', datasource, interests=interests)
        self.setEnabled(datasource is not None)

    def currentDatasource(self) -> Datasource:
        item = self.currentItem()
        return item and item.data(Qt.UserRole) or None

    def datasource_changed(self, datasource: Datasource,
                           change: Datasource.Change) -> None:
        if change.observable_changed:
            self._updateCurrent(datasource)
        if change.state_changed or change.metadata_changed:
            self._updateItem(datasource)

    @protect
    def onItemClicked(self, item: QListWidgetItem):
        self._datasource(item.data(Qt.UserRole))

    def _listData(self):
        return self._toolbox.datasources

    def _viewInterests(self):
        return Datasource.Change('state_changed', 'metadata_changed')

    def _formatItem(self, item:QListWidgetItem) -> None:
        if item is not None:
            datasource = item.data(Qt.UserRole)
            item.setForeground(Qt.green if datasource.prepared else Qt.black)
