from toolbox import View as ToolboxView
from network import Network, View as NetworkView
from qtgui.utils import QObserver, protect
from .helper import QToolboxViewList

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QListWidgetItem

class QNetworkList(QToolboxViewList, Network.Observer):
    """A list displaying the :py:class:`Network`s of a
    :py:class:`Toolbox`.

    By providing a NetworkView, the list becomes clickable, and
    selecting a Network from the list will set the observed
    network of the view, and vice versa, i.e. changing the
    observed network will change the current item in the list.
    """
    _network: NetworkView=None
    
    def __init__(self, network: NetworkView=None, **kwargs) -> None:
        """
        Arguments
        ---------
        toolbox: ToolboxView
        network: NetworkView
        parent: QWidget
        """
        super().__init__(interest='networks_changed', **kwargs)
        self.setNetworkView(network)

    def setNetworkView(self, network: NetworkView=None) -> None:
        interests = Network.Change('observable_changed')
        self._exchangeView('_network', network, interests=interests)
        self.setEnabled(network is not None)

    def currentNetwork(self) -> Network:
        item = self.currentItem()
        return item and item.data(Qt.UserRole) or None

    def network_changed(self, network: Network,
                        change: Network.Change) -> None:
        if change.observable_changed:
            self._updateCurrent(network)

    @protect
    def onItemClicked(self, item: QListWidgetItem):
        self._network(item.data(Qt.UserRole))

    class ViewObserver(QToolboxViewList.ViewObserver, Network.Observer):
        interests = Network.Change('busy_changed')

        def data(self, toolbox: ToolboxView):
            return toolbox.networks

        def formatItem(self, item:QListWidgetItem) -> None:
            if item is not None:
                network = item.data(Qt.UserRole)
                item.setForeground(Qt.red if network.busy else Qt.black)

        def network_changed(self, network: Network,
                            change: Network.Change) -> None:
            if change.busy_changed:
                self._listWidget._updateItem(network)
