"""
File: resources.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
"""

# Generic imports
import logging

# Qt imports
from PyQt5.QtGui import QResizeEvent
from PyQt5.QtWidgets import QTabWidget, QGridLayout

# toolbox imports
from toolbox import Toolbox
from network import Network
from dltb.datasource import Datasource

# GUI imports
from .panel import Panel
from ..utils import QAttribute
from ..widgets.resource import QNetworkInspector
from ..widgets.resource import QDatasourceInspector
from ..widgets.resource import QToolInspector

# logging
LOG = logging.getLogger(__name__)



class ResourcesPanel(Panel, QAttribute, qattributes={
        Toolbox: False, Network: False, Datasource: False}):
    """A Panel for managing resources used by the Toolbox.

    The :py:class:`ResourcesPanel` can be used with or without a
    :py:class:`Toolbox`:

    * without a Toolbox, the select signals will set the observables
      of other view and control widgets.

    * with a Toolbox, the select signals will set the corresponding
      properties of the Toolbox. Views and control widgets will observe
      the Toolbox and to change their observable accordingly.

    There a currently the following sections in this panel:
    * Networks

    * Datasources

    * Tools

    _networkInspector: QNetworkInspector = None
    _datasourceInspector: QDatasourceInspector = None
    _toolInspector: QToolInspector = None
    """

    def __init__(self, toolbox: Toolbox = None, network: Network = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._initUI()
        self._layoutUI()
        self.setToolbox(toolbox)
        self.setNetwork(network)

    def _initUI(self) -> None:
        self._networkInspector = QNetworkInspector()
        self.addAttributePropagation(Toolbox, self._networkInspector)
        self.addAttributePropagation(Network, self._networkInspector)

        self._datasourceInspector = QDatasourceInspector()
        self.addAttributePropagation(Toolbox, self._datasourceInspector)
        # self.addAttributePropagation(Datasouce, self._datasourceInspector)

        self._toolInspector = QToolInspector()

    def _layoutUI(self, style: str = 'Tabs') -> None:
        layout = QGridLayout()
        if style == 'Grid':
            self._networkInspector.sizePolicy().setHorizontalStretch(1)
            layout.addWidget(self._networkInspector, 0, 0)

            self._datasourceInspector.sizePolicy().setHorizontalStretch(1)
            layout.addWidget(self._datasourceInspector, 0, 1)
        else:
            tabs = QTabWidget()
            tabs.setTabPosition(tabs.West)
            tabs.addTab(self._networkInspector, 'Networks')
            tabs.addTab(self._datasourceInspector, 'Datasources')
            tabs.addTab(self._toolInspector, 'Tools')

            # select your preferred resource ...
            tabs.setCurrentWidget(self._datasourceInspector)

            layout.addWidget(tabs, 0, 0)
        self.setLayout(layout)

    #
    # Datasource
    #

    def setDatasource(self, datasource: Datasource, key: str = None) -> None:
        """Set a :py:class:`Datasource` for this :py:class:`ResourcesPanel`.
        The datasource will be propagated to the
        :py:class:`QDatasourceInspector` of this :py:class:`ResourcesPanel`.

        """
        self._datasourceInspector.setDatasource(datasource, key)
