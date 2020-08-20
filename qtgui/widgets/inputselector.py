from typing import Any

from datasource import Datasource, Random, Indexed, Data, ClassIdentifier

from ..utils import QObserver, protect


from util.image import Region, PointsBasedLocation


from ..utils import QObserver
from .datasource import QDatasourceSelectionBox, QInputNavigator

from PyQt5.QtWidgets import QWidget, QGroupBox, QHBoxLayout
from PyQt5.QtWidgets import QHBoxLayout, QSizePolicy


from toolbox import Toolbox

# FIXME[old]: this class seems to be no longer in use - check
# and recycle parts that may still be useful!
class QInputSelector(QWidget, QObserver, qobservables={
        # FIXME[hack]: check what we are really interested in ...
        Datasource: Datasource.Change.all()}):
    """A Widget to select input data (probably images).

    This Widget consists of two subwidgets:
    1. a :py:class:`QDatasourceSelectionBox` to select a
       :py:class:`Datasource` and
    2. a :py:class:`QInputNavigator` to navigate in the
       :py:class:`Datasource`.

    There are different modes of selection: from an array, from a file,
    from a directory or from some predefined data source.
    Modes: there are currently different modes ('array' or 'dir').
    For each mode there exist a corresponding data source. The widget
    has a current mode and will provide data only from the
    corresponding data source.

    FIXME[attention]
    ATTENTION: this mode concept may be changed in future versions! It
    seems more plausible to just maintain a list of data sources.

    .. warning:: This docstring must be changed once the mode concept
    is thrown overboard

    Attributes
    ----------
    _source_selector: QDatasourceSelectionBox
        A widget to change the currently selected datasource.
    _navigator: QInputNavigator
        A widget to navigate in the currently selected datasource.
    _index: int
        The index of the current data entry.
    """
    _source_selector: QDatasourceSelectionBox = None
    _navigator: QInputNavigator = None

    def __init__(self, toolbox: Toolbox=None, parent=None):
        '''Initialization of the QInputSelector.

        Parameters
        ----------
        parent  :   QWidget
                    The parent argument is sent to the QWidget constructor.
        '''
        super().__init__(parent)
        self._initUI()
        self.setToolbox(toolbox)

    def _initUI(self):
        self._source_selector = QDatasourceSelectionBox()
        self._navigator = QInputNavigator()

        sourceBox = QGroupBox('Data sources')
        sourceBox.setLayout(self._source_selector.layout())

        navigationBox = QGroupBox('Navigation')
        navigationBox.setLayout(self._navigator.layout())

        layout = QHBoxLayout()
        layout.addWidget(sourceBox)
        layout.addWidget(navigationBox)
        self.setLayout(layout)

