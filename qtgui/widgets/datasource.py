"""
.. moduleauthor:: Ulf Krumnack

.. module:: qtgui.widgets.datasource

This module contains widgets for viewing and controlling
:py:class:`Datasource`s. It aims at providing support for all
abstract interfaces defined in 
`datasource.datasource`.
"""

# standard imports
from typing import Iterator
import logging

# Qt imports
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QHideEvent, QKeyEvent
from PyQt5.QtWidgets import QListWidgetItem

# GUI imports
from ..utils import QObserver, protect
from .register import QRegisterList, QRegisterController, QPrepareButton

# toolbox imports
from base import Observable, MetaRegister
from toolbox import Toolbox
from datasource import Datasource, View as DatasourceView

# logging
LOG = logging.getLogger(__name__)


class QDatasourceList(QRegisterList, QObserver, qobservables={
        Toolbox: {'datasource_changed', 'datasources_changed'}}):
    """A list displaying the Datasources of a Toolbox.

    By providing a DatasourceView, the list becomes clickable, and
    selecting a Datasource from the list will set the observed
    Datasource, and vice versa, i.e. changing the observed datasource
    will change the current item in the list.

    Entries of this list can be of different nature:
    * instances of class :py:class:`Datasource`
    * ids (str) to identify registered (but potentially uninitialized)
      instances of the class :py:class:`Datasource`
    * subclasses of :py:class:`Datasource`, that allow to instantiate
      a new  datasource (provided that sufficient initialization parameters
      are provided).

    A :py:class:`QDatasourceList` can be run in different modes:
    * display the (initialized) datasources of a :py:class:`Toolbox`
    * display the (potentiallly uninitialized) datasources registered
      at class :py:class:`Datasource`
    * display a list of subclasses of class :py:class:`Datasource`
      that may be used to initialize a new datasource.

    The list displayed by :py:class:`QDatasourceList` is intended to
    reflect the current state of affairs, reflecting changes in the
    mentioned lists or individual datasources.  Hence it implements
    different observer interfaces and registers as observer whenever
    possible (even for individual datasources displayed in the list).
    """

    def __init__(self, **kwargs) -> None:
        """
        Arguments
        ---------
        toolbox: Toolbox
        datasource: DatasourceView
        parent: QWidget
        """
        super().__init__(interests=Toolbox.Change('datasource_changed'),
                         **kwargs)

    def _toolboxIterator(self) -> Iterator:
        return (((str(data), data) for data in self._toolbox.datasources)
                if self._toolbox else super()._toolboxIterator())

    def toolbox_changed(self, toolbox: Toolbox, change: Toolbox.Change) -> None:
        """React to a change in the :py:class:`Toolbox`. The only change
        of interest is a change of the current datasource. This
        will be reflected in the list.
        """
        if change.datasource_changed:  # the current datasource has changed
            pass  # FIXME[todo]
        elif change.datasources_changed:  # the list of datasources has changed
            pass  # FIXME[todo]

    def currentDatasource(self) -> Datasource:
        # FIXME[question]: is this used anywhere?
        item = self.currentItem()
        key = item and item.isKey() and item.key()
        return Datasource[key] if Datasource.key_is_initialized(key) else None



from .register import RegisterItemList, QRegisterItemComboBox

class DatasourceItemList(RegisterItemList, qobservables={
        Toolbox: {'datasources_changed', 'datasource_changed'},
        Datasource: {'state_changed'}}):
    """
    There are different ways to use a :py:class:`DatasourceItemList`:
    * Standalone: selectable :py:class:`Datasource`\ s have to be
      added and removed explicitly by calling
      :py:method:`addDatasource` and :py:class:`removeDatasource`.
    * Directly observing the py:class:`Datasource` register. The
      :py:class:`DatasourceItemList` will be updated when new datasources
      are initialized.
    * Based on a :py:class:`Toolbox`: datasources registered with the
      :py:class:`Toolbox` can be selected. The :py:class:`DatasourceItemList`
      will be updated when datasources are added or removed from the
      :py:class:`Toolbox`.

    A :py:class:`DatasourceItemList` can be configured to only show a
    sublist of the actual datasource list. This is useful when run
    in Register or Toolbox mode, but only a subset of the available
    datasources shall be presented:
    * preparation: just show datasources that are prepared
    * superclass: just show datasources that are subclasses of a given
      superclass, e.g., only face datasources.

    Attributes
    ----------
    _toolbox: Toolbox
        A :py:class:`Toolbox` providing a list of
        :py:class:`Datasource`\ s. Can be set with :py:meth:`setToolbox`,
        making the :py:class:`DatasourceItemList` an observer of that
        :py:class:`Toolbox`, reacting to 'datasources_changed' signals.

    """
        
    def __init__(self, toolbox: Toolbox = None,
                 datasource: Datasource = None, **kwargs) -> None:
        """Initialization of the :py:class:`DatasourceItemList`.

        Parameters
        ----------
        """
        super().__init__(register=Datasource, **kwargs)
        #self.setToolbox(toolbox)
        self.setDatasource(datasource)

    def setToolbox(self, toolbox: Toolbox) -> None:
        print(f"\nWarning: Setting Datasource Toolbox ({toolbox}) is currently disabled ...\n")
        # palette = QPalette();
        # palette.setColor(QPalette.Background, Qt.darkYellow)
        # self.setAutoFillBackground(True)
        # self.setPalette(palette)
        return
        if toolbox is not None:
            self._updateFromIterator(toolbox.datasources)
        else:
            self._updateFromIterator(iter(()))

    def toolbox_changed(self, toolbox: Toolbox, change: Toolbox.Change) -> None:
        """React to changes in the Toolbox. We are only interested
        when the list of datasources has changed, in which case we will
        update the content of the QComboBox to reflect the available
        Datasources.
        """
        print(f"DatasourceItemList.toolbox_changed({change})")
        print(f"\nWarning: Datasource Toolbox changes ({toolbox}, {change}]) are currently disabled ...\n")
        return
        if change.datasources_changed:
            self._updateFromIterator(toolbox.datasources)

    def setDatasourcek(self, datasource: Datasource) -> None:
        self.setEnabled(datasource is not None)

    def datasource_changed(self, datasource: Datasource,
                           change: Datasource.Change) -> None:
        print(f"DatasourceItemList.datasource_changed({datasource}, {change})")

    def currentDatasource(self) -> Datasource:
        """The currently selected :py:class:`Datasource`.
        This may be `None` if no datasource is selected.
        """
        return self.currentData()

    @protect
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Process key events. The :py:class:`QImageView` supports
        the following keys:

        r: toggle the keepAspectRatio flag
        """
        key = event.key()
        #LOG.debug("debug: DatasourceItemList.keyPressEvent: key=%d", key)
        print(f"debug: DatasourceItemList.keyPressEvent: key={key}")
        if key == Qt.Key_T: # t = debug toolbox:
            if self._toolbox is None:
                print("No Toolbox")
            else:
                print("Toolbox datasources:")
                for datasource in self._toolbox.datasources:
                    print("datasource:", datasource)
        else:
            super().keyPressEvent(event)


class QDatasourceSelector(QRegisterItemComboBox, DatasourceItemList):
    """A widget to select a :py:class:`Datasource` from a list of
    :py:class:`Datasource`\ s.

    """
    def __init__(self, **kwargs) -> None:
        """Initialization of the :py:class:`QDatasourceSelector`.

        Parameters
        ----------
        """
        print(f"QDatasourceSelector.__init__({kwargs})")
        super().__init__(**kwargs)
        self.activated[str].connect(self.onActivated)

    @protect
    def onActivated(self, name) -> None:
        print(f"QDatasourceSelector.onActivated({name})")
            


from PyQt5.QtWidgets import QWidget, QPushButton

from datasource import Datasource, Loop, Indexed

class QDatasourceObserver(QObserver, qobservables={
        Datasource: {'busy_changed', 'state_changed'}}):
    """A QObserver observing a Datasource. This is intended to be
    inherited by classes observing a :py:class:`Datasource`.

    Attributes
    ----------
    _datasource: Datasource
        A datasource controller used by this Button to control the
        (loop mode) of the Datasource.
    """
    _interests: Datasource.Change = None

    def __init__(self, datasource: Datasource = None,
                 interests: Datasource.Change = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._interests = interests or \
            Datasource.Change('busy_changed', 'state_changed')
        self.setDatasource(datasource)

    def datasource_changed(self, datasource: Datasource,
                           info: Datasource.Change) -> None:
        if info.state_changed:
            self.update()


class QLoopButton(QPushButton, QDatasourceObserver):
    """A Button to control a :py:class:`Datasource` of type
    :py:class:`Loop`. Such datasource can be in a loop mode, meaning
    that they continously produce new data (e.g., webcam, movies,
    etc.).

    The :py:class:`QLoopButton` can observe a :py:class:`Datasource`
    and adapt its appearance and function based on the state of the
    datasource.

    """

    def __init__(self, text: str='Loop', **kwargs) -> None:
        super().__init__(text, **kwargs)
        self.setCheckable(True)
        self.clicked.connect(self.onClicked)

    @protect
    def onClicked(self, checked: bool) -> None:
        LOG.info("QLoopButton.onClicked(%s): %s", checked, self._datasource)
        if isinstance(self._datasource, Loop):
            LOG.info("QLoopButton: looping=%s", self._datasource.looping)
            self._datasource.loop(checked)

    def update(self) -> None:
        """Update this QLoopButton based on the state of the
        :py:class:`Datasource`.
        """
        enabled = (isinstance(self._datasource, Loop) and
                   self._datasource.prepared and
                   (self._datasource.looping or not self._datasource.busy))
        checked = enabled and self._datasource.looping
        self.setEnabled(enabled)
        self.setChecked(checked)

    def hideEvent(self, event: QHideEvent) -> None:
        if isinstance(self._datasource, Loop) and self.isChecked():
            self._datasource.loop(False)
        super().hideEvent(event)

from PyQt5.QtWidgets import QWidget, QPushButton

from datasource import Datasource, Loop, Snapshot

class QSnapshotButton(QPushButton, QDatasourceObserver):
    """A Button to control a :py:class:`Datasource` of type
    :py:class:`Loop`. Pressing this button will will obtain a
    snapshot from the datasource.

    The :py:class:`QSnapshotButton` can observe a :py:class:`Datasource`
    and adapt its appearance and function based on the state of that
    datasource.

    The :py:class:`QSnapshotButton` will only be enabled if a
    :py:class:`Datasource` was registered with the
    :py:meth:`setDatasource` method and if this datasource is not busy
    (e.g., by looping).

    """

    def __init__(self, text: str='Snapshot', **kwargs) -> None:
        super().__init__(text, **kwargs)
        self.clicked.connect(self.onClicked)

    def onClicked(self):
        """Create a snapshot as reaction to a button click.
        """
        if isinstance(self._datasource, Snapshot):
            self._datasource.fetch_snapshot()

    def update(self) -> None:
        """Update this QLoopButton based on the state of the
        :py:class:`Datasource`.
        """
        enabled = (isinstance(self._datasource, Snapshot) and
                   self._datasource.prepared and
                   not self._datasource.busy)
        self.setEnabled(enabled)


from PyQt5.QtWidgets import QWidget, QPushButton

from datasource import Datasource, Loop, Random

class QRandomButton(QPushButton, QDatasourceObserver):
    """A Button to control a :py:class:`Datasource` of type
    :py:class:`datasource.Random`. Pressing this button will
    obtain a entry from the datasource.

    The :py:class:`QRandomButton` can observe a :py:class:`Datasource`
    and adapt its appearance and function based on the state of that
    datasource. The :py:class:`QRandomButton` will only be enabled if a
    :py:class:`Datasource` was registered with the
    :py:meth:`setDatasource` method and if this
    datasource is not busy (e.g., by looping).
    """

    def __init__(self, text: str='Random', **kwargs) -> None:
        super().__init__(text, **kwargs)
        self.clicked.connect(self.onClicked)

    def onClicked(self):
        if isinstance(self._datasource, Random):
            self._datasource.fetch(random=True)

    def update(self) -> None:
        """Update this QLoopButton based on the state of the
        :py:class:`Datasource`.
        """
        enabled = (isinstance(self._datasource, Random)
                   and self._datasource.prepared and
                   not self._datasource.busy)

        self.setEnabled(enabled)


class QBatchButton(QPushButton, QDatasourceObserver):
    """A Button to control a :py:class:`Datasource` of type
    :py:class:`datasource.Random`. Pressing this button will
    obtain a entry from the datasource.

    The :py:class:`QRandomButton` can observe a :py:class:`Datasource`
    and adapt its appearance and function based on the state of that
    datasource. The :py:class:`QRandomButton` will only be enabled if
    a :py:class:`Datasource` was registered with the
    :py:meth:`setDatasource` method and if this datasource is not busy
    (e.g., by looping).

    """

    def __init__(self, text: str='Batch', **kwargs) -> None:
        super().__init__(text, **kwargs)
        self._batch_size = 8
        self.clicked.connect(self.onClicked)

    def onClicked(self):
        if isinstance(self._datasource, Random):
            self._datasource.fetch(batch=self._batch_size, random=True)

    def update(self) -> None:
        """Update this QLoopButton based on the state of the
        :py:class:`Datasource`.
        """
        enabled = (isinstance(self._datasource, Datasource)
                   and self._datasource.prepared and
                   not self._datasource.busy)
        self.setEnabled(enabled)


from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontMetrics, QIntValidator, QIcon
from PyQt5.QtWidgets import QHBoxLayout, QSizePolicy
from PyQt5.QtWidgets import QWidget, QPushButton, QLineEdit, QLabel
from .navigation import QIndexControls as QBaseIndexControls

class QIndexControls(QBaseIndexControls, QDatasourceObserver, qobservables={
        Datasource: {'busy_changed', 'state_changed', 'data_changed'}}):
    """A group of Widgets to control an :py:class:`Indexed`
    :py:class:`Datasource`. The controls allow to select elements
    from the datasource based on their index.

    The :py:class:`QIndexControls` can observe a :py:class:`Datasource`
    and adapt their appearance and function based on the state of that
    datasource.

    The :py:class:`QIndexControls` will only be enabled if a
    :py:class:`Datasource` was registered with the
    :py:meth:`setDatasource` method and if this
    datasource is not busy (e.g., by fetching or looping).

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.indexChanged.connect(self.onIndexChanged)

    @protect
    def onIndexChanged(self, index: int) -> None:
        self._datasource.fetch_index(index)

    def setDatasource(self, datasource: Datasource) -> None:
        #print(f"\nQIndexControls.setDatasource({datasource}): self._datasource={self._datasource}")
        #print(f"QIndexControls._qObserverHelpers={self._qObserverHelpers}")
        #self._datasource = datasource  # FIXME[hack]: should be done by QObserver
        # FIXME[todo]: we should only observe this datasource, if it is
        # isinstance(datasource, Indexed)!
        if isinstance(datasource, Indexed) and datasource.prepared:
            self.setElements(len(datasource))
            if datasource.fetched and not datasource.data.is_batch:
                self.setIndex(datasource.data.index)
        else:
            self.setElements(-1)
        LOG.debug("QIndexControls._elements=%d", self._elements)

    def datasource_changed(self, datasource: Datasource,
                           info: Datasource.Change) -> None:
        #print(f"QIndexControls.datasource_changed({datasource}, {info})")
        #print(f"QIndexControls._qObserverHelpers={self._qObserverHelpers}")
        #print(f"QIndexControls._elements={self._elements}")
        enabled = not datasource.busy
        if info.state_changed:
            #self.update(not datasource.busy)
            if isinstance(datasource, Indexed) and datasource.prepared:
                self.setElements(len(datasource))
            else:
                self.setElements(-1)
                enabled = False
        if info.data_changed:
            if (isinstance(datasource, Indexed) and
                datasource.fetched and not datasource.data.is_batch and
                datasource.data.index is not None): # FIXME[bug]: the last condition should not happen, but seems to be the case wie datasource 'Movie'
                # The index may have changed
                self.setIndex(datasource.data.index)
        self.update(enabled)


class QDatasourceNavigator(QWidget, QObserver, qattributes={Datasource: False},
                           qobservables={Toolbox: {'datasource_changed'}}):
    """The QDatasourceNavigator offers control widgets to navigate through
    a :py:class:`Datasource`. The actual controls depend on the type
    of the datasource and the arrangement can be adapted by providing
    layout parameters.

    The :py:class:`Datasource` to navigate can be set via the
    :py:meth:`setDatasource` method. Alternatively, a :py:class:`Toolbox`
    can be set using the :py:class:`setToolbox` method.
    If a toolbox is set, its current datasource will be used and it
    is no longer allowed to set the Datasource via :py:meth:`setDatasource`.
    
    _indexControls: QIndexControls
        controls for an indexed datasource (Indexed)

    _randomButton: QRandomButton
        select random entry from the Datasource (Random)

    _snapshotButton: QSnapshotButton
        Snapshot button (Snapshot)

    _loopButton: QLoopButton
        start/stop looping the Datasource (Loop)

    _batchButton: QLoopButton
        start/stop looping the Datasource (Loop)

    _selector: QDatasourceSelector

    _prepareButton: QPrepareButton

    """

    def __init__(self, datasource: Datasource = None, **kwargs):
        """Initialization of the :py:class:`QDatasourceNavigator`.

        Parameters
        ---------
        datasource: Datasource
            The datasource to be controlled by this
            :py:class:`QDatasourceNavigator`
        """
        super().__init__(**kwargs)
        self._initUI()
        self._layoutUI()

        # if we have a toolbox, the datasource is taken from that
        # toolbox. Without a toolbox, the datasource has to be set
        # explicitly.
        if self._toolbox is None:
            self.setDatasource(datasource)

    def _initUI(self, selector: bool = True) -> None:
        """Initialize the user interface.

        """
        self._indexControls = QIndexControls()
        self.addAttributePropagation(Datasource, self._indexControls)
        
        self._randomButton = QRandomButton()
        self.addAttributePropagation(Datasource, self._randomButton)

        self._snapshotButton = QSnapshotButton()
        self.addAttributePropagation(Datasource, self._snapshotButton)

        self._loopButton = QLoopButton()
        self.addAttributePropagation(Datasource, self._loopButton)

        self._batchButton = QBatchButton()
        self.addAttributePropagation(Datasource, self._batchButton)

        if selector:
            self._selector = QDatasourceSelector()
            self._selector.currentIndexChanged.\
                connect(self.onCurrentDatasourceChanged)
            self._prepareButton = QPrepareButton()
        else:
            self._selector = None
            self._prepareButton = None

    def _layoutUI(self):
        two_rows = True
        row = QHBoxLayout()
        row2 = QHBoxLayout() if two_rows else row
        if self._selector is not None:
            row.addWidget(self._selector)
        if self._prepareButton is not None:
            row.addWidget(self._prepareButton)

        row2.addStretch()
        row2.addWidget(self._indexControls)
        row.addStretch()
        row.addWidget(self._randomButton)
        row.addWidget(self._snapshotButton)
        row.addWidget(self._loopButton)
        row.addWidget(self._batchButton)

        if two_rows:
            layout = QVBoxLayout()
            row2.addStretch()
            layout.addLayout(row2)
            layout.addLayout(row)
        else:
            layout = row

        self.setLayout(layout)

    def setToolbox(self, toolbox: Toolbox) -> None:
        """Set the toolbox for this :py:class:`QDatasourceNavigator`.
        If not None, the :py:class:`QDatasourceNavigator` will obtain
        its :py:class:`Datasource` from that toolbox. Otherwise it
        will run in standalone mode and the :py:class:`Datasource`
        has to be set explicitly.
        """
        LOG.debug("QDatasourceNavigator.setToolbox(%s)", toolbox)

    def setDatasource(self, datasource: Datasource) -> None:
        """Set the datasource for this :py:class:`QDatasourceNavigator`.
        Depending on the type of the new datasource, controls will
        become visible.
        """
        LOG.debug("QDatasourceNavigator.setDatasource(%s)", datasource)
        self._indexControls.setVisible(isinstance(datasource, Indexed))
        self._randomButton.setVisible(isinstance(datasource, Random))
        self._snapshotButton.setVisible(isinstance(datasource, Snapshot))
        self._loopButton.setVisible(isinstance(datasource, Loop))
        self._batchButton.setVisible(isinstance(datasource, Datasource))
        if self._prepareButton is not None:
            self._prepareButton.setPreparable(datasource)

    def toolbox_changed(self, toolbox: Toolbox, info) -> None:
        """React to a change of the Toolbox datasource.
        """
        LOG.debug("QDatasourceNavigator.toolbox_changed(%s, %s)",
                  toolbox, info)
        if info.datasource_changed:
            self.setDatasource(toolbox and toolbox.datasource)

    @protect
    def onCurrentDatasourceChanged(self, index: int) -> None:
        """The signal `currentIndexChanged` is sent whenever the currentIndex
        in the combobox changes either through user interaction or
        programmatically.
        """
        datasource = self._selector.currentDatasource()
        if self._toolbox is not None:
            self._toolbox.datasource = datasource
        else: 
            self.setDatasource(datasource)


class QDatasourceController(QRegisterController, Datasource.Observer):
    """The :py:class:`QDatasourceController` can control general
    aspects of a datasource. This includes:
    * initialization (in case the datasource initialization is registered)
    * prepare/unprepare for initialized datasources
    Display a description of the :py:class:`Datasource`.

    Attributes
    ----------

    **Signals:**

    A :py:class:`QDatasourceController` emits different signals 
    corresponding to the different actions that can be initiated:
    * initializeKeyClicked(str): The value is the key.
    * initializeClassClicked(str): The value is the fully qualified class name.
    
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(Datasource, **kwargs)

    def _layoutUI(self):
        super()._layoutUI()

        # FIXME[hack]
        self.button = QPushButton("Set as Toolbox Datasource")

        layout = self.layout()
        layout.addWidget(self.button)


# FIXME[old]: seems not to be used anymore
class QInputNavigator(QWidget, QObserver, qobservables={
        Datasource: {'busy_changed', 'state_changed', 'data_changed'}}):
    """A :py:class:`QInputNavigator` displays widgets to navigate in the
    Datasource.  The actual set of widgets depends on the type of
    Datasource and will be adapated when the Datasource is changed.

    """
    #_controller: DatasourceController = None

    _q_controller: QDatasourceController = None
    #
    _infoDatasource: QLabel = None
    
    # display the state of the Datasource:
    # "none" / "unprepared" / "busy" / "ready"
    _stateLabel = None

    # prepare or unprepare the Datasource
    _prepareButton = None

    _navigator: QDatasourceNavigator = None

    def __init__(self, datasource: Datasource = None, **kwargs):
        """Initialization of the QInputNavigator.

        Arguments
        ---------
        datasource: Datasource
            A Controller allowing to navigate in the Datasource.
        """
        super().__init__(**kwargs)
        self._initUI()
        self._layoutUI()
        self.setDatasource(datasource)

    def _initUI(self):
        """Initialize the user interface.

        """
        self._q_controller = QDatasourceController()

        self._prepareButton = QPushButton('Prepare')
        self._prepareButton.setSizePolicy(QSizePolicy.Maximum,
                                          QSizePolicy.Maximum)
        self._prepareButton.setCheckable(True)
        self._prepareButton.clicked.connect(self._prepareButtonClicked)
        
        self._infoDatasource = QLabel()
        self._stateLabel = QLabel()
        
        self._navigator = QDatasourceNavigator()

    def _layoutUI(self):
        """The layout of the :py:class:`QInputNavigator` may change
        depending on the :py:class:`Datasource` it controls.

        """
        # FIXME[todo]: we may also add some configuration options
        
        # We have no Layout yet: create initial Layout
        layout = QVBoxLayout()
        layout.addWidget(self._q_controller)
        layout.addWidget(self._infoDatasource)
        layout.addWidget(self._stateLabel)

        # buttons 2: control the datsource
        buttons2 = QHBoxLayout()
        buttons2.addWidget(self._prepareButton)
        buttons2.addStretch()
        layout.addLayout(buttons2)

        layout.addWidget(self._navigator)

        self.setLayout(layout)
            
        #if self._controller is None or not self._controller:
        #    # no datasource controller or no datasource: remove buttons
        #    if self._buttons is not None:
        #        for button in self._buttonList:
        #            button.setParent(None)
        #        self._layout.removeItem(self._buttons)
        #        self._buttons = None
        #else:
        #    # we have a datasource: add the buttons
        #    if self._buttons is None:
        #        self._buttons = QHBoxLayout()
        #        for button in self._buttonList:
        #            self._buttons.addWidget(button)
        #        self._layout.addLayout(self._buttons)

    def _enableUI(self):
        enabled = bool(self._controller)
        self._prepareButton.setEnabled(bool(self._controller))
        if bool(self._controller):
            self._prepareButton.setChecked(self._controller.prepared)

    @protect
    def _prepareButtonClicked(self, checked: bool):
        '''Callback for clicking the 'next' and 'prev' sample button.'''
        if not self._controller:
            return
        if checked:
            self._controller.prepare()
        else:
            self._controller.unprepare()

    def setDatasource(self, datasource: Datasource) -> None:
        self._navigator.setDatasource(datasource)

    def datasource_changed(self, datasource: Datasource, info) -> None:
        """React to chanes in the datasource. Changes of interest are:
        (1) change of datasource ('observable_changed'): we may have to
            adapt the controls to reflect the type of Datasource.
        (2) usability of the datasource ('state_changed', 'busy_changed'):
            we may have to disable some controls depending on the state
            (prepared, busy).
        (3) change of selected image (data_changed): we may want to
            reflect the selection in our controls.
        """
        if info.observable_changed:
            self._updateState()
            self._enableUI()

        if info.observable_changed or info.state_changed:
            self._updateDescription(datasource)

        if info.state_changed or info.busy_changed:
            self._updateState()
            self._enableUI()

    def _updateDescription(self, datasource):
        info = ''
        if datasource:
            text = datasource.get_description()
        else:
            text = "No datasource"
        self._infoDatasource.setText(text)
    
    def _updateState(self):
        if not self._controller:
            text = "none"
        elif not self._controller.prepared:
            text = "unprepared"
        elif self._controller.busy:
            text = "busy"
            busy_message = self._controller.busy_message
            if busy_message:
                text += f" ({busy_message})"
        else:
            text = "ready"
        self._stateLabel.setText("State: " + text)






















#
# FIXME[old]
#



from datasource import (Datasource, DataArray, DataFile, DataDirectory,
                         DataWebcam, Video)
from toolbox import Toolbox
from qtgui.utils import QObserver, protect

from PyQt5.QtWidgets import (QWidget, QPushButton, QRadioButton, QGroupBox,
                             QHBoxLayout, QVBoxLayout, QSizePolicy,
                             QInputDialog, QComboBox,
                             QFileDialog, QListView, QAbstractItemView,
                             QTreeView)

# QInputSourceSelector
# FIXME[old]: seems not to be used
class QDatasourceSelectionBox(QWidget, QObserver, qobservables={
        Toolbox: {'datasources_changed'}}):
    """The QDatasourceSelectionBox provides a control to select a data
    source. It is mainly a graphical user interface to the datasource
    module, adding some additional logic.

    The QDatasourceSelectionBox provides four main ways to select input
    data:
    1. a collection of predefined data sets from which the user can
       select via a dropdown box
    2. a file or directory, that the user can select via a file browser
    3. a camera
    4. a URL (not implemented yet)
    """

    _toolbox: Toolbox=None

    # FIXME[old]:
    _datasource: Datasource = None
    
    def __init__(self, toolbox: Toolbox=None, **kwargs):
        """Initialization of the :py:class:`QDatasourceSelectionBox`.

        Parameters
        ---------
        parent: QWidget
            The parent argument is sent to the QWidget constructor.
        """
        super().__init__(**kwargs)
        self._initUI()
        self._layoutUI()
        self.setToolbox(toolbox)

    def _initUI(self):
        '''Initialize the user interface.'''

        #
        # Differnt types of Datasources
        #
        self._radioButtons = {
            'Name': QRadioButton('Predefined'),
            'Filesystem': QRadioButton('Filesystem'),
            'Webcam': QRadioButton('Webcam'),
            'Video': QRadioButton('Video')
        }
        self._radioButtons['Video'].setEnabled(False)
        for b in self._radioButtons.values():
            b.clicked.connect(self._radioButtonChecked)

        #
        # A list of predefined datasources
        #
        dataset_names = list(Datasource.register_keys())
        self._datasetDropdown = QComboBox()
        self._datasetDropdown.addItems(dataset_names)
        self._datasetDropdown.currentIndexChanged.\
            connect(self._predefinedSelectionChange)
        self._datasetDropdown.setEnabled(False)

        #
        # A button to select the Datasource
        #
        self._openButton = QPushButton('Open')
        self._openButton.clicked.connect(self._openButtonClicked)

    def _layoutUI(self):

        size_policy = self._datasetDropdown.sizePolicy()
        size_policy.setRetainSizeWhenHidden(True)
        self._datasetDropdown.setSizePolicy(size_policy)

        self._openButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        radioLayout = QVBoxLayout()
        for b in self._radioButtons.values():
            radioLayout.addWidget(b)

        buttonsLayout = QVBoxLayout()
        buttonsLayout.addWidget(self._datasetDropdown)
        buttonsLayout.addWidget(self._openButton)

        layout = QHBoxLayout()
        layout.addLayout(radioLayout)
        layout.addLayout(buttonsLayout)
        self.setLayout(layout)

    def datasource_changed(self, datasource, info):
        '''The QDatasourceSelectionBox is only affected by changes of
        the Datasource.
        '''
        if info.datasource_changed:
            self._setDatasource(datasource)

    def _setDatasource(self, datasource: Datasource):
        if isinstance(datasource, Datasource):
            self._radioButtons['Name'].setChecked(True)
            key = datasource.key
            index = self._datasetDropdown.findText(key)
            if index == -1:
                pass # should not happen!
            elif index != self._datasetDropdown.currentIndex():
                self._datasetDropdown.setCurrentIndex(index)
        elif isinstance(datasource, DataWebcam):
            self._radioButtons['Webcam'].setChecked(True)
        elif isinstance(datasource, Video):
            self._radioButtons['Video'].setChecked(True)
        elif isinstance(datasource, DataFile):
            self._radioButtons['Filesystem'].setChecked(True)
        elif isinstance(datasource, DataDirectory):
            self._radioButtons['Filesystem'].setChecked(True)
        else:
            self._radioButtons['Filesystem'].setChecked(True)

        self._datasetDropdown.setEnabled(self._radioButtons['Name'].isChecked())
        # FIXME[old]
        # if isinstance(datasource, DataArray):
        #     info = (datasource.getFile()
        #             if isinstance(datasource, DataFile)
        #             else datasource.get_description())
        #     if info is None:
        #         info = ''
        #     if len(info) > 40:
        #         info = info[0:info.find('/', 10) + 1] + \
        #             '...' + info[info.rfind('/', 0, -20):]
        #     self._radioButtons['Name'].setText('Name: ' + info)
        # elif isinstance(datasource, DataDirectory):
        #     self._radioButtons['Filesystem'].setText('File: ' +
        #                                     datasource.getDirectory())
        #####################################################################
        #                Disable buttons, if necessary                      #
        #####################################################################


    def _radioButtonChecked(self):
        '''Callback for clicking the radio buttons.'''
        name = self.sender().text()
        if name == 'Name':
            self._datasetDropdown.setEnabled(True)
            self._openButton.setText('Load')
        elif name == 'Filesystem':
            self._datasetDropdown.setEnabled(False)
            self._openButton.setText('Open')
        elif name == 'Webcam':
            self._datasetDropdown.setEnabled(False)
            self._openButton.setText('Run')
        elif name == 'Video':
            self._datasetDropdown.setEnabled(False)
            self._openButton.setText('Play')
        self._datasetDropdown.setEnabled(self._radioButtons['Name'].isChecked())

    @protect
    def _openButtonClicked(self):
        """An event handler for the ``Open`` button. Pressing this
        button will select a datasource. How exactly this works
        depends on the type of the Datasource, which is selected by
        the radio buttons.
        """

        datasource = None
        if self._radioButtons['Name'].isChecked():
            # Name: this will select a predefined Datasource based on its
            # name which is selectend in the _datasetDropdown QComboBox.

            #self._datasetDropdown.setVisible(True)
            name = self._datasetDropdown.currentText()
            datasource = Datasource[name]

        elif self._radioButtons['Filesystem'].isChecked():
            # CAUTION: I've converted the C++ from here
            # http://www.qtcentre.org/threads/43841-QFileDialog-to-select-files-AND-folders
            # to Python. I'm pretty sure this makes use of
            # implemention details of the QFileDialog and is thus
            # susceptible to sudden breakage on version change. It's
            # retarded that there is no way for the file dialog to
            # accept either files or directories at the same time so
            # this is necessary.
            #
            # UPDATE: It appears setting the selection mode is
            # unnecessary if only single selection is desired. The key
            # insight appears to be using the non-native file dialog
            dialog = QFileDialog(self)
            dialog.setFileMode(QFileDialog.Directory)
            dialog.setOption(QFileDialog.DontUseNativeDialog, True)
            nMode = dialog.exec_()
            fname = dialog.selectedFiles()[0]
            import os
            if os.path.isdir(fname):
                datasource = DataDirectory(fname)
            else:
                datasource = DataFile(fname)
        elif self._radioButtons['Webcam'].isChecked():
            datasource = DataWebcam()
        elif self._radioButtons['Video'].isChecked():
            # FIXME[hack]: use file browser ...
            # FIXME[problem]: the opencv embedded in anoconda does not
            # have ffmpeg support, and hence cannot read videos
            datasource = Video("/net/home/student/k/krumnack/AnacondaCON.avi")

        LOG.info("QDatasourceSelectionBox._openButtonClicked: "
                 "We have selected the following Datasource:")
        if datasource is None:
            print("QDatasourceSelectionBox._openButtonClicked:   -> no Datasource")
        else:
            print(f"QDatasourceSelectionBox._openButtonClicked:   type: {type(datasource)}")
            print(f"QDatasourceSelectionBox._openButtonClicked:   prepared:  {datasource.prepared}")

        try:
            datasource.prepare()
            print(f"QDatasourceSelectionBox._openButtonClicked:   len:  {len(datasource)}")
            print(f"QDatasourceSelectionBox._openButtonClicked:   description:  {datasource.datasource.get_description()}")
        except Exception as ex:
            print(f"QDatasourceSelectionBox._openButtonClicked:   preparation failed ({ex})!")
            datasource = None

        # FIXME[hack]: not really implemented. what should happen is:
        # - change the datasource for the DatasourceView/Controller
        #    -> this should notify the observer 'observable_changed'
        # what may happen in response is
        # - add datasource to some list (e.g. toolbox)
        # - emit some pyqt signal?
        if datasource is not None:
            if self._toolbox:
                # Set the datasource of the Toolbox.
                # This will also insert the dataset in the Toolbox's list
                # if datasource, if it is not already in there.
                self._toolbox.datasource = datasource

    @protect
    def _predefinedSelectionChange(self,i):
        if self._radioButtons['Name'].isChecked():
            self._datasetDropdown.setVisible(True)
            key = self._datasetDropdown.currentText()
            datasource = Datasource[key]

            if self._datasource is not None:
                self._datasource(datasource)
            if self._toolbox is not None:
                self._toolbox.datasource = datasource

    def setToolbox(self, toolbox: Toolbox) -> None:
        self._exchangeView('_toolbox', toolbox,
                           interests=Toolbox.Change('datasources_changed'))

    def toolbox_changed(self, toolbox: Toolbox, change):
        self._setDatasource(self._toolbox.datasource)

