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
from PyQt5.QtWidgets import QHBoxLayout, QSizePolicy
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel

# toolbox imports
from base import Observable, MetaRegister
from toolbox import Toolbox
from datasource import Datasource, Datafetcher
from datasource import Indexed, Random, Snapshot, Loop

# GUI imports
from ..utils import QObserver, protect
from .register import QRegisterList, QRegisterController, QPrepareButton
from .register import RegisterItemList, QRegisterItemComboBox
from .navigation import QIndexControls as QBaseIndexControls

# logging
LOG = logging.getLogger(__name__)


class QDatasourceList(QRegisterList, QObserver, qobservables={
        Toolbox: {'datasource_changed', 'datasources_changed'}}):
    """A list displaying the Datasources of a Toolbox.

    By providing a Datasource, the list becomes clickable, and
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
        datasource: Datasource
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


class DatasourceItemList(RegisterItemList, qobservables={
        Toolbox: {'datasources_changed', 'datasource_changed'}}):
    """There are different ways to use a :py:class:`DatasourceItemList`:
    * Standalone: selectable :py:class:`Datasource`\\ s have to be
      added and removed explicitly by calling
      :py:meth:`addDatasource` and :py:class:`removeDatasource`.
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
        self.setToolbox(toolbox)
        self.setDatasource(datasource)

    def setToolbox(self, toolbox: Toolbox) -> None:
        """Set a :py:class:`Toolbox` from which a list of datasources can
        be obtained.
        """
        # palette = QPalette();
        # palette.setColor(QPalette.Background, Qt.darkYellow)
        # self.setAutoFillBackground(True)
        # self.setPalette(palette)
        return
        if toolbox is not None:
            self._updateFromIterator(toolbox.datasources)
        else:
            self._updateFromIterator(iter(()))

    def toolbox_changed(self, toolbox: Toolbox,
                        change: Toolbox.Change) -> None:
        """React to changes in the Toolbox. We are only interested
        when the list of datasources has changed, in which case we will
        update the content of the QComboBox to reflect the available
        Datasources.
        """
        if change.datasources_changed:
            self._updateFromIterator(toolbox.datasources)

    def currentDatasource(self) -> Datasource:
        """Get the currently selected :py:class:`Datasource`.
        This may be `None` if no datasource is selected.
        """
        return self.currentItem()

    def setCurrentDatasource(self, datasource: Datasource) -> None:
        """Select the given :py:class:`Datasource`.

        Arguments
        ---------
        datasource: Datasource
            The datasource to become the currently selected element
            in this list. `None` will deselect the current element.

        Raises
        ------
        ValueError:
            The given :py:class:`Datasource` is not an element of this
            :py:class:`DatasourceItemList`.
        """
        self.setCurrentItem(datasource)

    @protect
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Process key events. The :py:class:`QImageView` supports
        the following keys:

        r: toggle the keepAspectRatio flag
        """
        key = event.key()
        LOG.debug("DatasourceItemList.keyPressEvent: key=%d", key)

        if key == Qt.Key_T:  # t = debug toolbox:
            if self._toolbox is None:
                print("No Toolbox")
            else:
                print("Toolbox datasources:")
                for datasource in self._toolbox.datasources:
                    print("datasource:", datasource)
        else:
            super().keyPressEvent(event)


class QDatasourceSelector(QRegisterItemComboBox, DatasourceItemList,
                          qobservables={Datasource: {'state_changed'}}):
    """A widget to select a :py:class:`Datasource` from a list of
    :py:class:`Datasource`\ s.

    """
    datasourceSelected = pyqtSignal(object)

    def __init__(self, **kwargs) -> None:
        """Initialization of the :py:class:`QDatasourceSelector`.

        Parameters
        ----------
        """
        super().__init__(**kwargs)
        self.currentIndexChanged.connect(self._oncurrentIndexChanged)

    @protect
    def _oncurrentIndexChanged(self, index: int) -> None:
        """A forward to map item selection to Datasource selection.
        """
        self.datasourceSelected.emit(self.itemData(index))

    def datasource_changed(self, datasource: Datasource,
                           change: Datasource.Change) -> None:
        """React to a change of an observed datasource.
        """
        LOG.debug("QDatasourceSelector: datasource %s changed %s.",
                  datasource, change)
        self._updateItems()

    def _addItem(self, item: Datasource) -> None:
        """Add an item to this :py:class:`QRegisterItemComboBox`.
        It is assumed that the item is not yet contained in this
        :py:class:`QRegisterItemComboBox`.
        """
        super()._addItem(item)
        self.observe(item, interests=Datasource.Change('state_changed'))

    def _removeItem(self, item: Datasource) -> None:
        """Remove an item from this :py:class:`QRegisterItemComboBox`.  It is
        assumed that the item is contained in this
        :py:class:`QRegisterItemComboBox`, otherwise a
        :py:class:`ValueError` is raised.

        """
        self.unobserve(item)
        super()._removeItem(item)


class QDatasourceObserver(QObserver, qobservables={
        Datafetcher: {'state_changed', 'datasource_changed'}}):
    """A QObserver observing a :py:class:`Datafetcher`. This is intended
    to be inherited by classes observing a :py:class:`Datafetcher`.

    Attributes
    ----------
    _datafetcher: Datafetcher
        A :py:class:`Datafetcher` used by this Button to control the
        (loop mode) of the Datasource.
    """
    _interests: Datasource.Change = None

    def __init__(self, datafetcher: Datafetcher = None,
                 interests: Datafetcher.Change = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._interests = interests or \
            Datafetcher.Change('busy_changed', 'state_changed')
        self.setDatafetcher(datafetcher)

    def datafetcher_changed(self, datafetcher: Datafetcher,
                            info: Datafetcher.Change) -> None:
        if info.state_changed or info.datasource_changed:
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

    def __init__(self, text: str = 'Loop', **kwargs) -> None:
        super().__init__(text, **kwargs)
        self.setCheckable(True)
        self.clicked.connect(self.onClicked)

    @protect
    def onClicked(self, checked: bool) -> None:
        LOG.info("QLoopButton: looping=%s", self._datafetcher.looping)
        self._datafetcher.looping = checked

    def update(self) -> None:
        """Update this QLoopButton based on the state of the
        :py:class:`Datafetcher`.
        """
        enabled = (self._datafetcher is not None and
                   self._datafetcher.loopable and
                   self._datafetcher.ready)
        checked = enabled and self._datafetcher.looping
        self.setEnabled(enabled)
        self.setChecked(checked)

    def hideEvent(self, event: QHideEvent) -> None:
        """When hiding the button, we should stop the loop,
        assuming it makes no sense to waste resources by playing
        it in the background.
        """
        if self._datafetcher is not None and self._datafetcher.loopable:
            self._datafetcher.looping = False
        super().hideEvent(event)


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

    def __init__(self, text: str = 'Snapshot', **kwargs) -> None:
        super().__init__(text, **kwargs)
        self.clicked.connect(self.onClicked)

    @protect
    def onClicked(self, _checked: bool):
        """Create a snapshot as reaction to a button click.
        """
        # FIXME[bug]: May throw a RuntimeError (when busy)
        self._datafetcher.fetch(snapshot=True)

    def update(self) -> None:
        """Update this QLoopButton based on the state of the
        :py:class:`Datasource`.
        """
        self.setEnabled(self._datafetcher is not None and
                        self._datafetcher.snapshotable and
                        self._datafetcher.ready and
                        not self._datafetcher.looping)


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

    def __init__(self, text: str = 'Random', **kwargs) -> None:
        super().__init__(text, **kwargs)
        self.clicked.connect(self.onClicked)

    @protect
    def onClicked(self, _checked: bool):
        # FIXME[bug]: May throw a RuntimeError (when busy)
        self._datafetcher.fetch(random=True)

    def update(self) -> None:
        """Update this QLoopButton based on the state of the
        :py:class:`Datasource`.
        """
        enabled = (self._datafetcher is not None and
                   self._datafetcher.randomable and
                   self._datafetcher.ready and
                   not self._datafetcher.looping)
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

    def __init__(self, text: str = 'Batch', **kwargs) -> None:
        super().__init__(text, **kwargs)
        self._batch_size = 8
        self.clicked.connect(self.onClicked)

    @protect
    def onClicked(self, _checked: bool):
        # FIXME[bug]: May throw a RuntimeError (when busy)
        if isinstance(self._datafetcher.datasource, Random):
            self._datafetcher.fetch(batch=self._batch_size, random=True)

    def update(self) -> None:
        """Update this QLoopButton based on the state of the
        :py:class:`Datasource`.
        """
        enabled = (self._datafetcher is not None and
                   self._datafetcher.randomable and
                   self._datafetcher.ready and
                   not self._datafetcher.looping)
        self.setEnabled(enabled)


class QIndexControls(QBaseIndexControls, QDatasourceObserver, qobservables={
        Datafetcher: {'state_changed', 'data_changed',
                      'datasource_changed', 'prepared_changed'}}):
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
        LOG.info("QIndexControls: index changed=%d", index)
        # FIXME[bug]: May throw a RuntimeError (when busy)
        self._datafetcher.fetch(index=index)

    def datafetcher_changed(self, datafetcher: Datafetcher,
                            info: Datafetcher.Change) -> None:
        LOG.debug("QIndexControls: datafetcher %s changed %s",
                  datafetcher, info)
        enabled = not datafetcher.looping

        if info.datasource_changed or info.prepared_changed:
            datasource = datafetcher.datasource
            # we can obtain the length only from a prepared datasource
            if datafetcher.indexable and datasource.prepared:
                self.setElements(len(datasource))
            else:
                self.setElements(-1)
                enabled = False

        if info.data_changed:
            data = datafetcher.data
            if datafetcher.indexable and datafetcher.fetched:
                # The index may have changed
                index = data[0].index if data.is_batch else data.index
                self.setIndex(index)
                LOG.debug("QIndexControls: index=%d", index)

        if info.state_changed:
            enabled = enabled and datafetcher.ready

        self.update(enabled)


class QDatasourceNavigator(QWidget, QObserver, qattributes={
        Toolbox: False, Datasource: False},
        qobservables={Datafetcher: {'datasource_changed'}}):
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

    def __init__(self, toolbox: Toolbox = None,
                 datasource: Datasource = None,
                 datasource_selector: bool = True, **kwargs):
        """Initialization of the :py:class:`QDatasourceNavigator`.

        Parameters
        ---------
        datasource: Datasource
            The datasource to be controlled by this
            :py:class:`QDatasourceNavigator`
        """
        super().__init__(**kwargs)
        self._initUI(datasource_selector)
        self._layoutUI()

        self.setToolbox(toolbox)
        if datasource is not None:
            self.setDatasource(datasource)
        elif self._selector is not None:
            self.setDatasource(self._selector.currentDatasource())

    def _initUI(self, selector: bool = True) -> None:
        """Initialize the user interface.

        """
        self._indexControls = QIndexControls()
        self.addAttributePropagation(Datafetcher, self._indexControls)

        self._randomButton = QRandomButton()
        self.addAttributePropagation(Datafetcher, self._randomButton)

        self._snapshotButton = QSnapshotButton()
        self.addAttributePropagation(Datafetcher, self._snapshotButton)

        self._loopButton = QLoopButton()
        self.addAttributePropagation(Datafetcher, self._loopButton)

        self._batchButton = QBatchButton()
        self.addAttributePropagation(Datafetcher, self._batchButton)

        if selector:
            self._selector = QDatasourceSelector()
            self._selector.datasourceSelected.\
                connect(self._onDatasourceSelected)
            self.addAttributePropagation(Toolbox, self._selector)

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
        self.setDatafetcher(Datafetcher() if toolbox is None else
                            toolbox.datafetcher)

    def setDatasource(self, datasource: Datasource) -> None:
        """Set the datasource for this :py:class:`QDatasourceNavigator`.
        Depending on the type of the new datasource, controls will
        become visible.
        """
        LOG.debug("QDatasourceNavigator.setDatasource(%s)", datasource)
        if self._datafetcher is not None:
            self._datafetcher.datasource = datasource
        elif datasource is not None:
            setDatafetcher(Datafetcher(datasource))

    def setDatafetcher(self, datafetcher: Datafetcher) -> None:
        """Set the :py:class:`Datafetcher` for this
        :py:class:`QDatasourceNavigator`.
        Depending on the type of the :py:class:`Datasource` from which
        data are fetched, the controls of this
        :py:class:`QDatasourceNavigator` will become visible or hidden.
        """
        LOG.debug("QDatasourceNavigator.setDatafetcher(%s)", datafetcher)
        self._updateDatasource(datafetcher and datafetcher.datasource)

    def datafetcher_changed(self, datafetcher: Datafetcher, info) -> None:
        """React to a change of the Toolbox datasource.
        """
        LOG.debug("QDatasourceNavigator.datafetcher_changed(%s, %s)",
                  datafetcher, info)
        if info.datasource_changed:
            self._updateDatasource(datafetcher.datasource)

    def _updateDatasource(self, datasource: Datasource) -> None:
        """Update the navigation controls based on a
        :py:class:`Datasource`.
        Some control elements will be hidden if not applicable for
        that datasource.

        Arguments
        ---------
        datasource: Datasource
            The datasource to which to adapt. May be `None`.
        """
        self._indexControls.setVisible(isinstance(datasource, Indexed))
        self._randomButton.setVisible(isinstance(datasource, Random))
        self._snapshotButton.setVisible(isinstance(datasource, Snapshot))
        self._loopButton.setVisible(isinstance(datasource, Loop))
        self._batchButton.setVisible(isinstance(datasource, Datasource))
        if self._prepareButton is not None:
            self._prepareButton.setPreparable(datasource)

    @protect
    def _onDatasourceSelected(self, datasource: Datasource) -> None:
        """The signal `datasourceChanged` is sent whenever the selection of
        the datasource in the QComboBox changes, either through user
        interaction or programmatically.
        """
        # the datafetcher will notify all interested parties (including us)
        # that the datasource has changed.
        self._datafetcher.datasource = datasource


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
            text = datasource.description
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
# FIXME[old]: seems not to be used (only by widgets.inputselector)
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
        #             else datasource.description)
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
            print("QDatasourceSelectionBox._openButtonClicked:"
                  "   -> no Datasource")
        else:
            print(f"QDatasourceSelectionBox._openButtonClicked:"
                  "   type: {type(datasource)}")
            print(f"QDatasourceSelectionBox._openButtonClicked:"
                  "   prepared:  {datasource.prepared}")

        try:
            datasource.prepare()
            print(f"QDatasourceSelectionBox._openButtonClicked:"
                  "   len:  {len(datasource)}")
            print(f"QDatasourceSelectionBox._openButtonClicked:"
                  "   description:  {datasource.datasource.description}")
        except Exception as ex:
            print(f"QDatasourceSelectionBox._openButtonClicked:"
                  "   preparation failed ({ex})!")
            datasource = None

        # FIXME[hack]: not really implemented. what should happen is:
        # - change the datasource for the Datasource/Controller
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

