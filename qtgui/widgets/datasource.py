"""
.. moduleauthor:: Ulf Krumnack

.. module:: qtgui.widgets.datasource

This module contains widgets for viewing and controlling
:py:class:`Datasource`s. It aims at providing support for all
abstract interfaces defined in
`datasource.datasource`.
"""

# standard imports
from typing import Union
import logging

# Qt imports
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QHideEvent, QFontMetrics, QDoubleValidator
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QSizePolicy
from PyQt5.QtWidgets import QWidget, QPushButton, QDoubleSpinBox


# toolbox imports
from toolbox import Toolbox
from dltb.datasource import Datasource, Datafetcher
from dltb.datasource import Indexed, Random, Livesource
from dltb.base.register import RegisterEntry, InstanceRegisterEntry

# GUI imports
from ..utils import QObserver, QPrepareButton, protect
from .register import ToolboxAdapter
from .register import QRegisterListWidget, QRegisterComboBox
from .register import QInstanceRegisterEntryController
from .navigation import QIndexControls as QBaseIndexControls

# logging
LOG = logging.getLogger(__name__)


class DatasourceAdapter(ToolboxAdapter, qobservables={
        Toolbox: {'datasource_changed', 'datasources_changed'}}):
    # pylint: disable=abstract-method
    """A :py:class:`ToolboxAdapter` that is especially interested in
    :py:class:`Datasource`.
    """

    datasourceSelected = pyqtSignal(object)

    def updateFromToolbox(self) -> None:
        """Update the list from the :py:class:`Toolbox`.
        """
        self.updateFromIterable(self._toolbox.datasources)

    def toolbox_changed(self, _toolbox: Toolbox,
                        change: Toolbox.Change) -> None:
        # pylint: disable=invalid-name
        """React to a change in the :py:class:`Toolbox`. The only change
        of interest is a change of the current datasource. This
        will be reflected in the list.
        """
        if change.datasource_changed:  # the current datasource has changed
            self._formatAllItems()
        elif change.datasources_changed:  # the list of datasources has changed
            self.updateFromToolbox()

    def datasource(self) -> Datasource:
        """The currently selected Datasource in this
        :py:class:`QDatasourceList`.
        """
        item = self._currentItem()
        if self._toolbox is not None:
            # items are of type Datasource
            return item

        # items are of type InstanceRegisterEntry
        return None if item is None else item.obj

    def setDatasource(self, datasource: Datasource) -> None:
        """Set the current :py:class:`Datasource`.
        """
        if datasource is None:
            self._setCurrentItem(None)
        else:
            self._setCurrentText(datasource.key)


class QDatasourceListWidget(DatasourceAdapter, QRegisterListWidget):
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
        """
        super().__init__(register=Datasource.instance_register, **kwargs)

    @protect
    def _oncurrentIndexChanged(self, index: int) -> None:
        """A forward to map item selection to Datasource selection.
        """
        self.datasourceSelected.emit(self.itemData(index))


class QDatasourceComboBox(DatasourceAdapter, QRegisterComboBox):
    """A :py:class:`QComboBox` to select a :py:class:`Datasource`.
    """

    def __init__(self, **kwargs) -> None:
        """
        """
        super().__init__(register=Datasource.instance_register, **kwargs)
        self.currentIndexChanged.connect(self._oncurrentIndexChanged)

    @protect
    def _oncurrentIndexChanged(self, _index: int) -> None:
        """A forward to map item selection to Datasource selection.
        """
        self.datasourceSelected.emit(self.datasource())


class QDatafetcherObserver(QObserver, qobservables={
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

    def datafetcher_changed(self, _datafetcher: Datafetcher,
                            info: Datafetcher.Change) -> None:
        # pylint: disable=invalid-name
        """React to a change in the state of the controlled
        :py:class:`Datafetcher`.
        """
        if info.state_changed or info.datasource_changed:
            self.update()


class QLoopButton(QPushButton, QDatafetcherObserver):
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
        """Click on this :py:class:`QLoopButton` will start or stop
        looping.
        """
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


class QFramesPerSecondEdit(QDoubleSpinBox, QDatafetcherObserver, qobservables={
        Datafetcher: {'state_changed', 'datasource_changed',
                      'config_changed'}}):
    """A widget for adjusting the playback speed in frames per seconds.

    Collaboration with a :py:class:`Datafetcher`:
    When a `Datafetcher` is set for this `QFramesPerSecondEdit`, the
    displayed value will corresponding to the
    :py:prop:`Datafetcher.frames_per_second` of that `Datafetcher`.
    The `QFramesPerSecondEdit` will also observe a `Datafetcher`
    which provides the data to be displayed and will then adapt its state
    (enabled) depending on the the state of that `Datafetcher`
    (:py:prop:`Datafetcher.ready`).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.setRange(0.5, 10.0)
        self.setSingleStep(0.5)
        self.setMinimumWidth(QFontMetrics(self.font()).width('8') * 4)
        self.setToolTip("Datapoints per Second")

        # editingFinished: This signal is emitted when the Return or
        # Enter key is pressed or the line edit loses focus.
        self.valueChanged.connect(self.onValueChanged)

    def setDatafetcher(self, datafetcher: Datafetcher) -> None:
        """Set the datafetcher whose frames per second value is to
        be edited by this :py:class:`QFramesPerSecondEdit`.
        """
        super().setDatafetcher(datafetcher)
        self._datafetcher = datafetcher
        self.setValue(10.0 if datafetcher is None else
                      datafetcher.frames_per_second)

    def datafetcher_changed(self, datafetcher: Datafetcher,
                            info: Datafetcher.Change) -> None:
        # pylint: disable=invalid-name
        """React to a change in the state of the controlled
        :py:class:`Datafetcher`.
        """
        if info.config_changed:
            if datafetcher.frames_per_second != self.value():
                self.setValue(datafetcher.frames_per_second)
        super().datafetcher_changed(datafetcher, info)

    @protect
    def onValueChanged(self, value: float) -> None:
        """React to the `EditingFinished` signal of the line editor.  This
        signal is emitted when the Return or Enter key is pressed or
        the line edit loses focus.

        """
        if self._datafetcher is not None:
            frames_per_second = max(0.5, min(value, 10.0))
            self._datafetcher.frames_per_second = frames_per_second

    def update(self) -> None:
        """Update this QLoopButton based on the state of the
        :py:class:`Datafetcher`.
        """
        enabled = (self._datafetcher is not None and
                   self._datafetcher.ready)
        self.setEnabled(enabled)
        super().update()


class QSnapshotButton(QPushButton, QDatafetcherObserver):
    """A Button to control a :py:class:`Datasource` of type
    :py:class:`Livesource`. Pressing this button will will obtain a
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


class QRandomButton(QPushButton, QDatafetcherObserver):
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
        """Fetch a random item from the datasource.
        """
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


class QBatchButton(QPushButton, QDatafetcherObserver):
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
        self._batchSize = 8
        self.clicked.connect(self.onClicked)

    @protect
    def onClicked(self, _checked: bool):
        """Fetch a batch of data.
        """
        # FIXME[bug]: May throw a RuntimeError (when busy)
        if isinstance(self._datafetcher.datasource, Random):
            self._datafetcher.fetch(batch=self._batchSize, random=True)

    def update(self) -> None:
        """Update this QLoopButton based on the state of the
        :py:class:`Datasource`.
        """
        enabled = (self._datafetcher is not None and
                   self._datafetcher.randomable and
                   self._datafetcher.ready and
                   not self._datafetcher.looping)
        self.setEnabled(enabled)


class QIndexControls(QBaseIndexControls, QDatafetcherObserver, qobservables={
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
        """React to a change of the current index by fetching
        the corresponding entry.
        """
        LOG.info("QIndexControls: index changed=%d", index)
        # FIXME[bug]: May throw a RuntimeError (when busy)
        self._datafetcher.fetch(index=index)

    def datafetcher_changed(self, datafetcher: Datafetcher,
                            info: Datafetcher.Change) -> None:
        # pylint: disable=invalid-name
        """React to a change in the state of the controlled
        :py:class:`Datafetcher`.
        """
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
            LOG.debug("QIndexControls: data_changed: data available: %s",
                      data is not None)
            if data is not None and datafetcher.indexable:
                # The index may have changed
                index = data[0].index if data.is_batch else data.index
                self.setIndex(index)
                LOG.debug("QIndexControls: index=%d", index)

        if info.state_changed:
            LOG.debug("QIndexControls: Datafecher state changed: ready=%s",
                      datafetcher.ready)
            enabled = enabled and datafetcher.ready

        self.update(enabled)


class QDatasourceNavigator(QWidget, QObserver, qattributes={
        Toolbox: False, Datasource: False}, qobservables={
            Datafetcher: {'datasource_changed'}}):
    """The `QDatasourceNavigator` offers control widgets to navigate through
    a :py:class:`Datasource`. The actual controls depend on the type
    of the datasource and the arrangement can be adapted by providing
    layout parameters.

    The :py:class:`Datasource` to navigate can be set via the
    :py:meth:`setDatasource` method. Alternatively, a :py:class:`Toolbox`
    can be set using the :py:class:`setToolbox` method.
    If a toolbox is set, its current datasource will be used and it
    is no longer allowed to set the Datasource via :py:meth:`setDatasource`.

    The `QDatasourceNavigator` performs navigation via a
    :py:class:`Datafetcher`.  User actions are directly executed by
    that `Datafetcher`.  There are no signals emitted on any
    navigation actions.  Interested parties should observe the
    underlying `Datafetcher`.  If a `Toolbox` is set, its `Datafetcher`
    will be used, otherwise a private `Datafetcher` is created.

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

    _selector: QDatasourceComboBox

    _prepareButton: QPrepareButton


    Arguments
    ---------
    datasource_selector:
        A flag indicating if a :py:class:`QDatasourceSelector`
        should be included in this `QDatasourceNavigator`.

    style:
        The layout style of this `QDatasourceNavigator`. Valid
        are `'wide'` (a one-row layout) and 'narrow'` (a multi-row layout).

    """

    def __init__(self, toolbox: Toolbox = None,
                 datasource: Datasource = None,
                 datasource_selector: bool = True,
                 style: str = 'narrow', **kwargs):
        """Initialization of the :py:class:`QDatasourceNavigator`.

        Parameters
        ---------
        datasource: Datasource
            The datasource to be controlled by this
            :py:class:`QDatasourceNavigator`
        """
        super().__init__(**kwargs)
        self._initUI(datasource_selector)
        self._layoutUI(style)

        self.setToolbox(toolbox)
        if datasource is not None:
            self.setDatasource(datasource)
        elif self._selector is not None:
            self.setDatasource(self._selector.datasource())

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

        self._framesPerSecondEdit = QFramesPerSecondEdit()
        self.addAttributePropagation(Datafetcher, self._framesPerSecondEdit)

        self._batchButton = QBatchButton()
        self.addAttributePropagation(Datafetcher, self._batchButton)

        if selector:
            self._selector = QDatasourceComboBox()
            self._selector.datasourceSelected.\
                connect(self._onDatasourceSelected)
            self.addAttributePropagation(Toolbox, self._selector)

            self._prepareButton = QPrepareButton()
            self.addAttributePropagation(Datasource, self._selector)
        else:
            self._selector = None
            self._prepareButton = None

    def _layoutUI(self, style: str) -> None:
        row = QHBoxLayout()
        row2 = QHBoxLayout() if style == 'narrow' else row
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
        row.addWidget(self._framesPerSecondEdit)
        row.addWidget(self._batchButton)

        if style == 'narrow':
            layout = QVBoxLayout()
            row2.addStretch()
            layout.addLayout(row2)
            layout.addLayout(row)
        else:
            layout = row
            self.setSizePolicy(QSizePolicy.MinimumExpanding,
                               QSizePolicy.Fixed)

        self.setLayout(layout)
        print(self.sizeHint())
        print("random:", self._randomButton.sizeHint())
        print("index:", self._indexControls.sizeHint(),
              self._indexControls.minimumSize())
        print("first:", self._indexControls._firstButton.sizeHint())
        print("field:", self._indexControls._indexField.sizeHint())
        print("label:", self._indexControls._indexLabel.sizeHint())

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
            # the datafetcher will notify all interested parties
            # (including us) that the datasource has changed.
            self.setDatafetcher(Datafetcher(datasource))

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
        # pylint: disable=invalid-name
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
        self._snapshotButton.setVisible(isinstance(datasource, Livesource))
        loopable = (isinstance(datasource, Livesource) or
                    isinstance(datasource, Indexed))
        self._loopButton.setVisible(loopable)
        self._framesPerSecondEdit.setVisible(loopable)
        self._batchButton.setVisible(isinstance(datasource, Datasource))
        if self._prepareButton is not None:
            self._prepareButton.setPreparable(datasource)

    @protect
    def _onDatasourceSelected(self, datasource: Datasource) -> None:
        """The signal `datasourceChanged` is sent whenever the selection of
        the datasource in the QComboBox changes, either through user
        interaction or programmatically.
        """
        # FIXME[hack]: we need a more consistent way of what to store
        # (Datasource or InstanceRegisterEntry) and what to report ...
        if isinstance(datasource, InstanceRegisterEntry):
            datasource = datasource.obj
        self.setDatasource(datasource)


class QDatasourceController(QInstanceRegisterEntryController, qobservables={
        Toolbox: {'datasource_changed'}}):
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
        # pylint: disable=attribute-defined-outside-init
        super()._layoutUI()
        self._toolboxButton = QPushButton("Set as Toolbox Datasource")
        self._toolboxButton.clicked.connect(self._onToolboxButtonClicked)

        layout = self.layout()
        layout.addWidget(self._toolboxButton)

    def datasource(self) -> object:
        """The :py:class:`Datasource` currently controlled by this
        :py:class:`QDatabaseController` or None if no datasource is
        controlled or the datasource is not yet initialized.
        """
        if self._registerEntry is None:
            return None
        return self._registerEntry.obj

    def setRegisterEntry(self, entry: Union[RegisterEntry, str]) -> None:
        """Set a new :py:class:`ClassRegisterEntry` to control.
        """
        super().setRegisterEntry(entry)
        self._updateToolboxButton()

    def setToolbox(self, toolbox: Toolbox) -> None:
        """Set the toolbox for this :py:class:`QDatasourceController`.
        """
        self._updateToolboxButton()

    def _updateToolboxButton(self) -> None:
        """Update the state of the toolbox button. The button
        is checked, if there is a toolbox and a datasource,
        and that datasource is the current datasource of the toolbox.
        """
        toolbox = self.toolbox()
        datasource = self.datasource()
        if toolbox is None or datasource is None:
            self._toolboxButton.setChecked(False)
            self._toolboxButton.setEnabled(False)
        else:
            checked = toolbox.datasource is datasource
            self._toolboxButton.setChecked(checked)
            self._toolboxButton.setEnabled(not checked)

    @protect
    def _onToolboxButtonClicked(self, _checked: bool):
        """React to a click on the "Set as Toolbox Datasource"-Button.
        If a :py:class:`Toolbox` is present, this currently selected
        :py:class:`Datasource` will be assigned as the active datasource
        of the toolbox.
        """
        if self._toolbox is not None:
            self._toolbox.datasource = self.datasource()

    def toolbox_changed(self, toolbox: Toolbox,
                        change: Toolbox.Change) -> None:
        # pylint: disable=invalid-name
        """The resources of the :py:class:`Toolbox` have changed.
        """
        LOG.debug("%s.toolbox_changed(%s, %s)",
                  type(self).__name__, type(toolbox).__name__, change)
        if 'datasource_changed' in change:
            self._updateToolboxButton()
