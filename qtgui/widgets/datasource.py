"""
.. moduleauthor:: Ulf Krumnack

.. module:: qtgui.widgets.datasource

This module contains widgets for viewing and controlling
:py:class:`Datasource`s. It aims at providing support for all
abstract interfaces defined in 
`datasource.datasource`.
"""


from typing import Iterable

from base import Observable
from toolbox import (Toolbox, View as ToolboxView,
                     Controller as ToolboxController)
from datasource import Datasource, View as DatasourceView

from ..utils import QObserver, protect
from .helper import QToolboxViewList

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QListWidgetItem


class QDatasourceList(QToolboxViewList, QObserver, Datasource.Observer):
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

    A :py:class:`QDatasourceList` provides different signals that
    correspond to the different types of entries that can be selected:
    * keySelected:
    * classSelected:
    * instanceSelected: 
    """
    _datasource: DatasourceView=None

    keySelected = pyqtSignal(str)
    classSelected = pyqtSignal(type)
    instanceSelected = pyqtSignal(object)

    def __init__(self, datasource: DatasourceView=None, **kwargs) -> None:
        """
        Arguments
        ---------
        toolbox: ToolboxView
        datasource: DatasourceView
        parent: QWidget
        """
        super().__init__(interest='datasources_changed', **kwargs)
        self.setDatasourceView(datasource)
        # FIXME[todo]: toggle between name and id display (not implemented yet)
        self._showNames = True
        self._showPrepared = False  # only show prepared datasources

        self.keySelected.connect(self.onKeySelected)
        self.classSelected.connect(self.onClassSelected)
        self.instanceSelected.connect(self.onInstanceSelected)

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

    @property
    def showNames(self) -> bool:
        return self._showNames

    @showNames.setter
    def showNames(self, flag: bool) -> None:
        if self._showNames != flag:
            self._showNames = flag
            self.update()

    @property
    def showPrepared(self) -> bool:
        return self._showPrepared

    @showPrepared.setter
    def showPrepared(self, flag: bool) -> None:
        if self._showPrepared != flag:
            self._showPrepared = flag
            self.update()

    def update(self):
        for i in range(self.count()):
            item = self.item(i)
            data = item.data(Qt.UserRole)
            text = str(data)
            item.setText(f"[{text}]" if self._showNames else text)
            if self._showPrepared and isinstance(data, Datasource):
                item.setHidden(not data.prepared)
            else:
                item.setHidden(False)
        super().update()

    @protect
    def onItemClicked(self, item: QListWidgetItem):
        """Respond to a click in the datasource list.
        Assign the corresponding :py:class:`Datasource`
        to our :py:class:`DatatasourceController`.
        """
        data = item.data(Qt.UserRole)
        if isinstance(data, str):
            self.keySelected.emit(data)
        elif isinstance(data, type):
            self.classSelected.emit(data)
        elif isinstance(data, Datasource):
            self.instanceSelected.emit(data)
        else:
            raise ValueError("Invalid data associed with clicked item "
                             f"'{item.text()}': {data}")
        
    @protect
    def onKeySelected(self, key: str) -> None:
        print(f"Key selected: {key}")
        if isinstance(self._toolbox, ToolboxController):
            self._toolbox.add_datasource(key)

    @protect
    def onClassSelected(self, cls: type) -> None:
        print(f"Class selected: {cls}")

    @protect
    def onInstanceSelected(self, datasource: Datasource) -> None:
        print(f"Datasource selected: {datasource}")
        if self._datasource is not None:
            self._datasource(datasource)

    @protect
    def keyPressEvent(self, event) -> None:
        """Process key events. The :py:class:`QImageView` supports
        the following keys:

        r: toggle the keepAspectRatio flag
        """
        key = event.key()
        print(f"QDatasourceList: key pressed: {key}")
        if key == Qt.Key_N:
            self.showNames = not self.showNames
        elif key == Qt.Key_P:
            self.showPrepared = not self.showPrepared
        elif key == Qt.Key_R:
            self._updateListFromRegister(Datasource)
        elif key == Qt.Key_T:
            self._updateListFromToolbox(self._toolbox)
        else:
            super().keyPressEvent(event)
            

    class ViewObserver(QToolboxViewList.ViewObserver, Datasource.Observer):
        interests = Datasource.Change('state_changed', 'metadata_changed')
        
        def data(self, toolbox: ToolboxView) -> Iterable[Observable]:
            """Return an iterator for the
            :py:class:`Datasource`s of the
            :py:class:`Toolbox` to be listed.
            """
            return toolbox.datasources or []

        def formatItem(self, item:QListWidgetItem) -> None:
            if item is not None:
                datasource = item.data(Qt.UserRole)
                item.setForeground(Qt.green if datasource.prepared
                                   else Qt.black)

        def datasource_changed(self, datasource: Datasource,
                               change: Datasource.Change) -> None:
            if change.state_changed or change.metadata_changed:
                self._listWidget._updateItem(datasource)



from datasource import (Datasource, DataArray, DataFile, DataDirectory,
                         DataWebcam, DataVideo,
                         Controller as DatasourceController)
from toolbox import Toolbox, ToolboxController
from qtgui.utils import QObserver, protect

from PyQt5.QtWidgets import (QWidget, QPushButton, QRadioButton, QGroupBox,
                             QHBoxLayout, QVBoxLayout, QSizePolicy,
                             QInputDialog, QComboBox,
                             QFileDialog, QListView, QAbstractItemView,
                             QTreeView)

# QInputSourceSelector
class QDatasourceSelectionBox(QWidget, QObserver, Toolbox.Observer,
                              Datasource.Observer):
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

    _toolbox: ToolboxController=None

    # FIXME[old]:
    _datasource: DatasourceController=None
    
    def __init__(self, toolbox: ToolboxController=None, **kwargs):
        """Initialization of the :py:class:`QDatasourceSelectionBox`.

        Parameters
        ---------
        parent: QWidget
            The parent argument is sent to the QWidget constructor.
        """
        super().__init__(**kwargs)
        self._initUI()
        self._layoutUI()
        self.setToolboxController(toolbox)

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
        dataset_names = list(Datasource.keys())
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
            id = datasource.id
            index = self._datasetDropdown.findText(id)
            if index == -1:
                pass # should not happen!
            elif index != self._datasetDropdown.currentIndex():
                self._datasetDropdown.setCurrentIndex(index)
        elif isinstance(datasource, DataWebcam):
            self._radioButtons['Webcam'].setChecked(True)
        elif isinstance(datasource, DataVideo):
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
            datasource = DataVideo("/net/home/student/k/krumnack/AnacondaCON.avi")

        print("We have selected the following Datasource:")
        if datasource is None:
            print("  -> no Datasource")
        else:
            print(f"  type: {type(datasource)}")
            print(f"  prepared:  {datasource.prepared}")

        try:
            datasource.prepare()
            print(f"  len:  {len(datasource)}")
            print(f"  description:  {datasource.datasource.get_description()}")
        except Exception as ex:
            print(f"  preparation failed ({ex})!")
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
                self._toolbox.datasource_controller(datasource)

    @protect
    def _predefinedSelectionChange(self,i):
        if self._radioButtons['Name'].isChecked():
            self._datasetDropdown.setVisible(True)
            name = self._datasetDropdown.currentText()
            datasource = Datasource[name]

            if self._datasource is not None:
                self._datasource(datasource)
            if self._toolbox is not None:
                self._toolbox.set_datasource(datasource)

    def setToolboxController(self, toolbox: ToolboxController) -> None:
        self._exchangeView('_toolbox', toolbox,
                           interests=Toolbox.Change('datasources_changed'))

    def toolbox_changed(self, toolbox: Toolbox, change):
        self._setDatasource(self._toolbox.datasource)

    def setDatasourceController(self,
                                datasource: DatasourceController) -> None:
        self._exchangeView('_datasource', datasource)
        # FIXME[todo]: interests?


from PyQt5.QtWidgets import QWidget, QPushButton

from datasource import (Datasource, Loop, Indexed,
                        Controller as DatasourceController)

class QDatasourceObserver(QObserver, Datasource.Observer):
    """
    Attributes
    ----------
    _datasourceController: DatasourceController
        A datasource controller used by this Button to control the
        (loop mode) of the Datasource.
    """
    _datasourceController: DatasourceController = None
    _datasource: Datasource = None
    _interests: Datasource.Change = None

    def __init__(self, datasource: DatasourceController=None,
                 interests: Datasource.Change=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._interests = interests or \
            Datasource.Change('observable_changed', 'busy_changed',
                              'state_changed')
        self.setDatasourceController(datasource)

    def setDatasourceController(self, datasource: DatasourceController) -> None:
        self._exchangeView('_datasourceController', datasource,
                           interests=self._interests)

    def datasource_changed(self, datasource: Datasource,
                           change: Datasource.Change) -> None:
        """Callback for a change of the Datasource.
        We are interested when the Datasource itself has changed and
        when its state (looping/pause) has changed.
        """
        if change.observable_changed:
            self._datasource = datasource
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
        self.toggled.connect(self.onToggled)

    def onToggled(self):
        if (self._datasourceController is not None and
            self._datasourceController.isinstance(Loop)):
            #
            checked = self.isChecked()
            self._datasourceController.loop(checked)

    def update(self) -> None:
        """Update this QLoopButton based on the state of the
        :py:class:`Datasource`.
        """
        enabled = (isinstance(self._datasource, Loop) and
                   self._datasource.prepared and
                   (self._datasource.looping or not self._datasource.busy))
        checked = enabled and self._datasourceController.looping
        self.setEnabled(enabled)
        self.setChecked(checked)


from PyQt5.QtWidgets import QWidget, QPushButton

from datasource import (Datasource, Loop, Snapshot,
                         Controller as DatasourceController)

class QSnapshotButton(QPushButton, QDatasourceObserver):
    """A Button to control a :py:class:`Datasource` of type
    :py:class:`Loop`. Pressing this button will will obtain a
    snapshot from the datasource.

    The :py:class:`QSnapshotButton` can observe a :py:class:`Datasource`
    and adapt its appearance and function based on the state of that
    datasource.

    The :py:class:`QSnapshotButton` will only be enabled if a
    :py:class:`Datasource` was registered with the
    :py:meth:`setDatasourceController` method and if this
    datasource is not busy (e.g., by looping).

    """

    def __init__(self, text: str='Snapshot', **kwargs) -> None:
        super().__init__(text, **kwargs)
        self.clicked.connect(self.onClicked)

    def onClicked(self):
        if (self._datasourceController is not None and
            self._datasourceController.isinstance(Snapshot)):
            #
            self._datasourceController.fetch_snapshot()

    def update(self) -> None:
        """Update this QLoopButton based on the state of the
        :py:class:`Datasource`.
        """
        enabled = (isinstance(self._datasource, Snapshot) and
                   self._datasource.prepared and
                   not self._datasource.busy)
        self.setEnabled(enabled)


from PyQt5.QtWidgets import QWidget, QPushButton

from datasource import (Datasource, Loop, Random,
                         Controller as DatasourceController)

class QRandomButton(QPushButton, QDatasourceObserver):
    """A Button to control a :py:class:`Datasource` of type
    :py:class:`datasource.Random`. Pressing this button will
    obtain a entry from the datasource.

    The :py:class:`QRandomButton` can observe a :py:class:`Datasource`
    and adapt its appearance and function based on the state of that
    datasource. The :py:class:`QRandomButton` will only be enabled if a
    :py:class:`Datasource` was registered with the
    :py:meth:`setDatasourceController` method and if this
    datasource is not busy (e.g., by looping).
    """

    def __init__(self, text: str='Random', **kwargs) -> None:
        super().__init__(text, **kwargs)
        self.clicked.connect(self.onClicked)

    def onClicked(self):
        if (self._datasourceController is not None and
            self._datasourceController.isinstance(Random)):
            #
            self._datasourceController.fetch(random=True)

    def update(self) -> None:
        """Update this QLoopButton based on the state of the
        :py:class:`Datasource`.
        """
        enabled = (isinstance(self._datasource, Random)
                   and self._datasource.prepared and
                   not self._datasource.busy)

        self.setEnabled(enabled)

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontMetrics, QIntValidator, QIcon
from PyQt5.QtWidgets import QHBoxLayout, QSizePolicy
from PyQt5.QtWidgets import QWidget, QPushButton, QLineEdit, QLabel


class QIndexControls(QWidget, QDatasourceObserver):
    """A group of Widgets to control an :py:class:`Indexed`
    :py:class:`Datasource`. The controls allow to select elements
    from the datasource based on their index.

    The :py:class:`QIndexControls` can observe a :py:class:`Datasource`
    and adapt their appearance and function based on the state of that
    datasource.

    The :py:class:`QIndexControls` will only be enabled if a
    :py:class:`Datasource` was registered with the
    :py:meth:`setDatasourceController` method and if this
    datasource is not busy (e.g., by fetching or looping).

    """
    # select the first entry in the Datasource (Indexed)
    _firstButton = None
    # select last entry in the Datasource (Indexed)
    _lastButton = None

    # select previous entry in the Datasource (Indexed)
    _prevButton = None
    # select next entry in the Datasource (Indexed)
    _nextButton = None

    # select specific index in the Datasource (Indexed)
    _indexField = None
    # a label for information to be shown together with the _indexField
    _indexLabel = None

    def __init__(self, **kwargs) -> None:
        interests = Datasource.Change('observable_changed', 'state_changed',
                                      'busy_changed', 'data_changed')
        super().__init__(interests=interests, **kwargs)
        self._initUI()
        self._layoutUI()

    def _initUI(self):
        """Initialize the user interface.

        """
        #
        # Navigation in indexed data source
        #
        self._firstButton = self._initButton('|<', 'go-first')
        self._prevButton = self._initButton('<<', 'go-previous')
        self._nextButton = self._initButton('>>', 'go-next')
        self._lastButton = self._initButton('>|', 'go-last')

        # _indexField: A text field to manually enter the index of
        # desired input.
        self._indexField = QLineEdit()
        self._indexField.setMaxLength(8)
        self._indexField.setAlignment(Qt.AlignRight)
        self._indexField.setSizePolicy(
            QSizePolicy.Maximum, QSizePolicy.Maximum)
        self._indexField.setMinimumWidth(
            QFontMetrics(self.font()).width('8') * 8)
        # textChanged: This signal is emitted whenever the text
        # changes.  this signal is also emitted when the text is
        # changed programmatically, for example, by calling setText().
        # We will not react to this signal, since we are also setting
        # text ourselves as a reaction to a change of the datasource.

        # textEdited: This signal is emitted whenever the text is
        # edited. This signal is not emitted when the text is changed
        # programmatically, e.g., by setText().  The text argument is
        # the new text.
        self._indexField.textEdited.connect(self._editIndex)
        
        # editingFinished: This signal is emitted when the Return or
        # Enter key is pressed or the line edit loses focus.
        self._indexField.editingFinished.connect(self._editIndex)

        self._indexLabel = QLabel()
        self._indexLabel.setMinimumWidth(
            QFontMetrics(self.font()).width('8') * 8)
        self._indexLabel.setSizePolicy(
            QSizePolicy.Maximum, QSizePolicy.Expanding)

    def _initButton(self, label: str, icon: str=None):
        button = QPushButton()
        icon = QIcon.fromTheme(icon, QIcon())
        if icon.isNull():
            button.setText(label)
        else:
            button.setIcon(icon)
        button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        button.clicked.connect(self._buttonClicked)
        return button

    def _layoutUI(self):
        """Layout the :py:class:`IndexControls`.

        """
        layout = QHBoxLayout()
        layout.addWidget(self._firstButton)
        layout.addWidget(self._prevButton)
        layout.addWidget(self._indexField)
        layout.addWidget(QLabel('of'))
        layout.addWidget(self._indexLabel)
        layout.addWidget(self._nextButton)
        layout.addWidget(self._lastButton)
        self.setLayout(layout)

    @protect
    def _buttonClicked(self, checked: bool):
        '''Callback for clicking the 'next' and 'prev' sample button.'''
        if self.sender() == self._firstButton:
            self._datasourceController.rewind()
        elif self.sender() == self._prevButton:
            self._datasourceController.rewind_one()
        elif self.sender() == self._nextButton:
            self._datasourceController.advance_one()
        elif self.sender() == self._lastButton:
            self._datasourceController.advance()
        
    @protect
    def _editIndex(self, text):
        '''Event handler for the edit field.'''
        if self._controller is None:
            return
        self._controller.edit_index(text)

    def datasource_changed(self, datasource: Datasource,
                           info: Datasource.Change) -> None:
        if info.state_changed:
            if isinstance(datasource, Indexed) and datasource.prepared:
                elements = len(datasource)
                self._indexField.setValidator(QIntValidator(0, elements))
            else:
                self._indexField.setValidator(None)
        super().datasource_changed(datasource, info)

    def update(self) -> None:
        """Update this :py:class:`IndexControls` based on the state of the
        :py:class:`Datasource`.
        """
        datasource = self._datasource
        if isinstance(datasource, Indexed) and datasource.prepared:
            enabled = not datasource.busy
            elements = len(datasource)           
            index = datasource.index
        else:
            enabled = False
            elements = '*'
            index = ''
        self._firstButton.setEnabled(enabled and index > 0)
        self._prevButton.setEnabled(enabled and index > 0)
        self._nextButton.setEnabled(enabled and index+1 < elements)
        self._lastButton.setEnabled(enabled and index+1 < elements)

        self._indexField.setEnabled(enabled)
        self._indexField.setText(str(index))
        self._indexLabel.setText(str(elements))

class QDatasourceNavigator(QWidget, QObserver, Datasource.Observer):
    
    # controls for an indexed datasource (Indexed)
    _indexControls: QIndexControls = None

    # select random entry from the Datasource (Random)
    _randomButton: QRandomButton = None

    # Snapshot button (Snapshot)
    _snapshotButton: QSnapshotButton = None

    # start/stop looping the Datasource (Loop)
    _loopButton: QLoopButton = None

    # FIXME[old]: try to work without a DatasourceControler
    _datasourceController: DatasourceController = None

    def __init__(self, datasource: DatasourceController=None, **kwargs):
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
        self.setDatasourceController(datasource)

    def _initUI(self):
        """Initialize the user interface.

        """
        self._indexControls = QIndexControls()
        self._randomButton = QRandomButton()
        self._snapshotButton = QSnapshotButton()
        self._loopButton = QLoopButton()

    def _layoutUI(self):
        layout = QHBoxLayout()
        layout.addWidget(self._indexControls)
        layout.addStretch()
        layout.addWidget(self._randomButton)
        layout.addWidget(self._snapshotButton)
        layout.addWidget(self._loopButton)
        self.setLayout(layout)

    def setDatasource(self, datasource: Datasource) -> None:
        self._indexControls.setVisible(isinstance(datasource, Indexed))
        self._randomButton.setVisible(isinstance(datasource, Random))
        self._snapshotButton.setVisible(isinstance(datasource, Snapshot))
        self._loopButton.setVisible(isinstance(datasource, Loop))

    #
    # FIXME[old]: try to work without a DatasourceControler
    #
        
    def setDatasourceController(self, datasource: DatasourceController) -> None:
        interests = Datasource.Change('observable_changed')
        self._exchangeView('_datasourceController', datasource, interests=interests)
        self._indexControls.setDatasourceController(datasource)
        self._randomButton.setDatasourceController(datasource)
        self._snapshotButton.setDatasourceController(datasource)
        self._loopButton.setDatasourceController(datasource)

    def datasource_changed(self, datasource: Datasource, info) -> None:
        """React to changes in the datasource. Changes of interest are:
        (1) change of datasource ('observable_changed'): we may have to
            adapt the controls to reflect the type of Datasource.
        (2) usability of the datasource ('state_changed', 'busy_changed'):
            we may have to disable some controls depending on the state
            (prepared, busy).
        (3) change of selected image (data_changed): we may want to
            reflect the selection in our controls.
        """
        if info.observable_changed:
            self.setDatasource(datasource)

class QInputNavigator(QWidget, QObserver, Datasource.Observer):
    """A :py:class:`QInputNavigator` displays widgets to navigate in the
    Datasource.  The actual set of widgets depends on the type of
    Datasource and will be adapated when the Datasource is changed.

    """
    _controller: DatasourceController = None
    
    #
    _infoDatasource: QLabel = None
    
    # display the state of the Datasource:
    # "none" / "unprepared" / "busy" / "ready"
    _stateLabel = None

    # prepare or unprepare the Datasource
    _prepareButton = None

    _navigator: QDatasourceNavigator = None

    def __init__(self, datasource: DatasourceController=None, **kwargs):
        '''Initialization of the QInputNavigator.

        Parameters
        ---------
        datasource: DatasourceController
            A Controller allowing to navigate in the Datasource.
        '''
        super().__init__(**kwargs)
        self._initUI()
        self._layoutUI()
        self.setDatasourceController(datasource)

    def _initUI(self):
        """Initialize the user interface.

        """
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

    def setDatasourceController(self,
                                datasource: DatasourceController) -> None:
        interests = Datasource.Change('observable_changed', 'busy_changed',
                                      'state_changed', 'data_changed')
        self._exchangeView('_controller', datasource, interests=interests)
        self._navigator.setDatasourceController(datasource)

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
