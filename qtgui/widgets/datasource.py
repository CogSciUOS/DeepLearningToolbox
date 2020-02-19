
from toolbox import Toolbox, View as ToolboxView
from datasources import Datasource, View as DatasourceView

from ..utils import QObserver, protect
from .helper import QToolboxViewList

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QListWidgetItem


class QDatasourceList(QToolboxViewList, QObserver, Datasource.Observer):
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

    @protect
    def onItemClicked(self, item: QListWidgetItem):
        self._datasource(item.data(Qt.UserRole))


    class ViewObserver(QToolboxViewList.ViewObserver, Datasource.Observer):
        interests = Datasource.Change('state_changed', 'metadata_changed')
        
        def data(self, toolbox: ToolboxView):
            return toolbox.datasources

        def formatItem(self, item:QListWidgetItem) -> None:
            if item is not None:
                datasource = item.data(Qt.UserRole)
                item.setForeground(Qt.green if datasource.prepared
                                   else Qt.black)

        def datasource_changed(self, datasource: Datasource,
                               change: Datasource.Change) -> None:
            if change.state_changed or change.metadata_changed:
                self._listWidget._updateItem(datasource)



from datasources import (Datasource, DataArray, DataFile, DataDirectory,
                         DataWebcam, DataVideo, Predefined,
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
    """The QDatasourceSelectionBox provides a controls to select a data
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
    
    def __init__(self, toolbox: ToolboxController=None, parent=None):
        """Initialization of the :py:class:`QDatasourceSelectionBox`.

        Parameters
        ---------
        parent: QWidget
            The parent argument is sent to the QWidget constructor.
        """
        super().__init__(parent)
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
        dataset_names = Predefined.get_data_source_ids()
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
        if isinstance(datasource, Predefined):
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
            datasource = Predefined.get_data_source(name)

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
                # if datasources, if it is not already in there.
                self._toolbox.datasource_controller(datasource)

    @protect
    def _predefinedSelectionChange(self,i):
        if self._radioButtons['Name'].isChecked():
            self._datasetDropdown.setVisible(True)
            name = self._datasetDropdown.currentText()
            datasource = Predefined.get_data_source(name)

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

from datasources import (Datasource, Loop,
                         Controller as DatasourceController)

class QLoopButton(QPushButton, QObserver, Datasource.Observer):
    """A Button to control a :py:class:`Datasource` of type
    :py:class:`Loop`. Such datasources can be in a loop mode, meaning
    that they continously produce new data (e.g., webcam, movies,
    etc.).

    The :py:class:`QLoopButton` can observe a :py:class:`Datasource`
    and adapt its appearance and function based on the state of the
    datasource.

    Attributes
    ----------
    _datasourceController: DatasourceController
        A datasource controller used by this Button to control the
        (loop mode) of the Datasource.
    """

    _datasourceController: DatasourceController = None

    def __init__(self, text: str='Loop', datasource: DatasourceController=None,
                 parent: QWidget=None) -> None:
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setDatasourceController(datasource)
        self.toggled.connect(self.onToggled)

    def setDatasourceController(self, datasource: DatasourceController) -> None:
        interests = Datasource.Change('observable_changed', 'state_changed')
        self._exchangeView('_datasourceController', datasource,
                           interests=interests)

    def datasource_changed(self, datasource: Datasource,
                           change: Datasource.Change) -> None:
        """Callback for a change of the DataSource.
        We are interested when the Datasource itself has changed and
        when its state (looping/pause) has changed.
        """
        self.update()

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
        enabled = (self._datasourceController is not None and
                   self._datasourceController.isinstance(Loop) and
                   self._datasourceController.prepared)
        checked = enabled and self._datasourceController.looping
        self.setEnabled(enabled)
        self.setChecked(checked)


from PyQt5.QtWidgets import QWidget, QPushButton

from datasources import (Datasource, Loop, Snapshot,
                         Controller as DatasourceController)

class QSnapshotButton(QPushButton, QObserver, Datasource.Observer):
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


    Attributes
    ----------
    _datasourceController: DatasourceController
        A datasource controller used by this Button to control the
        (loop mode) of the Datasource.
    """
    _datasourceController: DatasourceController = None

    def __init__(self, text: str='Snapshot',
                 datasource: DatasourceController=None,
                 parent: QWidget=None) -> None:
        super().__init__(text, parent)
        self.setDatasourceController(datasource)
        self.clicked.connect(self.onClicked)

    def setDatasourceController(self, datasource: DatasourceController) -> None:
        interests = Datasource.Change('observable_changed', 'state_changed')
        self._exchangeView('_datasourceController', datasource,
                           interests=interests)

    def datasource_changed(self, datasource: Datasource,
                           change: Datasource.Change) -> None:
        """Callback for a change of the DataSource.
        We are interested when the Datasource itself has changed and
        when its state (looping/pause) has changed.
        """
        self.update()

    def onClicked(self):
        if (self._datasourceController is not None and
            self._datasourceController.isinstance(Snapshot)):
            #
            self._datasourceController.snapshot()

    def update(self) -> None:
        """Update this QLoopButton based on the state of the
        :py:class:`Datasource`.
        """
        enabled = (self._datasourceController is not None and
                   self._datasourceController.isinstance(Snapshot) and
                   self._datasourceController.prepared)

        if enabled and self._datasourceController.isinstance(Loop):
            enabled = not self._datasourceController.looping

        self.setEnabled(enabled)


from PyQt5.QtWidgets import QWidget, QPushButton

from datasources import (Datasource, Loop, Random,
                         Controller as DatasourceController)

class QRandomButton(QPushButton, QObserver, Datasource.Observer):
    """A Button to control a :py:class:`Datasource` of type
    :py:class:`datasources.Random`. Pressing this button will
    obtain a entry from the datasource.

    The :py:class:`QRandomButton` can observe a :py:class:`Datasource`
    and adapt its appearance and function based on the state of that
    datasource. The :py:class:`QRandomButton` will only be enabled if a
    :py:class:`Datasource` was registered with the
    :py:meth:`setDatasourceController` method and if this
    datasource is not busy (e.g., by looping).


    Attributes
    ----------
    _datasourceController: DatasourceController
        A datasource controller used by this Button to control the
        (loop mode) of the Datasource.
    """
    _datasourceController: DatasourceController = None

    def __init__(self, text: str='Snapshot',
                 datasource: DatasourceController=None,
                 parent: QWidget=None) -> None:
        super().__init__(text, parent)
        self.setDatasourceController(datasource)
        self.clicked.connect(self.onClicked)

    def setDatasourceController(self, datasource: DatasourceController) -> None:
        interests = Datasource.Change('observable_changed', 'state_changed')
        self._exchangeView('_datasourceController', datasource,
                           interests=interests)

    def datasource_changed(self, datasource: Datasource,
                           change: Datasource.Change) -> None:
        """Callback for a change of the DataSource.
        We are interested when the Datasource itself has changed and
        when its state (looping/pause) has changed.
        """
        self.update()

    def onClicked(self):
        if (self._datasourceController is not None and
            self._datasourceController.isinstance(Random)):
            #
            self._datasourceController.fetch(random=True)

    def update(self) -> None:
        """Update this QLoopButton based on the state of the
        :py:class:`Datasource`.
        """
        enabled = (self._datasourceController is not None and
                   self._datasourceController.isinstance(Random) and
                   self._datasourceController.prepared)

        if enabled and self._datasourceController.isinstance(Loop):
            enabled = not self._datasourceController.looping

        self.setEnabled(enabled)
