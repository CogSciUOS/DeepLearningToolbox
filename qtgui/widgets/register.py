"""Qt widgets to access a :py:class:`Register`.

"""

# standard imports
from typing import Union
import logging

# Qt imports
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeyEvent, QColor, QPalette

from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel, QRadioButton,
                             QHBoxLayout, QVBoxLayout, QListWidgetItem,
                             QScrollArea, QSizePolicy)

# toolbox imports
from dltb.base.register import Register, RegisterEntry, Registrable
from dltb.base.register import RegisterClass
from dltb.base.register import ClassRegisterEntry, InstanceRegisterEntry
from toolbox import Toolbox

# GUI imports
from ..utils import QObserver, QDebug, QPrepareButton, protect
from ..adapter import ItemAdapter, QAdaptedListWidget, QAdaptedComboBox

# logging
LOG = logging.getLogger(__name__)


class RegisterAdapter(QObserver, ItemAdapter, qobservables={
        Register: {'entry_added', 'entry_removed', 'entry_changed'}
        }, itemToText=lambda entry: entry.key):
    # pylint: disable=abstract-method
    """A list of register entries.  Register entries are objects
    registered in a :py:class:`Register`, implementing the
    :py:class:`Registrable` interface. Those objects have
    :py:attr:`key` property providing a unique string allowing to
    refer to the object in the register.

    The :py:class:`RegisterAdapter` can observe a
    :py:class:`Register` and react to changes by adding
    or removing entries.
    """

    def __init__(self, register: Register = None, **kwargs) -> None:
        """Initialization of the :py:class:`RegisterAdapter`.

        Parameters
        ----------
        """
        super().__init__(**kwargs)
        self._onlyInitialized = False
        self.setRegister(register)

    def setRegister(self, register: Register) -> None:
        """Set a new :py:class:`Register` for this
        :py:class:`RegisterAdapter`. The entries of this list
        will be updated from the register.

        Arguments
        ---------
        register: Register
            The :py:class:`Register` from which the list
            will be updated. If `None` the list will be cleared.
        """
        self.setFromIterable(iter(()) if register is None else register)

    def _updateFromRegister(self) -> None:
        """Update this :py:class:`RegisterAdapter` to reflect the
        the current state of the register, taken the display flag
        `onlyInitialized` into account.
        """
        self.updateFromIterable(iter(()) if self._register is None else
                                self._register)

    def register_changed(self, register: Register,
                         change: Register.Change, key: str = None) -> None:
        # pylint: disable=invalid-name
        """Called upon a change in the :py:class:`Register`.

        Arguments
        ---------
        register: Register
            The :py:class:`Register` that was changed.
        change: Register.Change
        key: str
            The key that was changed.
        """
        LOG.info("%s.register_changed: %s [%r], key=%s",
                 type(self).__name__, type(register).__name__, change, key)

        if key is None:
            # FIXME[concept]: key may be None, if the notification is
            # received uppon showing the widget after it was hidden.
            # This means that we can in fact not rely on key having a
            # meaningful value in the GUI - if we want to change this,
            # we would have to make the notification upon show more
            # sophisticated!
            self._updateFromRegister()
            return

        if change.entry_added:
            self._addItem(register[key])
        if change.entry_changed:
            self._formatItem(register[key])
        if change.entry_removed:
            self._removeText(key)

    #
    # Filter methods for class/instance registers
    #

    def onlyInitialized(self) -> bool:
        """A flag indicating if only initialized entries (`True`)
        or all entry (`False`) are listed.
        """
        return self._onlyInitialized

    def setOnlyInitialized(self, onlyInitialized: bool = True) -> None:
        """Specify if only initialized entries (`True`) or all entry (`False`)
        shall be listed.
        """
        if onlyInitialized != self._onlyInitialized:
            self._onlyInitialized = onlyInitialized
            self._updateFromRegister()

    #
    # FIXME[old]: old methods
    #

    @protect
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Process key events. The :py:class:`RegisterAdapter` supports
        the following keys:

        I: toggle the `onlyInitialized` flag
        U: update display of the entries of this :py:class:`RegisterAdapter`

        Note: in a QComboBox this event is only received if the combobox
        is closed (not while currently selecting an entry).
        """
        key = event.key()
        LOG.debug("RegisterAdapter.keyPressEvent: key=%d", key)
        if key == Qt.Key_U:  # update
            LOG.info("Updateting from register: %s",
                     self._register and type(self._register).__name__)
            self._updateFromRegister()
            LOG.debug("Updated RegisterAdapter: %s",
                      self._register and list(self._register.keys()))
        elif key == Qt.Key_I:  # toggle onlyInitialized
            self.setOnlyInitialized(not self.onlyInitialized())
        else:
            super().keyPressEvent(event)

    def currentEntry(self) -> Registrable:
        """Get the currently selected entry.
        This may be `None` if no entry is selected.
        """
        return self._currentItem()

    def setCurrentEntry(self, entry: Registrable) -> None:
        """Select the given entry in this :py:class:`RegisterAdapter`.

        Arguments
        ---------
        entry: Registrable
            The entry to become the currently selected entry
            in this list. `None` will deselect the current element.

        Raises
        ------
        ValueError:
            The given entry is not an element of this
            :py:class:`RegisterAdapter`.
        """
        self._setCurrentItem(entry)

    def debug(self) -> None:
        """Output debug information.
        """
        super().debug()
        print(f"debug: RegisterAdapter[{type(self).__name__}]:")
        # print(f"debug:   * register: {self.register()}")
        print(f"debug:   * register: {self._register}")
        print(f"debug:   * onlyInitialized: {self.onlyInitialized()}")
        if self._register is not None:
            print(f"debug:   * register entries:")
            for entry in self._register:
                print(f"debug:     {'+' if not entry.initialized else '-'} "
                      f"{entry.key} [{repr(entry)}]")


class ToolboxAdapter(RegisterAdapter, qobservables={Toolbox: set()}):

    def __init__(self, toolbox: Toolbox = None, **kwargs) -> None:
        """
        Arguments
        ---------
        toolbox: Toolbox
        """
        super().__init__(**kwargs)
        self.setToolbox(toolbox)

    def setToolbox(self, toolbox: Toolbox) -> None:
        """Set a :py:class:`Toolbox`. If a toolbox is set, the list
        will be filled from that toolbox, no longer from the register.
        """
        if toolbox is None:
            self.observe(self._register)
            self._updateFromRegister()
        else:
            self.unobserve(self._register)
            self.updateFromToolbox()

    def updateFromToolbox(self) -> None:
        """Update the list from the :py:class:`Toolbox`.
        """
        raise NotImplementedError("A ToolboxAdapter should implement "
                                  "'updateFromToolbox'")

    def debug(self) -> None:
        """Output debug information for this :py:class:`ToolboxAdapter`.
        """
        super_debug = getattr(super(), 'debug')
        if super_debug is not None:
            super_debug()
        print(f"debug: ToolboxAdapter[{type(self).__name__}]: "
              f"Toolbox={self.toolbox()}")
        toolbox = self.toolbox()
        if toolbox is not None:
            for index, datasource in enumerate(toolbox.datasources):
                print("debug:   "
                      f"{'**' if datasource is toolbox.datasource else '  '}"
                      f" ({index}) {datasource} [{type(datasource)}]")


class QRegisterListWidget(QAdaptedListWidget, RegisterAdapter):
    """A :py:class:`QListWidget` for selecting entries from a
    :py:class:`Register`.
    """


class QRegisterComboBox(QAdaptedComboBox, RegisterAdapter):
    """A :py:class:`QComboBox` for selecting entries from a
    :py:class:`Register`.
    """


# #############################################################################

#
# Class register
#


class QRegisterClassView(QWidget):
    """A widget for viewing a :py:class:`RegisterClass`.

    This is essentially just a :py:class:`QRegisterListWidget` that
    can be used in two modes: either list the `class_register` or
    the `instance_register` of the :py:class:`RegisterClass`.
    The view can be changed between these two options by setting
    the mode (:py:meth:`setMode`) to either `class` or `instance`.
    There are also some radio buttons included that allow to
    select the mode.
    """

    instanceSelected: pyqtSignal = pyqtSignal(object)
    classSelected: pyqtSignal = pyqtSignal(type)

    colorInitialized: QColor = QColor(Qt.white).lighter()
    colorInitializable: QColor = QColor(Qt.blue).lighter()
    colorUninitializable: QColor = QColor(Qt.red).lighter()

    def __init__(self, registerClass: RegisterClass = None, **kwargs) -> None:
        """
        """
        super().__init__(**kwargs)
        self._registerClass = None
        self._mode = None  # 'class' or 'instance'
        self._initUI()
        self._initLayout()
        self._listWidget.setListWidgetItemFormater(self._formatListWidgetItem)
        self.setRegisterClass(registerClass)

    def _initUI(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        """Initialize the user interface.
        """
        self._listWidget = QRegisterListWidget()
        self._listWidget.currentItemChanged.connect(self._onCurrentItemChanged)

        self._instanceButton = QRadioButton('Instances')
        self._instanceButton.clicked.connect(self._onRadioButtonClicked)
        self._classButton = QRadioButton('classes')
        self._classButton.clicked.connect(self._onRadioButtonClicked)

    def _initLayout(self) -> None:
        layout = QVBoxLayout()
        layout.addWidget(self._listWidget)
        row = QHBoxLayout()
        row.addWidget(self._classButton)
        row.addWidget(self._instanceButton)
        layout.addLayout(row)
        self.setLayout(layout)

    def registerClass(self) -> RegisterClass:
        """The :py:class:`RegisterClass` currently viewed.
        """
        return self._registerClass

    def setRegisterClass(self, registerClass: RegisterClass) -> None:
        """Set the :py:class:`RegisterClass` to be viewed.
        """
        self._registerClass = registerClass
        self.update()

    def mode(self) -> str:
        """The mode of this :py:class:`QRegisterClassView` (either
        `class` or `instance`).
        """
        return self._mode

    def setMode(self, mode: str) -> None:
        """The mode of this :py:class:`QRegisterClassView` (either
        `class` or `instance`).
        """
        if mode == self._mode:
            return  # nothing changed

        self._mode = mode
        self.update()
        if self._mode is None:
            raise ValueError("Invalide mode for QRegisterClassView: '{mode}'")

    def setClass(self, cls: type) -> None:
        """Set the currently selected class entry. This will switch
        the view into `'class'` mode.
        """
        self.setMode('class')
        registerEntry = cls and self._registerClass.class_register[cls]
        self._listWidget.setCurrentEntry(registerEntry)

    def setInstance(self, instance: object) -> None:
        """Set the currently selected instance entry. This will switch
        the view into `'instance'` mode.
        """
        self.setMode('instance')
        registerEntry = (instance and
                         self._registerClass.instance_register[instance.key])
        self._listWidget.setCurrentEntry(registerEntry)

    def update(self) -> None:
        """Update the display elements.
        """
        mode = self._mode
        registerClass = self._registerClass
        register = None

        if mode == 'class':
            self._classButton.setChecked(True)
            if registerClass is not None:
                register = registerClass.class_register
        elif mode == 'instance':
            self._instanceButton.setChecked(True)
            if registerClass is not None:
                register = registerClass.instance_register
        else:
            self._mode = None
            self._classButton.setChecked(False)
            self._instanceButton.setChecked(False)

        self._listWidget.setRegister(register)
        super().update()

    @protect
    def _onCurrentItemChanged(self, current: QListWidgetItem,
                              _previous: QListWidgetItem) -> None:
        """React to the selection of an item in this list.
        """
        entry = None if current is None else current.data(Qt.UserRole)
        if isinstance(entry, InstanceRegisterEntry):
            self.instanceSelected.emit(entry)
        elif isinstance(entry, ClassRegisterEntry):
            self.classSelected.emit(entry)  # FIXME[bug]: TypeError
            # QRegisterClassView.classSelected[type].emit():
            # argument 1 has unexpected type 'ClassRegisterEntry'

    @protect
    def _onRadioButtonClicked(self, _checked: bool) -> None:
        """React to a mode selection by the radio buttons.
        """
        self.setMode('class' if self._classButton.isChecked() else 'instance')

    def _formatListWidgetItem(self, item: QListWidgetItem) -> None:
        """Format a :py:class:`QListWidgetItem`.

        Arguments
        ---------
        item: QListWidgetItem
            The :py:class:`QListWidgetItem` to format. It is guaranteed
            that the associated data if of type
            :py:class:`ClassRegisterEntry`.
        """
        entry = item.data(Qt.UserRole)
        if entry.initialized:
            color = self.colorInitialized
        elif entry.initializable:
            color = self.colorInitializable
        else:
            color = self.colorUninitializable
        if self._mode == 'class':
            item.setBackground(color)
        elif self._mode == 'instance':
            item.setBackground(color)


#
# Controller
#


class QRegisterClassEntryController(QWidget, QObserver, qobservables={
        Register: {'entry_changed'}}, qattributes={
            RegisterEntry: True}):
    """A controller for entries of a :py:class:`RegisterClass`. This
    may be subclassed to either control :py:class:`ClassRegisterEntry`
    entries or :py:class:`InstanceRegisterEntry`. There are two
    subclasses, :py:class:`QClassRegisterEntryController` and
    :py:class:QInstanceRegisterEntryController` to control these
    specific types of entries.

    The class register entries themselves are not observable, but
    observing the corresponding :py:class:`Register` (either
    :py:attr:`RegisterClass.class_register` or
    :py:attr:`RegisterClass.instance_register`) allows us to get
    informed when the status of the class has changed via the
    `entry_changed` notification.
    """

    def __init_subclass__(cls, register: RegisterClass = None,
                          **kwargs) -> None:
        # pylint: disable=arguments-differ
        """
        """
        super().__init_subclass__(**kwargs)
        if register is not None:
            cls.register = register

    def __init__(self, **kwargs) -> None:
        """Initialization of the :py:class:`QRegisterClassEntryController`.
        """
        super().__init__(**kwargs)
        self._name = None  # the class name
        self._description = None  # FIXME[question]: what is this?
        self._initUI()
        self._layoutUI()

    def _initUI(self) -> None:
        # pylint: disable=attribute-defined-outside-init
        """Initialize the user interface for this
        :py:class:`QRegisterClassEntryController`.
        """
        if not hasattr(self, '_button'):
            self._button = QPushButton()
            self._button.setEnabled(False)
        self._button.clicked.connect(self._onButtonClicked)
        self._keyLabel = QLabel()
        self._stateLabel = QLabel()

        self._errorLabel = QLabel()
        self._errorLabel.setWordWrap(True)
        # FIXME[todo]: find the best way (portable, theme aware, ...)
        # to set QLabel style and apply this globally (to all QLabels
        # in the user interface)
        # self._errorLabel.setStyleSheet("QLabel { color : red; }")
        palette = self._errorLabel.palette()
        palette.setColor(self._errorLabel.foregroundRole(), Qt.red)
        self._errorLabel.setPalette(palette)

        self._descriptionLabel = QLabel()
        self._descriptionLabel.setWordWrap(True)
        self._descriptionLabel.setBackgroundRole(QPalette.Base)
        # self._descriptionLabel.setSizePolicy(QSizePolicy.Ignored,
        #                                      QSizePolicy.Ignored)
        # self._descriptionLabel.setScaledContents(True)

        self._scrollArea = QScrollArea()
        self._scrollArea.setBackgroundRole(QPalette.Dark)
        self._scrollArea.setWidget(self._descriptionLabel)
        self._scrollArea.setWidgetResizable(True)

    def _layoutUI(self) -> None:
        """Layout the user interface for this
        :py:class:`QRegisterClassEntryController`.
        """
        layout = QVBoxLayout()

        rows = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(QLabel('Key: '))
        row.addWidget(self._keyLabel)
        row.addStretch()
        rows.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel('State: '))
        row.addWidget(self._stateLabel)
        row.addStretch()
        rows.addLayout(row)

        row = QHBoxLayout()
        row.addLayout(rows)
        row.addStretch()
        row.addWidget(self._button)

        layout.addLayout(row)
        layout.addWidget(self._errorLabel)
        layout.addStretch()
        # layout.addWidget(self._descriptionLabel)
        layout.addWidget(self._scrollArea)
        self.setLayout(layout)

    def setRegisterEntry(self, entry: Union[RegisterEntry, str]) -> None:
        """Set a new :py:class:`ClassRegisterEntry` to control.

        Arguments
        ---------
        entry: Union[RegisterEntry, str]
            Either a subclass of the register class or the
            (fully qualified) name of such a class.
        """
        if isinstance(entry, str):
            # An entry key is provided
            entry = self._register[entry]
            self._registerEntry = entry
        # FIXME[todo]
        # elif not isinstance(entry, ClassRegisterEntry) and entry is not None:
        #    raise TypeError("Argument class has invalid type: "
        #                    f"{type(entry)}")

        self.update()

    def register_changed(self, register: Register,
                         change: Register.Change, key: str = None) -> None:
        # pylint: disable=invalid-name
        """Called upon a change in the :py:class:`ClassRegister`.

        Arguments
        ---------
        register: Register
            The :py:class:`Register` that was changed.
        change: Register.Change
        key: str
            The key that was changed.
        """
        LOG.info("%s.register_changed: %s [%r], key=%s",
                 type(self).__name__, type(register).__name__, change, key)
        self.update()

    @protect
    def _onButtonClicked(self, checked: bool) -> None:
        """The button has been clicked.
        """
        LOG.info("%s.buttonClicked(checked=%r): text=%s, key=%s",
                 type(self).__name__, checked, self._button.text(),
                 "None" if self._registerEntry is None else
                 self._registerEntry.key)
        if self._registerEntry is None:
            return  # nothing to do (should not happen) ...

        if not self._registerEntry.initialized:
            # initialize the class object represented by the current entry
            self._registerEntry.initialize()


class QClassRegisterEntryController(QRegisterClassEntryController):
    """Controller for a :py:class:`ClassRegisterEntry`. Such
    an entry represents an (unititialized or initialized) class
    object. The controller provides a button to initialize
    the class object (import the module that defines the class).
    If initialized, some additional information on the class
    is presented.
    """

    def __init__(self, registerClass: RegisterClass = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if registerClass is not None:
            self.setRegister(registerClass.class_register)

    def _initUI(self) -> None:
        """Initialize the user interface for this
        :py:class:`QClassRegisterEntryController`.
        """
        super()._initUI()
        self._button.setText("Initialize")

    def update(self) -> None:
        """Update the display of this
        :py:class:`QClassRegisterEntryController`.
        This may adapt the state of controls (enabled/disabled)
        and information displayed in labels and other elements,
        according to the current :py:class:`ClassRegisterEntry`.
        """
        entry = self._registerEntry

        if entry is None:
            self._descriptionLabel.setText("")
            self._stateLabel.setText("")
            self._keyLabel.setText("")
            self._button.setEnabled(False)
        else:
            self._keyLabel.setText(f"[{entry.key}]")
            if entry.initialized:
                self._stateLabel.setText("initialized")
                self._button.setEnabled(False)
                self._descriptionLabel.setText(entry.cls.__doc__)
            else:
                self._stateLabel.setText("uninitialized")
                self._button.setEnabled(True)
                self._descriptionLabel.setText(entry.module_name + "." +
                                               entry.class_name)
        super().update()


class QInitializeButton(QPrepareButton, qobservables={
        InstanceRegisterEntry: {'busy_changed', 'state_changed'}}):
    """An initialize button allows to initialize the class or
    instance represented by a :py:class:`ClassRegisterEntry`
    or :py:class:`InstanceRegisterEntry`, respectively.
    """

    def __init__(self, initialize: str = "Initialize", **kwargs) -> None:
        """Initialize the :py:class:`QInitializeButton`.
        """
        # _initialize: bool
        #     A flag indicating if this button is in initialize mode (True)
        #     or in prepare mode (False).
        self._initialize = False

        # _initializeText: str
        #     The label to be displayed on the button if it is in
        #     initialize mode.
        self._initializeText = initialize

        # _prepareText: str
        #     The label to be displayed on the button if it is in
        #     prepare mode (that is not in initalize mode).
        self._prepareText = "Prepare"
        super().__init__(**kwargs)

    def setInitialize(self) -> None:
        """Set this :py:class:`QInitializeButton` to be in
        initialization mode. In that mode, pressing the button
        will trigger initialization.
        """
        self._initialize = True
        self.setPreparable(None)
        self.updateState()

    def updateState(self) -> None:
        """Update this :py:class:`QInitializeButton` based on the
        state of the :py:class:`Preparable`.
        """
        if self._preparable is None and self._initialize:
            # we are in initialize mode
            entry = self._instanceRegisterEntry
            if entry is None:
                self.setEnabled(False)
                self.setChecked(False)
                self.setText("no object")
            else:
                if entry.busy:
                    self.setEnabled(False)
                    self.setChecked(True)
                    if entry.initialized:
                        self.setText("Uninitializing")
                    else:
                        self.setText("Initializing")
                else:
                    self.setEnabled(entry.initializable)
                    self.setChecked(entry.initialized)
                    self.setText("Initialize")
        else:
            self._initialize = False
            self.setText(self._prepareText)
            super().updateState()

    def entry_changed(self, entry: InstanceRegisterEntry,
                      change: InstanceRegisterEntry.Change) -> None:
        """React to a change of the observed
        :py:class:`InstanceRegisterEntry`. Such a change means that
        the entry was initialized, that is that an object was created.
        """
        if not entry.busy:
            self.setPreparable(entry.obj)
        else:
            self.updateState()


class QInstanceRegisterEntryController(QRegisterClassEntryController,
        qobservables={InstanceRegisterEntry: {'state_changed'}}):
    """A controller for an :py:class:`InstanceRegisterEntry`. This
    controller allows to instantiate and initialize a registered
    instance of a class.

    """

    def __init__(self, registerClass: RegisterClass = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if registerClass is not None:
            self.setRegister(registerClass.instance_register)

    def _initUI(self) -> None:
        """Initialize the user interface for this
        :py:class:`QInstanceRegisterEntryController`.
        """
        self._button = QInitializeButton()
        self.addAttributePropagation(InstanceRegisterEntry, self._button)
        super()._initUI()

    def update(self) -> None:
        """Update the display of this
        :py:class:`QInstanceRegisterEntryController`.
        This may adapt the state of controls (enabled/disabled)
        and information displayed in labels and other elements,
        according to the current :py:class:`InstanceRegisterEntry`.
        """
        entry = self._registerEntry

        if entry is None:
            self._descriptionLabel.setText("")
            self._stateLabel.setText("")
            self._keyLabel.setText("")
            self._button.setPreparable(None)
        else:
            self._keyLabel.setText(f"[{entry.key}]")
            if entry.initialized:
                self._stateLabel.setText("initialized")
                self._button.setPreparable(entry.obj)
                self._descriptionLabel.setText(entry.cls.__doc__)
            else:
                self._stateLabel.setText("uninitialized")
                self._button.setInitialize()
                self._descriptionLabel.setText(str(entry))
        super().update()

    @protect
    def _onButtonClicked(self, checked: bool) -> None:
        """The button has been clicked.
        """
        LOG.info("%s.buttonClicked(checked=%r): text=%s, key=%s",
                 type(self).__name__, checked, self._button.text(),
                 "None" if self._registerEntry is None else
                 self._registerEntry.key)
        if self._registerEntry is None:
            return  # nothing to do (should not happen) ...

        if not self._registerEntry.initialized:
            # initialize the class object represented by the current entry
            self._button.setEnabled(False)
            self._button.setText("Initializing")
            print("QInstanceRegisterEntryController: Initializing ...")
            self._registerEntry.initialize()

    def setRegisterEntry(self, entry: Union[RegisterEntry, str]) -> None:
        """Set a new :py:class:`ClassRegisterEntry` to control.

        Arguments
        ---------
        entry: Union[RegisterEntry, str]
            Either a subclass of the register class or the
            (fully qualified) name of such a class.
        """
        super().setRegisterEntry(entry)
        # FIXME[hack]: set the observable InstanceRegisterEntry ...
        self.setInstanceRegisterEntry(self._registerEntry)

    #
    # Observers
    #

    def entry_changed(self, entry: InstanceRegisterEntry,
                      change: InstanceRegisterEntry.Change) -> None:
        self.update()


class QInstanceRegisterComboBox(QRegisterComboBox):
    """A :py:class:`QComboBox` for selecting entries from a
    :py:class:`Register`.
    """

    def _formatItemAt(self, index: int) -> None:
        """Format the item at the given index to reflect
        the state of the underlying item.

        This method may be extended by subclasses.
        """
        super()._formatItemAt(index)

        # disable item if tool is not instantiable
        instance = self._getItemAt(index)  # instance or InstanceRegisterEntry
        if (isinstance(instance, InstanceRegisterEntry) and
                not instance.initializable):
            item = self.model().item(index)  # QtGui.QStandardItem
            item.setFlags(item.flags() & ~ Qt.ItemIsEnabled)
