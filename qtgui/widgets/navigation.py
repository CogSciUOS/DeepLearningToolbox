"""Qt based navigation widgets.
"""

# Qt imports
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFontMetrics, QIntValidator, QIcon
from PyQt5.QtWidgets import QHBoxLayout, QSizePolicy
from PyQt5.QtWidgets import QWidget, QPushButton, QLineEdit, QLabel

# GUI imports
from ..utils import protect


class QIndexControls(QWidget):
    """A group of Widgets to control an index.
    The controls allow to navigate through the index.

    _firstButton: QPushButton
        select the first entry
    
    _lastButton: QPushButton
        select last entry

    _prevButton: QPushButton
        select previous entry
    
    _nextButton: QPushButton
        select next entry

    _indexField = None
        select specific index

    _indexLabel = None
        a label for information to be shown together with the _indexField

    Signals
    -------
    indexChanged:
        Emitted when the index was changed by user action. The signal will
        not be emitted, when the index was changed programmatically.
    """

    indexChanged = pyqtSignal(int)

    def __init__(self, **kwargs) -> None:
        self._index = 0
        self._elements = -1
        super().__init__(**kwargs)
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
        # changes.  This signal is also emitted when the text is
        # changed programmatically, for example, by calling setText().
        # We will not react to this signal, since we may also set the
        # text ourselves.

        # textEdited: This signal is emitted whenever the text is
        # edited. This signal is not emitted when the text is changed
        # programmatically, e.g., by setText().  The text argument is
        # the new text.
        self._indexField.textEdited.connect(self.onIndexEdited)
        
        # editingFinished: This signal is emitted when the Return or
        # Enter key is pressed or the line edit loses focus.
        self._indexField.editingFinished.connect(self.onIndexEditingFinished)

        self._indexLabel = QLabel()
        self._indexLabel.setMinimumWidth(
            QFontMetrics(self.font()).width('8') * 8)
        self._indexLabel.setSizePolicy(
            QSizePolicy.Maximum, QSizePolicy.Expanding)

        self.update()

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
        """Callback for clicking the 'next' and 'prev' sample button."""
        if self.sender() == self._firstButton:
            self.setIndex(0, True)
        elif self.sender() == self._prevButton:
            self.setIndex(self._index - 1, True)
        elif self.sender() == self._nextButton:
            self.setIndex(self._index + 1, True)
        elif self.sender() == self._lastButton:
            self.setIndex(self._elements - 1, True)
        
    @protect
    def onIndexEdited(self, text: str) -> None:
        """React to a textEdited signal.
        This signal is emitted whenever the text is edited (but not
        when the text is changed programmatically).
        """
        self._editIndex(text)

    @protect
    def onIndexEditingFinished(self) -> None:
        """React to the `EditingFinished` signal of the line editor.  This
        signal is emitted when the Return or Enter key is pressed or
        the line edit loses focus.

        """
        self._editIndex(self._indexField.text())
        # FIXME[hack]: For some reason, the line editor is not closed
        # automatically after emiting the `editingFinished` signal in
        # reaction to hitting the enter or return key (it still has
        # the focus, shows a cursor and accepts input).  By disabling
        # the line editor, we can force closing it.
        self._indexField.setEnabled(False)
        self._indexField.setEnabled(True)

    def _editIndex(self, text):
        self.setIndex(int(text or 0), True)

    def index(self) -> int:
        return self._index

    def setIndex(self, index: int, emit: bool = False) -> None:
        if index >= self._elements:
            index = self._elements - 1
        if index < 0:
            index = 0
        if index != self._index:
            self._index = index
            if emit:
                self.indexChanged.emit(index)
            self.update()

    def elements(self) -> int:
        return self._elements

    def setElements(self, elements: int) -> None:
        if elements != self._elements:
            self._elements = elements
            if elements > 0:
                self._indexField.setValidator(QIntValidator(0, elements))
            else:
                self._indexField.setValidator(None)

    def update(self, enabled: bool = True) -> None:
        """Update this :py:class:`IndexControls`.
        """
        have_elements =  self._elements > 0
        enabled = enabled and have_elements
        self._firstButton.setEnabled(enabled and self._index > 0)
        self._prevButton.setEnabled(enabled and self._index > 0)
        self._nextButton.setEnabled(enabled and self._index+1 < self._elements)
        self._lastButton.setEnabled(enabled and self._index+1 < self._elements)

        # The value of `enabled` may change quickly (e.g. if an underlying
        # object becomes busy due to an index change). While reflecting
        # this change of state seems ok for the buttons, it is harmful for
        # the QLineEdit, as disabling leads to closing the editor. 
        self._indexField.setEnabled(have_elements)
        if int(self._indexField.text() or '0') != self._index:
            # Avoid setting identical text, as this may change the
            # cursor position
            self._indexField.setText(str(self._index) if have_elements else '')
        self._indexLabel.setText(str(self._elements) if have_elements else '*')
