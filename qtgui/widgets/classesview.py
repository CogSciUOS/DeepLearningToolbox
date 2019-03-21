from tools.activation import (Engine as ActivationEngine,
                              View as ActivationView)

from ..utils import QObserver

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QLabel, QCheckBox,
                             QGridLayout, QVBoxLayout)


class QClassesView(QWidget, QObserver, ActivationEngine.Observer):
    """Visualization of classification results.

    Fields
    ------
    _top_n: int
    _classes: List[QLabel]
    _scores: List[QLabel]
    _target: QLabel
    """

    _activation: ActivationView = None
    
    def __init__(self, parent: QWidget=None, top_n=5):
        """Initialization of the QActivationView.

        Parameters
        -----------
        parent  :   QtWidget
                    Parent widget (passed to super)
        """
        super().__init__(parent)
        self._top_n = top_n
        self._classes = []
        self._scores = []
        self._initUI()

    def _initUI(self):
        layout = QGridLayout()
        layout.addWidget(QLabel(f"<b>Class</b>", self), 0,0)
        layout.addWidget(QLabel(f"<b>Score</b>", self), 0,1)
        for i in range(self._top_n):
            self._classes.append(QLabel(f"None", self))
            self._classes[i].setMaximumWidth(self._classes[i].width())
            self._classes[i].setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(self._classes[i], i+1, 0)
            self._scores.append(QLabel("0", self))
            layout.addWidget(self._scores[i], i+1, 1)
        layout.addWidget(QLabel("<b>Correct</b>", self), self._top_n+1, 0)
        self._target = QLabel(f"?", self)
        self._target.setMaximumWidth(self._target.width())
        layout.addWidget(self._target, self._top_n+1, 1)
        
        layout2 = QVBoxLayout()
        self._checkbox_active = QCheckBox("Classification")
        # FIXME[stub]: this checkbox currently does not do anything
        layout2.addLayout(layout)
        layout2.addWidget(self._checkbox_active)
        self.setLayout(layout2)
        
    def setActivationView(self, activation: ActivationView) -> None:
        interests = ActivationEngine.Change('input_changed',
                                            'activation_changed')
        self._exchangeView('_activation', activation, interests=interests)

    def activation_changed(self, activation: ActivationEngine,
                           info: ActivationEngine.Change) -> None:
        """The QClassesView is only interested if the classification result
        changes.
        """
        if activation is None:
            return
        
        if info.input_changed:
            #print("CLASSIFICATION: "
            #      f"{activation.input_target} "
            #      f"({activation.input_target_text})")
            #self._target.setText(f"{activation.input_target} "
            #                     f"({activation.input_target_text})")
            self._target.setText(f"{activation.input_target_text}")

        if info.activation_changed:
            classes, scores, target = activation.top_n_classifications(self._top_n)
            target = activation.input_target_text.replace(' ', '_')  # FIXME[hack]
            if classes is not None:
                for i in range(self._top_n):
                    self._classes[i].setText(str(classes[i]))
                    self._classes[i].setToolTip(str(classes[i]))
                    #print(f"{i}: {target} vs. {classes[i]}")
                    if target is None:
                        self._classes[i].setStyleSheet('')
                    elif target == classes[i]:
                        self._classes[i].setStyleSheet(f'color: green')
                    else:
                        self._classes[i].setStyleSheet(f'color: red')
                        self._scores[i].setText(str(scores[i]))
            else:
                for i in range(self._top_n):
                    self._classes[i].setText("None")
                    self._scores[i].setText("0")


