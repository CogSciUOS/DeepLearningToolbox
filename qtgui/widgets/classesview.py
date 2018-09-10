from PyQt5.QtWidgets import QWidget, QLabel, QGridLayout
from observer import Observer
from model import Model

class QClassesView(QWidget, Observer):
    """Visualization of classification results.

    Fields
    ------
    _top_n: int
    _classes: List[QLabel]
    _scores: List[QLabel]
    _target: QLabel
    """
    
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
            layout.addWidget(self._classes[i], i+1, 0)
            self._scores.append(QLabel("0", self))
            layout.addWidget(self._scores[i], i+1, 1)
        layout.addWidget(QLabel("<b>Correct</b>", self), self._top_n+1, 0)
        self._target = QLabel(f"?", self)
        self._target.setMaximumWidth(self._target.width())
        layout.addWidget(self._target, self._top_n+1, 1)
        self.setLayout(layout)


    def modelChanged(self, model, info):
        """The QClassesView is only interested if the classification result
        changes.
        """
        if info.activation_changed:
            classes, scores, target = model.top_n_classifications(self._top_n)

            if classes is not None:
                for i in range(self._top_n):
                    self._classes[i].setText(str(classes[i]))
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
            self._target.setText(str(target))
            self.update()

