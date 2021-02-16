"""Widgets related to classifiers.
"""

# Qt imports
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QLabel, QCheckBox,
                             QGridLayout, QVBoxLayout)

# toolbox imports
from dltb.tool.activation import ActivationWorker
from network import Classifier

# GUI imports
from ..utils import QObserver


class QClassesView(QWidget, QObserver, qobservables={
        ActivationWorker: {'data_changed', 'work_finished'}}):
    """Visualization of classification results.

    Attributes
    ----------
    _top_n: int
        The number of top classifications to be shown.
    _target: int
        The index of the target class if known, None otherwise
    _datasource: Datasource
        Datasource used to look up labels.
    _label_format: str
        The format in which the labels are provided by the network.
    _output_format: str
        The output format in which the labels are presented.

    Graphical Components
    --------------------
    _classes: List[QLabel]
    _scores: List[QLabel]

    _targetLabel: QLabel
    """

    def __init__(self, parent: QWidget = None, top_n=5):
        """Initialization of the QActivationView.

        Parameters
        -----------
        parent  :   QtWidget
                    Parent widget (passed to super)
        """
        super().__init__(parent)
        self._top_n = top_n
        self._target = None

        self._datasource = None
        self._label_format = None
        self._output_format = None

        self._classes = []
        self._scores = []
        self._initUI()
        self._layoutUI()

    def _initUI(self):
        for i in range(self._top_n):
            self._classes.append(QLabel(f"None", self))
            self._classes[i].setMaximumWidth(self._classes[i].width())
            self._classes[i].setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._scores.append(QLabel("0", self))
        self._targetLabel = QLabel(f"?", self)
        self._targetLabel.setMaximumWidth(self._targetLabel.width())

    def _layoutUI(self):
        gridLayout = QGridLayout()
        self.setLayout(gridLayout)

        gridLayout.addWidget(QLabel(f"<b>Class</b>", self), 0, 0, Qt.AlignLeft)
        gridLayout.addWidget(QLabel(f"<b>Score</b>", self), 0, 1, Qt.AlignLeft)
        for i in range(self._top_n):
            gridLayout.addWidget(self._classes[i], i+1, 0)
            gridLayout.addWidget(self._scores[i], i+1, 1)
        gridLayout.addWidget(QLabel("<b>Correct</b>", self), self._top_n+1, 0)
        gridLayout.addWidget(self._targetLabel, self._top_n+2, 0)

    def setEnabled(self, enabled: bool):
        super().setEnabled(enabled)

    def changeEvent(self, event):
        super().changeEvent(event)

    def worker_changed(self, worker: ActivationWorker,
                       info: ActivationWorker.Change) -> None:
        """The QClassesView is only interested if the classification result
        changes.
        """

        if info.network_changed:
            if isinstance(worker.tool.network, Classifier):
                scheme = worker.tool.network.class_scheme
                self._output_format = \
                    ('text' if scheme.has_label('text') else None)
                if worker.tool.network.prepared:
                    worker.add_layer(worker.network.score_layer)

        if info.input_changed:
            if self._datasource is None:
                # self._target = worker.input_label # FIXME[old]
                self._target = '?'
                label_text = f'"{self._target}"'
            elif self._datasource == worker.input_datasource:
                try:
                    self._target = self._datasource.\
                        format_labels(worker.input_label,
                                      target=self._label_format)
                except ValueError:  # Format is not supported by datasource
                    self._target = '?' + str(worker.input_label)
                try:
                    label_text = str(self._datasource.
                                     format_labels(worker.input_label,
                                                   target=self._output_format))
                except ValueError:  # Format is not supported by datasource
                    label_text = '?' + str(worker.input_label)
            else:
                self._target = None
                label_text = "?"
            self._targetLabel.setText(label_text)
            self._targetLabel.setToolTip(None if self._target is None else
                                         f"network unit: {self._target}")

        if info.worker_changed or info.input_changed:
            self._update_scores()

    def _update_scores(self):
        if self._activationWorker is None:
            return

        classes, scores = self._activationWorker.top_classes(self._top_n)
        labels = classes
        target = self._target

        #
        # Getting labels from the datasource
        #
        if classes is not None and self._datasource is not None:
            labels = self._datasource.\
                format_labels(classes, target=self._output_format,
                              origin=self._label_format)

        for i in range(self._top_n):
            if classes is None:
                self._classes[i].setText('')
                self._classes[i].setStyleSheet('')
                self._classes[i].setToolTip('')
                self._scores[i].setText('')
            else:
                self._classes[i].setText(str(labels[i]))
                if self._datasource is not None:
                    self._classes[i].setToolTip(f"network unit: {classes[i]}")
                if target is None:
                    self._classes[i].setStyleSheet('')
                elif target == classes[i]:
                    self._classes[i].setStyleSheet(f'color: green')
                else:
                    self._classes[i].setStyleSheet(f'color: red')
                self._scores[i].setText(str(scores[i]))
