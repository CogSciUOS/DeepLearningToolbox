"""Widgets related to classifiers.
"""

# standard imports
from typing import Set
import logging

# third-party imports
import numpy as np

# Qt imports
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QLabel, QGridLayout

# toolbox imports
from dltb.tool.classifier import ClassIdentifier
from dltb.tool.activation import ActivationWorker
from dltb.network import Layer, Classifier

# GUI imports
from ..utils import QObserver

# logging
LOG = logging.getLogger(__name__)


class QClassesView(QWidget, QObserver, qobservables={
        ActivationWorker: {'tool_changed', 'data_changed', 'work_finished'}}):
    """Visualization of classification results for a
    :py:class:`SoftClassifier`. Class labels are presented as
    as ranked list. Associated class scores may be shown.

    The :py:class:`QClassesView` can observe an
    :py:class:`ActivationWorker` to obtain class scores.

    Attributes
    ----------
    _top: int
        The number of top classifications to be shown.
    _scheme: ClassScheme
        The class scheme according to which classification is done.
    _outputFormat: str
        The output format in which the labels are presented. This
        should be a valid name for a format of the associated
        :py:class:`ClassScheme`.
    _internalFormat: str
        The format in which labels are reported internally. This
        should be a valid name for a format of the associated
        :py:class:`ClassScheme`.

    _classes: np.ndarray
        the ranked class indices
    _scores: np.ndarray
        the associated class scores
    _target: ClassIdentifier
        The index of the target class if known, None otherwise

    Graphical Components
    --------------------
    _classLabels: List[QLabel]
    _scoreLabels: List[QLabel]

    _targetLabel: QLabel
    """

    def __init__(self, top: int = 5, **kwargs):
        """Initialization of the QActivationView.

        Parameters
        -----------
        parent  :   QtWidget
                    Parent widget (passed to super)
        """
        super().__init__(**kwargs)
        self._top = top
        self._target = None
        self._rank = None
        self._classes = None
        self._scores = None
        self._scheme = None
        self._outputFormat = None
        self._internalFormat = None

        self._classLabels = []
        self._scoreLabels = []
        self._initUI()
        self._layoutUI()

    def _initUI(self):
        for _ in range(self._top):
            label = QLabel("None")
            label.setMaximumWidth(label.width() * 2)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._classLabels.append(label)
            self._scoreLabels.append(QLabel("0"))
        self._targetLabel = QLabel("?")
        self._targetLabel.setMaximumWidth(self._targetLabel.width())
        self._rankLabel = QLabel("?")

    def _layoutUI(self):
        gridLayout = QGridLayout()
        self.setLayout(gridLayout)

        gridLayout.addWidget(QLabel("<b>Class</b>", self), 0, 0, Qt.AlignLeft)
        gridLayout.addWidget(QLabel("<b>Score</b>", self), 0, 1, Qt.AlignLeft)
        for i in range(self._top):
            gridLayout.addWidget(self._classLabels[i], i+1, 0)
            gridLayout.addWidget(self._scoreLabels[i], i+1, 1)
        gridLayout.addWidget(QLabel("<b>Correct</b>", self), self._top+1, 0)
        gridLayout.addWidget(self._targetLabel, self._top+2, 0)
        gridLayout.addWidget(self._rankLabel, self._top+2, 1)

    def setTarget(self, target: int,
                  rank: int = None, score: float = None) -> None:
        """Set the target class label. That is the label of the class
        considered to be the correct solution for the current task.
        """
        self._setTarget(target, rank, score)
        self._updateScores()

    def _setTarget(self, target: int,
                   rank: int = None, score: float = None) -> None:
        if self._scheme is not None:
            self._target = ClassIdentifier(target, scheme=self._scheme)
            self._rank = \
                '?' if rank is None else f"rank {rank+1} (score={score})"

    def setClassScores(self, classes: np.ndarray, scores: np.ndarray,
                       target: int = None) -> None:
        """Set the classes and class scores to be presented in this
        :py:class:`QClassesView`.

        Arguments
        ---------
        classes:
            The class indices. This is expected to be an array of `top`
            numeric class indices.
        scores:
            The associated class scores.
        """
        self._setClassScores(classes, scores)
        if target is not None:
            self._setTarget(target)
        self._updateScores()

    def _setClassScores(self, classes: np.ndarray, scores: np.ndarray) -> None:
        if self._scheme is not None and classes is not None:
            self._classes = [ClassIdentifier(index, scheme=self._scheme)
                             for index in classes]
        else:
            self._classes = None
        self._scores = scores

    def _updateScores(self):
        #
        # Update the class labels and scores
        #
        for i in range(self._top):
            if self._classes is None:
                self._classLabels[i].setText('')
                self._classLabels[i].setStyleSheet('')
                self._classLabels[i].setToolTip('')
                self._scoreLabels[i].setText('')
            else:
                classLabel = self._classLabels[i]
                tooltip = ("network unit: "
                           f"{self._classes[i].label(self._internalFormat)}")
                classLabel.setText(self._classes[i].label(self._outputFormat))
                classLabel.setToolTip(tooltip)
                if self._target is None:
                    classLabel.setStyleSheet('')
                elif self._target == self._classes[i]:
                    classLabel.setStyleSheet('color: green')
                else:
                    classLabel.setStyleSheet('color: red')
                self._scoreLabels[i].setText(str(self._scores[i]))

        if self._target is None:
            targetText = "?"
            tooltip = None
        else:
            targetText = self._target.label(self._outputFormat)
            tooltip = ("network unit: "
                       f"{self._target.label(self._internalFormat)}")
        self._targetLabel.setText(targetText)
        self._targetLabel.setToolTip(tooltip)
        self._rankLabel.setText('?' if self._rank is None else str(self._rank))

    #
    # ActivationWorker
    #

    def setActivationWorker(self, worker: ActivationWorker) -> None:
        """Hook to be invoked when the :py:class:`ActivationWorker`
        for this :py:class:`QClassesView` is changed.
        """
        LOG.info("Set ActivationWorker for QImageView: %s", worker)
        network = worker and worker.network
        self._scheme = \
            network.scheme if isinstance(network, Classifier) else None

    def worker_changed(self, worker: ActivationWorker,
                       info: ActivationWorker.Change) -> None:
        # pylint: disable=invalid-name
        """The QClassesView is only interested if the classification result
        changes.
        """
        data = worker.data
        network = worker.network
        network_is_classifier = isinstance(network, Classifier)
        LOG.debug("ActivationWorker changed: %s (%s), is classifier: %s",
                  worker, info, network_is_classifier)

        if info.tool_changed:
            if network_is_classifier:
                self._scheme = network.class_scheme
                self._outputFormat = \
                    ('text' if self._scheme.has_label('text') else None)
                self._internalFormat = network.labeling
                worker.set_classification()
            else:
                self._scheme = None
                self._outputFormat = None
                self._internalFormat = None

        if info.data_changed:
            if data is None or not data.has_attribute('label'):
                self._setTarget(None)
            else:
                self._setTarget(data.label)

        if info.work_finished and network_is_classifier:
            activations = worker.activations(network.score_layer)
            classes, scores = network.top_classes(activations, self._top)
            rank, score = network.class_rank(activations, data.label)
            self._setClassScores(classes, scores)
            self._setTarget(data.label, rank, score)
        else:
            self._setClassScores(None, None)

        self._updateScores()

    # FIXME[todo]: the layers_of_interest mechanism of ActivationWorker
    # does not work yet due to a problem with indirection observation via
    # QObserverHelper.
    def layers_of_interest(self, worker: ActivationWorker) -> Set[Layer]:
        # pylint: disable=invalid-name
        """The :py:class:`QClassesView` is only interested in the score layer
        of a (soft) :py:class:`Classifier` network.

        """
        layers = super().layers_of_interest(worker)
        network = worker and worker.network
        if isinstance(network, Classifier):
            layers |= network.score_layer
        return layers
