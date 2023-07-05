"""Widgets related to classifiers.
"""

# standard imports
from typing import Set, Optional
import logging
import itertools

# third-party imports
import numpy as np

# Qt imports
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QWidget, QLabel, QGridLayout, QLayout

# toolbox imports
from dltb.tool.classifier import ClassIdentifier
from dltb.tool.activation import ActivationWorker
from dltb.network import Layer, Classifier

# GUI imports
from ..utils import QObserver, protect

# logging
LOG = logging.getLogger(__name__)


class QClassesView(QWidget, QObserver, qobservables={
        ActivationWorker: {'tool_changed', 'data_changed',
                           'worker_changed', 'work_finished'}}):
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

    _classes: np.ndarray, dtype=ClassIdentifier
        the ranked class indices
    _scores: np.ndarray
        the associated class scores
    _target: ClassIdentifier
        The index of the target class if known, None otherwise

    Graphical Components
    --------------------
    Each of the following lists has length `TOP_MAX+2`.  The last two
    entries are reserved to display the true target label.

    _classRanks: List[QLabel]
        Label for displaying the rank.
    _classLabels: List[QLabel]
        Label for displaying the class label.
    _scoreLabels: List[QLabel]
        Label for displaying the class score.
    """
    TOP_MIN: int = 1
    TOP_MAX: int = 10

    def __init__(self, top: int = 5, **kwargs):
        """Initialization of the QActivationView.

        Parameters
        -----------
        parent  :   QtWidget
                    Parent widget (passed to super)
        """
        super().__init__(**kwargs)
        self._top = top

        self._targetClass = None
        self._targetRank = None
        self._targetScore = None

        self._classes = None
        self._scores = None
        self._scheme = None
        self._outputFormat = None
        self._internalFormat = None

        self._classLabels = []
        self._scoreLabels = []
        self._rankLabels = []
        self._initUI()
        self._layoutUI()

        # By default, a QWidget does not accept the keyboard focus, so
        # we need to enable it explicitly: Qt.StrongFocus means to
        # get focus by 'Tab' key as well as by mouse click.
        self.setFocusPolicy(Qt.StrongFocus)

    def _initUI(self):
        width = None
        for row in range(self.TOP_MAX + 2):
            label = QLabel("None")
            if width is None:
                width = label.fontMetrics().boundingRect(label.text()).width()
            label.setMinimumWidth(width)
            label.setMaximumWidth(width * 4)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._classLabels.append(label)

            label = QLabel("0")
            baseWidth = label.width()
            self._scoreLabels.append(label)

            label = QLabel(str(row+1))
            label.setMaximumWidth(baseWidth*4)
            self._rankLabels.append(label)

    def _layoutUI(self, oldTop: int = -2) -> None:
        if oldTop < 0:
            gridLayout = QGridLayout()
            # gridLayout.setSizeConstraint(QLayout.SetMinimumSize)
            gridLayout.addWidget(QLabel("<b>Rank</b>", self),
                                 0, 0, 1, 1, Qt.AlignLeft)
            gridLayout.addWidget(QLabel("<b>Class</b>", self),
                                 0, 1, 1, 3, Qt.AlignLeft)
            gridLayout.addWidget(QLabel("<b>Score</b>", self),
                                 0, 4, 1, 2, Qt.AlignLeft)
            self.setLayout(gridLayout)
        else:
            gridLayout = self.layout()

            # remove items that are over the top
            for i in range(self.top()+2, oldTop+2):
                gridLayout.removeItem(gridLayout.itemAtPosition(i+1, 0))
                gridLayout.removeItem(gridLayout.itemAtPosition(i+1, 1))
                gridLayout.removeItem(gridLayout.itemAtPosition(i+1, 4))

        # add items to fill new top
        for i in range(oldTop+2, self.top()+2):
            gridLayout.addWidget(self._rankLabels[i], i+1, 0)
            gridLayout.addWidget(self._classLabels[i], i+1, 1, 1, 3)
            gridLayout.addWidget(self._scoreLabels[i], i+1, 4, 1, 2)

    def top(self) -> int:
        """The number of top classes to display."""
        return self._top

    def setTop(self, top: int) -> None:
        """Set the number of top classes to display."""
        top = min(max(top, self.TOP_MIN), self.TOP_MAX)
        if top != self._top:
            # update the layout to reflect the new 'top' value
            oldTop, self._top = self._top, top
            # FIXME[bug]: changing the layout may leave artifacts at the
            # bottom of the widget. I have not yet found a way to prevent
            # this (the following does not help - maybe we have would need
            # a repaint, or clean the background?)
            # if self._top < oldTop:
            #     self._rankLabels[self._top+1].setText('')
            self._layoutUI(oldTop=oldTop)

            if self._top > oldTop:
                # we will display more values - get these values from
                # the ActivationWorker if available.
                self._setClassScoresFromWorker(self._activationWorker)

            self._updateScores()
            # Neither of the following helps to fix the above mentioned bugx
            # self.update()
            # self.repaint()

    def setTarget(self, target: int,
                  rank: int = None, score: float = None) -> None:
        """Set the target class label. That is the label of the class
        considered to be the correct solution for the current task.
        """
        self._setTarget(target, rank, score)
        self._updateScores()

    def _setTarget(self, target: Optional[int], rank: Optional[int] = None,
                   score: Optional[float] = None) -> None:
        """Set the target (ground truth) label.

        Arguments
        ---------
        target:
            The target class.
        rank:
            The rank the target was assigned to by the :py:class:`Classifier`.
        score:
            The confidence score that was assigned to the target by
            the :py:class:`Classifier`.
        """
        if self._scheme is not None and target is not None:
            self._targetClass = ClassIdentifier(target, scheme=self._scheme)
            self._targetRank = rank
            self._targetScore = score
        else:
            self._targetClass = None
            self._targetRank = None
            self._targetScore = None

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
        top = self.top()
        if self._classes is None:
            targetRow, ellipsisRow, emptyRow = None, None, None
        elif self._targetRank is not None:
            if self._targetRank < top:
                targetRow, ellipsisRow, emptyRow = self._targetRank, top, top+1
            elif self._targetRank == top:
                targetRow, ellipsisRow, emptyRow = top, top+1, None
            else:
                targetRow, ellipsisRow, emptyRow = top+1, top, None

        for row in itertools.chain(range(top+2)):

            if self._classes is None or row == emptyRow:
                self._rankLabels[row].setText('')
                self._classLabels[row].setText('')
                self._classLabels[row].setStyleSheet('')
                self._classLabels[row].setToolTip('')
                self._scoreLabels[row].setText('')
                continue

            if row == ellipsisRow:
                self._rankLabels[row].setText('...')
                self._classLabels[row].setText('...')
                self._classLabels[row].setStyleSheet('')
                self._classLabels[row].setToolTip('')
                self._scoreLabels[row].setText('...')
                continue

            styleSheet = 'color: green' if row == targetRow else 'color: red'
            rank = self._targetRank if row == targetRow else row
            rankText = '?' if rank is None else str(rank+1)
            if row == targetRow:
                classIdentifier, score = self._targetClass, self._targetScore
            elif rank < len(self._classes):
                classIdentifier, score = \
                    self._classes[rank], self._scores[rank]
            else:
                classIdentifier, score = None, None
            tooltip = '' if classIdentifier is None else \
                f"{classIdentifier.label(self._outputFormat)} (network " \
                f"unit: {classIdentifier.label(self._internalFormat)})"

            self._rankLabels[row].setText(rankText)

            classLabel = self._classLabels[row]
            classLabel.setStyleSheet(styleSheet)
            classLabel.setText('?' if classIdentifier is None else
                               classIdentifier.label(self._outputFormat))
            classLabel.setToolTip(tooltip)

            self._scoreLabels[row].setText("-" if score is None else
                                           str(score))

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
        LOG.debug("ActivationWorker changed: %s (%s), is classifier: %s/%s",
                  worker, info, network_is_classifier, worker.classification)

        if not worker.classification and not info.worker_changed:
            return  # nothing of interest

        if info.tool_changed or info.worker_changed:
            if network_is_classifier and worker.classification:
                self._scheme = network.class_scheme
                self._outputFormat = \
                    ('text' if self._scheme.has_label('text') else None)
                self._internalFormat = network.labeling
                worker.classification = True
            else:
                self._scheme = None
                self._outputFormat = None
                self._internalFormat = None

        if info.data_changed:
            if data is None or not data.has_attribute('label'):
                # we don't have a target label in the data.
                self._setTarget(None)
            else:
                self._setTarget(data.label)

        if info.work_finished and network_is_classifier:
            self._setClassScoresFromWorker(worker)
        else:
            self._setClassScores(None, None)

        self._updateScores()

    def _setClassScoresFromWorker(self, worker: ActivationWorker) -> None:
        network = None if worker is None else worker.network
        data = None if worker is None else worker.data

        if not isinstance(network, Classifier) or data is None:
            self._setClassScores(None, None)
            return
        
        activations = worker.activations(network.score_layer)
        classes, scores = network.top_classes(activations, self.top())
        if data is None or data.label is None:
            label = None
            rank, score = None, None
        else:
            label = data.label
            rank, score = network.class_rank(activations, label)
        self._setClassScores(classes, scores)
        self._setTarget(label, rank, score)

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


    #
    # Events
    #

    @protect
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Process key events. The :py:class:`QImageView` supports
        the following keys:

        +/-: change maximal/minimal number of classes to display
        r: toggle the keepAspectRatio flag
        """
        key = event.key()

        if key == Qt.Key_Plus:
            self.setTop(self.top()+1)
        elif key == Qt.Key_Minus:
            self.setTop(self.top()-1)
