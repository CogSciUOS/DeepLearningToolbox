import numpy as np

from PyQt5.QtWidgets import (QWidget, QProgressBar, QLabel, QCheckBox,
                             QVBoxLayout, QFormLayout)
from .matplotlib import QMatplotlib

from qtgui.utils import QObserverWidget
from base.observer import TrainingObservable

class QTrainingBox(QObserverWidget, TrainingObservable.Observer):

    def __init__(self, parent=None):
        """Initialization of the QTrainingBox.
        """
        super().__init__(parent)

        self._initUI()
        self._layoutComponents()
        self._range = np.arange(100, dtype=np.float32)
        self._trainingLoss = np.zeros(100, dtype=np.float32)
        self._validationLoss = np.zeros(100, dtype=np.float32)
        self._rangeIndex = 0

    def _initUI(self):
        self._plotLoss = QMatplotlib()
        
        self._progressEpoch = QProgressBar()
        self._progressEpoch.setFormat("%v/%m")
        self._progressBatch = QProgressBar()
        self._progressBatch.setFormat("%v (%p%)")

        self._labelBatch = QLabel()
        self._labelEpoch = QLabel()
        self._labelLoss = QLabel()
        self._labelAccuracy = QLabel()
        self._labelDuration = QLabel()
        
        self._checkboxPlot = QCheckBox()

    def _layoutComponents(self):
        form = QFormLayout()
        form.addRow("Batch:", self._labelBatch)
        form.addRow("Epoch:", self._labelEpoch)
        form.addRow("Loss:", self._labelLoss)
        form.addRow("Accuracy:", self._labelAccuracy)
        form.addRow("Duration:", self._labelDuration)
        form.addRow("Plot:", self._checkboxPlot)

        layout = QVBoxLayout()
        layout.addWidget(self._plotLoss)
        layout.addLayout(form)
        layout.addWidget(self._progressBatch)
        layout.addWidget(self._progressEpoch)
        self.setLayout(layout)

    def trainingChanged(self, training, change):
        if 'training_changed' in change:
            if training.epochs:
                self._progressEpoch.setRange(0, training.epochs)
            if training.batches:
                self._progressBatch.setRange(0, training.batches)

        if 'epoch_changed' in change:
            self._labelEpoch.setText(str(training.epoch))
            self._progressEpoch.setValue(training.epoch+1)

        if 'batch_changed' in change:
            self._labelBatch.setText(f"{training.batch}/"
                                     f"{training.batches}")
            self._labelDuration.setText(str(training.batch_duration))
            self._progressBatch.setValue(training.batch)

        if training.loss is not None:
            self._labelLoss.setText(str(training.loss))
            self._trainingLoss[self._rangeIndex] = training.loss
            if self._checkboxPlot.checkState():
                self._plotLoss.plot(self._range, self._trainingLoss)
            # self._plotLoss.plot(self._validationLoss)

        if training.accuracy is not None:
            self._labelAccuracy.setText(str(training.accuracy))

        self._rangeIndex = (self._rangeIndex + 1) % len(self._range)
