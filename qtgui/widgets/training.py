import numpy as np

from PyQt5.QtWidgets import (QWidget, QProgressBar, QLabel, QCheckBox,
                             QPushButton, QSpinBox, QVBoxLayout, QFormLayout)
from .matplotlib import QMatplotlib

from qtgui.utils import QObserver
from tools.train import Training, TrainingController
from network import Network, View as NetworkView

class QTrainingBox(QWidget, QObserver, Training.Observer, Network.Observer):
    """

    Attributes
    ----------
    range: numpy.ndarray

    trainingLoss: numpy.ndarray

    validationLoss: numpy.ndarray

    rangeIndex: int
    """

    _training: TrainingController = None
    _network: NetworkView = None

    def __init__(self, training: TrainingController=None,
                 network: NetworkView=None, parent=None):
        """Initialization of the QTrainingBox.
        """
        super().__init__(parent)

        self._initUI()
        self._layoutComponents()
        self._range = np.arange(100, dtype=np.float32)
        self._trainingLoss = np.zeros(100, dtype=np.float32)
        self._validationLoss = np.zeros(100, dtype=np.float32)
        self._rangeIndex = 0

        self.setTraining(training)
        self.setNetwork(network)

    def _initUI(self):
        def slot(checked: bool):
            if self._training.ready:
                self._training.start()
            elif self._training.running:
                self._training.stop()
        self._buttonTrainModel = QPushButton("Train")
        self._buttonTrainModel.clicked.connect(slot)

        self._plotLoss = QMatplotlib()
        self._checkboxPlot = QCheckBox()
        
        self._progressEpoch = QProgressBar()
        self._progressEpoch.setFormat("%v/%m")
        self._progressBatch = QProgressBar()
        self._progressBatch.setFormat("%v (%p%)")

        self._labelBatch = QLabel()
        self._labelEpoch = QLabel()
        self._labelLoss = QLabel()
        self._labelAccuracy = QLabel()
        self._labelDuration = QLabel()

        self._labelNetwork = QLabel()

        def slot(value: int):
            self._training.epochs = value
        self._spinboxEpochs = QSpinBox()
        self._spinboxEpochs.valueChanged.connect(slot)
        
        def slot(value: int):
            self._training.batch_size = value
        self._spinboxBatchSize = QSpinBox()       
        self._spinboxBatchSize.valueChanged.connect(slot)

    def _layoutComponents(self):
        form = QFormLayout()
        form.addRow("Network:", self._labelNetwork)
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
        layout.addWidget(self._buttonTrainModel)
        layout.addWidget(self._spinboxEpochs)
        layout.addWidget(self._spinboxBatchSize)
        self.setLayout(layout)

    def _enableComponents(self):
        enabled = (self._network is not None and
                   self._training is not None and self._training.ready)
        self._buttonTrainModel.setEnabled(enabled)
        enabled = enabled and not self._training.running
        self._spinboxEpochs.setEnabled(enabled)
        self._spinboxBatchSize.setEnabled(enabled)

    def setTraining(self, training: TrainingController):
        self._exchangeView('_training', training)
        # FIXME[test]: should be notified by the observable
        self._enableComponents()

    def setNetwork(self, network: NetworkView):
        self._exchangeView('_network', network)

    def network_changed(self, network, change):
        self._network(network)
        self._labelNetwork.setText(str(network))

    def training_changed(self, training, change):
        self._training(training)
        self._enableComponents()
        return
        
        if 'network_changed' in change:
            self._enableComponents()
            
        if 'training_changed' in change:
            if self._training.epochs:
                self._progressEpoch.setRange(0, self._training.epochs)
            if self._training.batches:
                self._progressBatch.setRange(0, self._training.batches)

            if self._training is not None:
                if self._training.running:
                    self._buttonTrainModel.setText("Stop")
                else:
                    self._buttonTrainModel.setText("Train")
            self._enableComponents()

        if 'epoch_changed' in change:
            if self._training.epoch is None:
                self._labelEpoch.setText("")
                self._progressEpoch.setValue(0)
            else:
                self._labelEpoch.setText(str(self._training.epoch))
                self._progressEpoch.setValue(self._training.epoch+1)

        if 'batch_changed' in change:
            if self._training.batch is not None:
                self._labelBatch.setText(f"{self._training.batch}/"
                                         f"{self._training.batches}")
                self._labelDuration.setText(str(self._training.batch_duration))
                self._progressBatch.setValue(self._training.batch)

        if 'parameter_changed' in change:
            self._spinboxEpochs.setRange(*self._training.epochs_range)
            self._spinboxEpochs.setValue(self._training.epochs)
            self._spinboxBatchSize.setRange(*self._training.batch_size_range)
            self._spinboxBatchSize.setValue(self._training.batch_size)

        if self._training.loss is not None:
            self._labelLoss.setText(str(self._training.loss))
            self._trainingLoss[self._rangeIndex] = self._training.loss
            if self._checkboxPlot.checkState():
                self._plotLoss.plot(self._range, self._trainingLoss)
            # self._plotLoss.plot(self._validationLoss)

        if self._training.accuracy is not None:
            self._labelAccuracy.setText(str(self._training.accuracy))

        self._rangeIndex = (self._rangeIndex + 1) % len(self._range)
