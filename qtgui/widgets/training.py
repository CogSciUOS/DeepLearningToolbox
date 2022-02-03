"""
File: training.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
"""

# standard imports
from typing import Optional

# third party imports
import numpy as np

# Qt imports
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QWidget, QProgressBar, QLabel, QCheckBox
from PyQt5.QtWidgets import QPushButton, QSpinBox
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QFormLayout
from .matplotlib import QMatplotlib

# toolbox imports
from dltb.network import Network
from dltb.datasource import Datasource
from dltb.tool.train import Trainer, Trainable


# GUI imports
from ..utils import QObserver, protect


class QTrainingBox(QWidget, QObserver, qobservables={
        Trainer: {'training_started', 'training_ended',
                  'epoch_started', 'epoch_ended',
                  'batch_started', 'batch_ended',
                  'trainee_changed', 'data_changed', 'trainer_changed'}}):
    """A widget for controlling the training process of a model.
    The widgets interacts with:

    * a trainer running the training

    The trainer provides access to further information:
    * a network to be trained
    * a datasource providing training data

    """
    MAX_EPOCHS = 100

    _trainer: Trainer

    
    def __init__(self, trainer: Optional[Trainer] = None,
                 epochs: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self._epochs = epochs

        self._initUI(epochs=epochs)
        self._layoutUI()

        self.setTrainer(trainer)

    def _initUI(self, epochs: int) -> None:
        self._btnStart = QPushButton('Start Training', self)
        self._btnStart.clicked.connect(self.startTraining)

        self._btnStop = QPushButton('Stop Training', self)
        self._btnStop.clicked.connect(self.stopTraining)

        self._epochBar = QProgressBar(self)
        self._epochBar.setMinimum(0)
        self._epochBar.setMaximum(0)

        self._batchBar = QProgressBar(self)
        self._batchBar.setMinimum(0)
        self._batchBar.setMaximum(0)

        self._lblInfo = QLabel()
        self._lblInfo2 = QLabel()

        @protect
        def slot(value: int) -> None:
            self._trainer.epochs = value
        self._spinboxEpochs = QSpinBox()
        self._spinboxEpochs.setRange(1, self.MAX_EPOCHS)
        self._spinboxEpochs.setValue(epochs)
        self._spinboxEpochs.valueChanged.connect(slot)

        # By default, a QWidget does not accept the keyboard focus, so
        # we need to enable it explicitly: Qt.StrongFocus means to
        # get focus by 'Tab' key as well as by mouse click.
        self.setFocusPolicy(Qt.StrongFocus)

    def _layoutUI(self) -> None:
        column = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(self._btnStart)
        row.addWidget(self._btnStop)
        row.addWidget(self._spinboxEpochs)
        column.addLayout(row)
        form = QFormLayout()
        form.addRow("Epochs", self._epochBar)
        form.addRow("Batches", self._batchBar)
        column.addLayout(form)
        column.addWidget(self._lblInfo)
        column.addWidget(self._lblInfo2)
        self.setLayout(column)

    def _updateUI(self) -> None:
        trainer = self._trainer
        haveTrainer = trainer is not None
        trainerReady = haveTrainer and trainer.ready
        trainingRunning = trainerReady and trainer.running
        haveEpochs = haveTrainer and (trainer.epochs is not None)
        finished = haveEpochs and (trainer.epoch == trainer.epochs)
        self._btnStart.setEnabled(trainerReady and not trainingRunning and
                                  not finished)
        self._btnStop.setEnabled(trainerReady and trainingRunning)
        self._epochBar.setVisible(haveEpochs)
        self._epochBar.setEnabled(trainerReady)
        self._batchBar.setVisible(haveEpochs)
        self._batchBar.setEnabled(trainerReady)
        if haveEpochs:
            epoch = trainer.epoch
            batch = trainer.batch
            if epoch is not None:
                self._epochBar.setValue(epoch)
            if batch is not None:
                self._batchBar.setValue(batch)

            if not epoch:
                min_epoch = 1
            elif batch and batch != trainer.batches:
                min_epoch = epoch + 1
            else:
                min_epoch = epoch
            self._spinboxEpochs.setRange(min_epoch, self.MAX_EPOCHS)
            self._spinboxEpochs.setValue(self._trainer.epochs)

    def _updateInfo(self) -> None:
        trainer = self._trainer
        if trainer is None:
            self._lblInfo.setText("No Trainer")
        else:
            info = ("Trainee: " +
                    ('no' if trainer.trainee is None else 'yes') +
                    ", Data: " +
                    ('no' if trainer.training_data is None else 'yes'))
            info2 = (f"Epoch: {trainer.epoch}/{trainer.epochs} "
                     f"[{trainer.steps_per_epoch} steps], "
                     f"Batch: {trainer.batch}/{trainer.batches}, "
                     f"Step: {trainer.step}")
            self._lblInfo.setText(info)
            self._lblInfo2.setText(info2)

    def _updateBars(self) -> None:
        trainer = self._trainer
        if trainer is None or trainer.steps_per_epoch is None:
            epochs = batches = None
        else:
            epochs = trainer.epochs
            batches = trainer.batches

        if epochs is None:
            self._epochBar.setMaximum(0)
            self._epochBar.setValue(0)
        else:
            self._epochBar.setMaximum(epochs)
            self._epochBar.setValue(trainer.epoch)

        if batches is None:
            self._batchBar.setMaximum(0)
            self._batchBar.setValue(0)
        else:
            self._batchBar.setMaximum(batches)
            self._batchBar.setValue(trainer.batch)

    def setTrainer(self, trainer: Trainer) -> None:
        self._updateBars()
        self._updateInfo()
        self._updateUI()

    def trainee(self) -> Trainable:
        """The trainee that is currently trained.
        """
        return None if self._trainer is None else self._trainer.trainee

    def setTrainee(self, trainee: Trainable) -> None:
        """Set the trainee to be trained.
        """
        if self._trainer is not None:
            self._trainer.trainee = trainee

    @protect
    def startTraining(self, checked: bool) -> None:
        """Start the training process.
        """
        if self._trainer is not None:
            self._trainer.train(epochs=self._spinboxEpochs.value(),
                                resume=True, run=True)

    @protect
    def stopTraining(self, checked: bool) -> None:
        """Stop the training process.
        """
        if self._trainer is not None:
            self._trainer.stop()

    #
    # Trainer Observer interface
    #

    def trainer_changed(self, trainer: Trainer, info: Trainer.Change) -> None:
        """React to training changes.
        """
        #print(f"trainer_changed: {info}: "
        #      f"epoch: {trainer.epoch}/{trainer.epochs}"
        #      f" ({self._epochBar.value()}/{self._epochBar.maximum()}), "
        #      f"batch: {trainer.batch}/{trainer.batches}"
        #      f" ({self._batchBar.value()}/{self._batchBar.maximum()})")
        if info.trainee_changed or info.data_changed or info.trainer_changed:
            self._updateInfo()
        elif info.batch_ended or info.epoch_ended:
            self._updateInfo()            
        if info.data_changed or info.trainer_changed:
            self._updateBars()
        self._updateUI()


    #
    # Events
    #

    @protect
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Process key events. The :py:class:`QImageView` supports
        the following keys:

        space: toggle tool tips
        r: toggle the keepAspectRatio flag
        """
        key = event.key()

        if key == Qt.Key_I:
            self._lblInfo.setVisible(not self._lblInfo.isVisible())
        else:
            super().keyPressEvent(event)


class QTrainingInfoBox(QWidget, QObserver, qobservables={
        # FIXME[hack]: check what we are really interested in ...
        Trainer: Trainer.Change.all()}):
    """

    Attributes
    ----------
    range: numpy.ndarray

    trainingLoss: numpy.ndarray

    validationLoss: numpy.ndarray

    rangeIndex: int
    """

    _trainer: Trainer = None

    def __init__(self, trainer: Optional[Trainer] = None, **kwargs):
        """Initialization of the QTrainingBox.
        """
        super().__init__(**kwargs)

        self._initUI()
        self._layoutComponents()

        self._range = np.arange(100, dtype=np.float32)
        self._rangeIndex = 0
        self._trainingLoss = np.zeros(100, dtype=np.float32)
        self._validationLoss = np.zeros(100, dtype=np.float32)

        self.setTrainer(trainer)

    def _initUI(self):
        def slot(checked: bool):
            if self._trainer.ready:
                self._trainer.start()
            elif self._trainer.running:
                self._trainer.stop()
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

        self._lblTrainee = QLabel()

        def slot(value: int):
            self._trainer.epochs = value
        self._spinboxEpochs = QSpinBox()
        self._spinboxEpochs.valueChanged.connect(slot)
        
        def slot(value: int):
            self._trainer.batch_size = value
        self._spinboxBatchSize = QSpinBox()       
        self._spinboxBatchSize.valueChanged.connect(slot)

    def _layoutComponents(self):
        form = QFormLayout()
        form.addRow("Trainee:", self._lblTrainee)
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
        enabled = (self._trainer is not None and self._trainer.ready)
        self._buttonTrainModel.setEnabled(enabled)
        enabled = enabled and not self._trainer.running
        self._spinboxEpochs.setEnabled(enabled)
        self._spinboxBatchSize.setEnabled(enabled)

    def setTrainer(self, trainer: Trainer) -> None:
        self._enableComponents()

    def trainer_changed(self, trainer, change):
        self._enableComponents()
        return

        if 'trainee_changed' in change:
            self._lblTrainee.setText(str(trainer.trainee))
            self._enableComponents()
            
        if 'training_changed' in change:
            if self._trainer.epochs:
                self._progressEpoch.setRange(0, self._trainer.epochs)
            if self._trainer.batches:
                self._progressBatch.setRange(0, self._trainer.batches)

            if self._trainer is not None:
                if self._trainer.running:
                    self._buttonTrainModel.setText("Stop")
                else:
                    self._buttonTrainModel.setText("Train")
            self._enableComponents()

        if 'epoch_changed' in change:
            if self._trainer.epoch is None:
                self._labelEpoch.setText("")
                self._progressEpoch.setValue(0)
            else:
                self._labelEpoch.setText(str(self._trainer.epoch))
                self._progressEpoch.setValue(self._trainer.epoch+1)

        if 'batch_changed' in change:
            if self._trainer.batch is not None:
                self._labelBatch.setText(f"{self._trainer.batch}/"
                                         f"{self._trainer.batches}")
                self._labelDuration.setText(str(self._trainer.batch_duration))
                self._progressBatch.setValue(self._trainer.batch)

        if 'parameter_changed' in change:
            self._spinboxEpochs.setRange(*self._trainer.epochs_range)
            self._spinboxEpochs.setValue(self._trainer.epochs)
            self._spinboxBatchSize.setRange(*self._trainer.batch_size_range)
            self._spinboxBatchSize.setValue(self._trainer.batch_size)

        if self._trainer.loss is not None:
            self._labelLoss.setText(str(self._trainer.loss))
            self._trainerLoss[self._rangeIndex] = self._trainer.loss
            if self._checkboxPlot.checkState():
                self._plotLoss.plot(self._range, self._trainerLoss)
            # self._plotLoss.plot(self._validationLoss)

        if self._trainer.accuracy is not None:
            self._labelAccuracy.setText(str(self._trainer.accuracy))

        self._rangeIndex = (self._rangeIndex + 1) % len(self._range)
