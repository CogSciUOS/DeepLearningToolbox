
from base.observer import Observable


class Training(Observable, method='trainingChanged', default='training_changed',
               changes=['training_changed', 'epoch_changed',
                        'batch_changed', 'metric_changed']):

    def __init__(self):
        super().__init__()
        self._running = False
        self._epochs = None
        self._epoch = None
        self._batches = None
        self._batch = None
        self._loss = None
        self._accuracy = None
        self._validation_loss = None
        self._validation_accuracy = None


    def start(self):
        self._running = True

    def stop(self):
        self._running = False
        
    @property
    def running(self):
        """The training is currently ongoing. It may be stopped by calling
        the stop method.
        """
        return self._running

    @property
    def epoch(self):
        return self._epoch

    @property
    def epochs(self):
        return self._epochs

    @property
    def batch(self):
        return self._batch

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def batches(self):
        return self._batches

    @property
    def loss(self):
        return self._loss

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def validation_loss(self):
        return self._validation_loss

    @property
    def validation_accuracy(self):
        return self._validation_accuracy

#from controller import Controller


class Trainer:  # (Controller):
    """A Trainer is a Controller for some training."""

    def __init__(self, training):
        self._training = training

    def start(self):
        if self._training and not self._training.running:
            util.runner.runTask(self._training.start)

    def stop(self):
        if self._training and self._training.running:
            self._training.stop()

    def pause(self):
        if self._training and self._training.running:
            self._training.stop()
