
from base.observer import Observable


class Training(Observable, method='trainingChanged',
               changes=['training_changed', 'epoch_changed',
                        'batch_changed', 'metric_changed',
                        'optimizer_changed', 'data_changed',
                        'network_changed']):

    def __init__(self):
        super().__init__()
        self._epochs = 20
        self._batch_size = 128
        self._running = False
        self._epoch = None
        self._batches = None
        self._batch = None
        self._loss = None
        self._accuracy = None
        self._validation_loss = None
        self._validation_accuracy = None
        self._network = None

    def start(self):
        self._running = True
        self.notifyObservers('training_changed')

    def stop(self):
        self._running = False
        self.notifyObservers('training_changed')
        
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

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, network):
        self._network = network
        self._model = network  # FIXME[hack]
        self.notifyObservers('network_changed')

#from controller import Controller

import util

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
        print("Pause Training")
        #if self._training and self._training.running:
        #    self._training.stop()

    def set_epochs(self, epochs):
        self._training._epochs = epochs

    def set_batch_size(self, batch_size):
        self._training._batch_size = batch_size

    # FIXME[hack]:
    def _hackNewModel(self):
        util.runner.runTask(self._hackNewModelHelper)

    def _hackNewModelHelper(self):
        # FIXME[hack]:
        original_dim = self._training._x_train.shape[1]
        intermediate_dim = 512
        latent_dim = 2
        from models.example_keras_vae_mnist import KerasAutoencoder
        self._training.network = KerasAutoencoder(original_dim)

    # FIXME[hack]:
    def _hackNewModel2(self):
        util.runner.runTask(self._hackNewModelHelper2)

    def _hackNewModelHelper2(self):
        # FIXME[hack]:
        from models.example_keras_vae_mnist import KerasAutoencoder
        self._training.network = KerasAutoencoder(original_dim)



