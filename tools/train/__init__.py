# FIXME[concept]:
# For training we need:
#  * a network to be trained
#  * training parameter
#     - epochs
#     - batch_size
#     - an optimizer (sgd, adam, ...) with specific parameters
#        - learning rate, ...
#        - momentum    
#  * training data
#  * an training engine: some implementation that puts all this together, e.g.
#     - for a KerasNetwork, we can use the fit() function
#
# Here is my idea:
#  - the Training will hold all this together


from base import BusyObservable, Controller as BaseController, View as BaseView


class Training(BusyObservable, method='training_changed',
               changes={'training_changed', 'epoch_changed',
                        'batch_changed', 'metric_changed',
                        'optimizer_changed', 'data_changed',
                        'network_changed', 'parameter_changed'},
               changeables={
                   'epochs': 'parameter_changed',
                   'epochs_range': 'parameter_changed',
                   'batch_size': 'parameter_changed',
                   'batch_size_range': 'parameter_changed',
                }):
    """
    Attributes
    ----------
    running: bool
        A flag indicating ongoing training.

    Changes
    -------
    training_changed:
        Emitted when the training status (running) was changed.
    epoch_changed:
        Emitted during training when the a new epoch was started.
    batch_changed:
        Emitted during training when a new batch was started.
    metric_changed:
        Emitted during training when new metrics (like loss,
        accuracy, etc.) are available.
    optimizer_changed:
        Currently not used ...
    data_changed:
        Emitted when the data set (training or validation data)
        was changed.
    network_changed:
        The :py:class:`Network` to be trained was changed. This
        means some fundamental change, like a new architecture.
        It is not fired for weight (model parameter) changes (which
        obviously occur all the time during training).
    parameter_changed:
        Some training parameter like epochs, epochs_range or
        batch_size (or batch_size_range) was changed.
    """
    def __init__(self):
        super().__init__()
        self.epochs = 20
        self.epochs_range = (1, 40)
        self.batch_size = 128
        self.batch_size_range = (1, 256)

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
        self.running = True
        self._network.busy_start("training")  # FIXME[old]: check the busy concept

        self._network.change('weights_changed')

    def stop(self):
        self._network.busy_stop("training")  # FIXME[old]: check the busy concept
        self.running = False

    @property
    def running(self):
        """The training is currently ongoing. It may be stopped by calling
        the stop method.
        """
        return self._running

    @running.setter
    def running(self, state: bool):
        if state != self._running:
            self._running = state
            self.notify_observers('training_changed')

    #@batch_size.setter
    def old_batch_size(self, size: int):
        self._training
        if self._training.batch_size_min <= size <= self._training.batch_size_max:
            if self._batch_size != size:
                self._batch_size = size
        else:
            raise ValueError(f"Invalid batch size {size}:"
                             f"allowed range is from {self.batch_size_min}"
                             f"to {self.batch_size_max}")

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

    # FIXME[old]
    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, network):
        self._network = network
        self._model = network  # FIXME[hack]
        print(f"Training({self}).notify_observers(network_changed)")
        self.notify_observers('network_changed')


class TrainingView(BaseView, view_type=Training):

    def __init__(self, training: Training = None, **kwargs):
        super().__init__(observable=training, **kwargs)


class TrainingController(TrainingView, BaseController):
    """A TrainingController is a Controller for some Training."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def ready(self):
        return self._training is not None and not self._training.running

    @property
    def running(self):
        return self._training is not None and self._training.running

    def start(self, network: 'Network', data):
        if self.ready:
            self.run(self._training.start)

    def stop(self):
        if self.running:
            self._training.stop()

    def pause(self):
        if self.running:
            print("Pause Training not implemented yet")
