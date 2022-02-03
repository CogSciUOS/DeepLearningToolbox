"""Traning logic for trainable objects.

The training
============

* training data: possibly multiple ones, e.g., in case of semi-supervised
  training with labeled and unlabeled data

* training typically proceeds in several epochs. In each epoch the complete
  dataset is presented to the trainee.

* each epoch can be subdivided into batches.

* it should be possible to interrupt the training and to resume the training
  from exactly that point later on.  Interruption should be possilbe via
  setting a flag (e.g., in an asynchronous invocation) or by a
  :py:class:`KeyboardInterrupt`.

* it should be possible to store the current state (snapshot/checkpoint) and
  to to restore this state later on.  Automatic checkpointing should be
  supported (at least once per epoch, but potentially even more frequently in
  case of large datasets)

* during training some statistics and other information should be collected

* optional: presentation of a progress bar or development of the loss function(s)
  (not part of the trainer).


trainer.start(model, datasource)

"""

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


# standard imports
from typing import Optional, Any, Collection, Union, Sequence
from collections.abc import Sized
import math
import sys

# thirdparty modules
import numpy as np

# toolbox imports
from ..base import Observable
from ..base.busy import BusyObservable, busy
from ..base.prepare import Preparable
from ..datasource import Datasource


class Trainable:
    """A `Trainable` can be trained by providing single examples
    or batches of examples.
    """

    def prepare_training(self, restore: bool = False) -> None:
        """allocate resources required for training.
        """

    def clean_training(self) -> None:
        """free up resources required for training
        """

    def train_single(self, example) -> None:
        """Train this `Trainable` on a single example.
        """

    def train_batch(self, batch, epoch: int) -> None:
        """Train this `Trainable` on a batch of data.
        """

    def store_checkpoint(self) -> None:
        """Store current state in a checkpoint.
        """
        # storing should include training_step (epoch/batch)

    def restore_from_checkpoint(self) -> None:
        """Restore state from a checkpoint.
        Restored information should include training_step (epoch/batch)
        """

    @staticmethod
    def get_hyperparamter(name: str) -> Any:
        """Get hyperparamters for thie `Trainable`.
        """
        raise KeyError(f"Unknown hyperparamter '{name}'")

    @property
    def training_statistics(self) -> Collection[str]:
        """The training statistics this `Trainable` can provide.
        """
        return set()


class Stoppable(Observable):
    """A process that can be stopped.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._run_flag = False
        self._stop_flag = False

        # _keyboard_interrupt_policy
        #  - 'store': store current state
        #  - 'ignore': ignore the keyboard interrupt (continue running)
        #  - 'message': output a message informing about the interrupt
        self._keyboard_interrupt_policy = set(('ignore', 'message'))
        self._keyboard_interrupt = None

    @property
    def running(self) -> bool:
        """A flag indicating if the `Trainer` is currently running a
        training process.
        """
        return self._run_flag

    @property
    def stopping(self) -> bool:
        """A flag indicating that the `Trainer` is currently stopping a running
        training process.
        """
        return self._stop_flag is not False

    def _begin_run(self) -> None:
        """Begin a run.
        """
        self._stop_flag = False
        self._run_flag = True

    def _end_run(self) -> None:
        """End a run.
        """
        self._stop_flag = False
        self._run_flag = False

    def stop(self) -> None:
        """Interrupt the training.
        This method is intended to be called from another thread to stop
        a training started with :py:meth:`start`.
        """
        if self._run_flag and self._stop_flag is False:
            self._stop_flag = True

    def _handle_keyboard_interrupt(self, ex: KeyboardInterrupt) -> None:
        if 'ignore' in self._keyboard_interrupt_policy:
            if 'message' in self._keyboard_interrupt_policy:
                print("Ignoring keyboard interrupt")
        else:
            if 'message' in self._keyboard_interrupt_policy:
                print("Stopping due to keyboard interrupt ...")
                self.stop()
                self._keyboard_interrupt = ex


class Trainer(Stoppable, Preparable, BusyObservable, method='trainer_changed',
              changes={'batch_started', 'batch_ended', 'epoch_started',
                       'epoch_ended', 'training_started', 'training_ended',
                       'trainee_changed', 'data_changed', 'trainer_changed'}):
    """A `Trainer` trains a trainee on some training data.

    Training data
    -------------

    Training data are the basis on which the trainee can improve its
    performance.

    Epochs
    ------

    If the training data is finite in size (it is `Sized`), then
    presenting the complete data to the trainee is referred to as an
    epoch.  Training may require multiple epochs, meaning the the
    training data have to be shown to the trainee more than once.

    Batch training and steps
    ------------------------

    This `Trainer` implements batch training, meaning she applies an
    iterative approach, presenting at each iteration a batch of data
    to the trainee.  The trainee should use these batches to improve
    her performance, meaning that she should be batch trainable.

    Each iteration is referred to as one step.  The `Trainer` keeps
    track how many training steps have been performed.

    Statistics
    ----------

    The `Trainer` can collect statistics describing the progress of
    the trainee.  What type of statistics can be collected depends on
    the type of the trainee, and should be reported by the
    :py:meth:`Trainable.training_statistics` method.

    Statistics can be recorded for every step or just for selected
    steps.

    Statistics may include: loss, accuracy, (validation_loss,
    validation_accuracy), ...


    Starting, stopping, and resuming the training
    ---------------------------------------------

    The `Trainer` allows to stop the training and to resume it later
    on.

    Keyboard interrupt
    ------------------

    How to deal with Keyboard interrupts?

    - ignore: do not react at all
    - stop: stop the training (and store checkpoint)
    - abort: stop the training (without storing a checkpoint)

    - message: issue a message
    - catch:

    Loss and history
    ----------------

    The trainer can record training progress in form of the development
    of the loss value(s).  Each loss is referred to by its (unique) name,
    which depends on and can be obtained from the :py:class:`Trainable`.

    Recording of loss values can be controlled by the `record`
    parameter.  This is a list of names of all loss values to be
    recorded (the value `True` indicates that all loss values should be
    recorded while `False` means that no loss value is recorded).

    Changes
    -------
    batch_started:
        An new batch will be started.
    batch_ended:
        A batch has been trained.
    epoch_started:
        An new epoch will be started.
    epoch_ended:
        An epoch ended.
    training_started:
        The training has started.
    training_ended:
        The training has ended.
    trainee_changed:
        The trainee was changed.
    data_changed:
        The training data were changed.

    """

    # epochs_range: minimal and maximal value for the epochs parameter
    # (currently not used)
    epochs_range = (1, 40)

    # batch_size_range: minimal and maximal value for the batch_size
    # parameter (currently not used)
    batch_size_range = (1, 256)

    # _history: the history of training statistics
    _history = None

    def __init__(self, trainee: Optional[Trainable] = None,
                 training_data:  Optional[Datasource] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._trainee = trainee
        self._training_data = training_data
        self._step = 0
        self._batch_size = 128

        # _history: a list of all training statistics collected
        #    from the trainee during training
        # _last_batch_result
        self._last_batch_result = None

        # _start_step and _end_step hold the first and last step
        # in the current training run.  These data allow to obtain
        # batch and epoch in the current training run.
        self._start_step = None
        self._end_step = None

        # The _batch_iterator gives batchwise access to the training data.
        # It will be (re)set when a new training run is started and
        # will be kept if the training is paused, allowing to resume
        # the training later on.
        self._batch_iterator = None

        # Hook (function, arguments) to run at the end of every epoch
        self._at_end_of_epoch = None
        self._at_end_of_batch = None

    #
    # Implementation of Preparable
    #

    def _prepared(self) -> bool:
        return self._history is not None and super()._prepared()

    def _prepare(self) -> None:
        super()._prepare()
        self._history = []

    def _unprepare(self) -> None:
        del self._history
        super()._unprepare()

    @property
    def trainee(self) -> Trainable:
        """The trainee assigned to this `Trainer`.
        """
        return self._trainee

    @trainee.setter
    def trainee(self, trainee: Trainable) -> Trainable:
        """Assign a new trainee assigned to this `Trainer`.
        """
        if self.running:
            raise RuntimeError("Cannot exchange trainee during training.")
        if trainee is not self._trainee:
            self._trainee = trainee
            self.notify_observers(self.Change('trainee_changed'))

    @property
    def training_data(self) -> Datasource:
        """The data used for training.
        """
        return self._training_data

    @training_data.setter
    def training_data(self, data: Datasource) -> None:
        if self.running:
            raise RuntimeError("Cannot exchange training data "
                               "during training.")
        if data is not self._training_data:
            self._training_data = data
            self.notify_observers(self.Change('data_changed',
                                              'trainer_changed'))

    @property
    def steps_per_epoch(self) -> Optional[int]:
        """The number of steps per epoch.
        """
        if not isinstance(self._training_data, Sized):
            return None
        return math.ceil(len(self._training_data) / self.batch_size)

    @property
    def epoch(self) -> Optional[int]:
        """The current epoch in the training process.
        """
        if self._start_step is None:
            return None
        steps_per_epoch = self.steps_per_epoch
        if not steps_per_epoch:
            return None
        return (self.step - self._start_step) // steps_per_epoch

    @property
    def epochs(self) -> int:
        """The total number of epochs in the training process.
        """
        if self._end_step is None:
            return None
        steps_per_epoch = self.steps_per_epoch
        if not steps_per_epoch:
            return None
        return (self._end_step - self._start_step) // steps_per_epoch

    @epochs.setter
    def epochs(self, epochs: int) -> None:
        steps_per_epoch = self.steps_per_epoch
        if steps_per_epoch is None:
            return  # ValueError?, would epochs=0 or epochs=1 be acceptable?

        start_step = self._start_step
        if start_step is None:
            start_step = self._step

        end_step = start_step + steps_per_epoch * epochs
        if end_step < self._step:
            raise ValueError(f"Illegal value for epochs: {epochs}")
        self._start_step = start_step
        self._end_step = end_step
        self.notify_observers('trainer_changed')

    @property
    def batch_size(self) -> int:
        """The batch size to apply during training.
        """
        return self._batch_size

    @property
    def batch(self) -> int:
        """The current batch number in the current training epoch.
        """
        steps_per_epoch = self.steps_per_epoch
        step = self.step
        if self._start_step is not None:
            step -= self._start_step
        if not steps_per_epoch:
            return step
        batch = step % steps_per_epoch
        if batch == 0 and step > 0:  # show full batch bar for full epoch
            batch = steps_per_epoch
        return batch

    @property
    def batches(self) -> Optional[int]:
        """The number of batches per epoch.
        """
        return self.steps_per_epoch

    @property
    def step(self) -> int:
        """The step in the current epoch.
        """
        return self._step

    @property
    def ready(self) -> bool:
        """A flag indicating if the `Trainer` is ready for training.
        """
        return self._trainee is not None and self._training_data is not None

    def at_end_of_epoch(self, hook, *args) -> None:
        """Add a function to be called at the end of each epoch.
        """
        self._at_end_of_epoch = (hook, *args)

    def at_end_of_batch(self, hook, *args) -> None:
        """Add a function to be called after each batch.
        """
        self._at_end_of_batch = (hook, *args)

    #
    # loss recording and history
    #

    # FIXME[todo]: this preliminary implementation is a stub. Should
    # be extended to support multiple loss values (e.g. for
    # adversarial autoencoder)

    def loss_history(self, loss: Optional[str] = None) -> np.ndarray:
        """Optain the loss history.
        """
        return self._history

    @busy("Training in progress")
    def train(self, epochs: Optional[int] = None,
              resume: bool = True, restore: bool = True,
              batch_size: Optional[int] = None,
              record: Union[bool, Sequence[str]] = True,
              progress: Optional = None) -> None:
        """Start the training process.

        Arguments
        ---------
        epochs:
            The number of epochs to train.
            If the training data are not `Sized`, this argument is ignored
            and training will continue until training data are exhausted
            or the training is actively interrupted from the outside.

        resume:
            Resume the training.  If `True` and a prior training run was
            interrupted, the training will continue at that point.
            If `False`, the training will start with a new epoch.

        restore:
            Restore the state of the model from a checkpoint if available
            before starting the training run.

        record:
            Record the development of the loss value(s).

        batch_size:
            The batch size to be used for training.

        progress:
            A progress bar to be updated during training.
        """
        if self.running:
            raise RuntimeError("Failed starting process: "
                               "process has already been startet.")
        if not self.ready:
            raise ValueError("Training failed: trainer not ready.")

        # mark that we are running now
        # FIXME[design]: this is somewhat redundant with the BusyOject
        self._begin_run()

        # prepare the trainee for training.
        self._trainee.prepare_training()

        # restoration should be done after preparation, as preparation
        # may add optimizers and other components that may want to
        # restore their state as well
        if restore:
            self._trainee.restore_from_checkpoint()
            # this may also set the _step counter

        batch_size = batch_size or self._batch_size
        if resume and self._start_step is not None:
            # resume an interrupted training run
            if epochs is not None:
                self._end_step = \
                    self._start_step + epochs * self.steps_per_epoch
                self.notify_observers('trainer_changed')
        else:
            # we start a new run:
            steps_per_epoch = self.steps_per_epoch
            if steps_per_epoch is not None:
                if epochs is None:
                    epochs = 1  # default: train for one epoch
                self._start_step = self._step
                self._end_step = self._step + epochs * steps_per_epoch
            self._batch_iterator = self._training_data.batches(size=batch_size)
            self.notify_observers('trainer_changed')

        self.notify_observers('training_started')
        try:
            self._run_training_loop(batch_size, progress)
        finally:
            #self._epoch = None
            #self._step = None
            self._end_run()

            # FIXME[concept]: in case of an error, we should signal
            # that to the observers ...
            self.notify_observers('training_ended')

    def _run_training_loop(self, batch_size, progress) -> None:
        """Run the actual training loop.
        """

        # we may allow that epochs is changed during training, hence
        # we cannot work with range(self.epoch, self.epochs) here.
        if progress is not None:
            status_line = progress.status_printer(sys.stderr)
            progress_bar = progress(initial=self.epoch, total=self.epochs,
                                    position=1, desc='Epoch', leave=False)
        while self.epoch < self.epochs:
            epoch = self.epoch
            self.notify_observers('epoch_started')

            if self._batch_iterator is None:
                self._batch_iterator = \
                    self._training_data.batches(size=batch_size)
            batch_iterator = self._batch_iterator
            if progress is not None:
                batch_iterator = progress(batch_iterator, leave=False,
                                          position=2, desc='Batch')
            for batch in batch_iterator:
                self.notify_observers('batch_started')

                try:
                    self._last_batch_result = \
                        self.trainee.train_batch(batch, epoch=epoch)
                    self._history.append(self._last_batch_result)
                    if progress is not None:
                        loss = self._last_batch_result
                        status_line(f"loss: {loss:.5f}")
                        # progress_bar.set_description(f"epoch: {loss:.5f}")
                        # batch_iterator.set_description(f"batch: {loss:.5f}")
                    if self._at_end_of_batch is not None:
                        self._at_end_of_batch[0](*self._at_end_of_batch[1:])
                except KeyboardInterrupt as ex:
                    self._handle_keyboard_interrupt(ex)
                finally:
                    self._step += 1
                if self.stopping:
                    break
                self.notify_observers('batch_ended')

            if self.stopping:
                break

            self._batch_iterator = None
            self.notify_observers('epoch_ended')
            if self._at_end_of_epoch is not None:
                self._at_end_of_epoch[0](*self._at_end_of_epoch[1:])

            if progress is not None:
                progress_bar.n = self.epoch
                progress_bar.total = self.epochs
                #progress_bar.update(1)
                progress_bar.refresh()

        if progress is not None:
            progress_bar.close()

            # create checkpoints if appropriate:
            # - every n-th epoch
            # - at a given schedule of epochs
            # - if performance is better than previous best performance

        if self.stopping:
            if 'store' in self._keyboard_interrupt_policy:
                if 'message' in self._keyboard_interrupt_policy:
                    print("Storing current state to checkpoint")
                self._trainee.store_checkpoint()
