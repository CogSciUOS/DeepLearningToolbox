from __future__ import absolute_import
from typing import List

import numpy as np
from collections import OrderedDict
from frozendict import FrozenOrderedDict


import importlib

from . import Network as BaseNetwork


class Network(BaseNetwork):
    """Abstract base class for the keras networks for the specific backends.

    Implements only the functionality that can be efficiently
    implemented in pure Keras.
    """
    
    @classmethod
    def framework_available(cls):
        spec = importlib.util.find_spec("keras")
        return spec is not None

    @classmethod
    def import_framework(cls):
        global keras
        # This unconditionally outputs a message "Using [...] backend."
        # to sys.stderr (in keras/__init__.py).
        keras = importlib.import_module('keras')
        from keras.models import load_model


    def __init__(self, **kwargs):
        """
        Load Keras model.
        Parameters
        ----------
        model_file
            Path to the .h5 model file.
        """
        
        
        # Set learning phase to train in case setting to test would
        # affect gradient computation.
        #
        # FIXME[todo]: Set it to test, to eliminate dropout, check for
        # gradient computation later.
        keras.backend.set_learning_phase(0)
        if 'model_file' in kwargs.keys():
            self._model = keras.models.load_model(kwargs.pop('model_file'))
        elif 'model' in kwargs.keys():
            self._model = kwargs['model']
        # Check the data format.
        kwargs['data_format'] = keras.backend.image_data_format()
        super().__init__(**kwargs)

    def _compute_layer_ids(self):
        """

        Returns
        -------

        """
        # Apparently config can be sometimes a dict or simply a list of `layer_specs`.
        config = self._model.get_config()
        if isinstance(config, dict):
            layer_specs = config['layers']
        elif isinstance(config, list):
            layer_specs = config
        else:
            raise ValueError('Config is neither a dict nor a list.')
        return [layer_spec['config']['name'] for layer_spec in layer_specs]


from keras.callbacks import Callback
from base.observer import TrainingObservable, change
import time

class ObservableCallback(Callback, TrainingObservable,
                         changes=['training_changed', 'epoch_changed',
                                  'batch_changed', 'metric_changed'],
                         default='training_changed',
                         method='trainingChanged'):  # FIXME[hack]: these attributes should be inherited from TrainingObservable!
    """Callback that notifies Observers.

    """
    
    def __init__(self, count_mode='samples'):
        Callback.__init__(self)
        TrainingObservable.__init__(self)
        if count_mode == 'samples':
            self.use_steps = False
        elif count_mode == 'steps':
            self.use_steps = True
        else:
            raise ValueError('Unknown `count_mode`: ' + str(count_mode))

    def on_train_begin(self, logs=None):
        """Called on the beginning of training.
        """
        self._epochs = self.params['epochs']
        self._batch_size = self.params['batch_size']
        self._samples = self.params['samples']
        self._batches = self._samples // self._batch_size
        self._epoch = 0
        self._batch = 0
        self._batch_duration = 0.
        self._running = True
        self._start = time.time()
        self.notifyObservers('training_changed')

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch
        self.notifyObservers('epoch_changed')

    def on_batch_begin(self, batch, logs=None):
        """
        on_batch_begin: logs include `size`,
          the number of samples in the current batch.
        """
        self._batch = batch
        self._batch_size = logs['size']
        self._batch_start = time.time()
        self.notifyObservers('batch_changed')

    def on_batch_end(self, batch, logs=None):
        """
        on_batch_end: logs include `loss`, and optionally `acc`
        """
        self._batch_duration = time.time() - self._batch_start
        self._loss = logs['loss']
        if 'acc' in logs:
            self._accuracy = logs['acc']
        self.notifyObservers('metrics_changed')
    
    def on_epoch_end(self, epoch, logs=None):
        """
        logs include `acc` and `loss`, and
        optionally include `val_loss`
        (if validation is enabled in `fit`), and `val_acc`
        (if validation and accuracy monitoring are enabled).
        """
        self._loss = logs['loss']
        if 'acc' in logs:
            self._accuracy = logs['acc']
        if 'val_loss' in logs:
            self._validation_loss = logs['val_loss']
        if 'val_acc' in logs:
            self._validation_accuracy = logs['val_acc']
        self.notifyObservers('metrics_changed')

    def on_train_end(self, logs=None):
        """Called at the end of training.

        Arguments:
        logs: dictionary of logs.
        """
        self._running = False
        self.notifyObservers('training_changed')

    @property
    def batch_duration(self):
        return self._batch_duration

    @property
    def eta(self):
        now = time.time()
        # current = current step
        # target = last step (-1 = unknown)
        if current: 
            time_per_unit = (now - self._start) / current
        else:
            time_per_unit = 0
        eta = time_per_unit * (self.target - current)
        info = ''
        if current < self.target and self.target is not -1:
            info += ' - ETA: %ds' % eta
        else:
            info += ' - %ds' % (now - self.start)
        return info
