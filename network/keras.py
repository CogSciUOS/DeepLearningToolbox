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

# FIXME[hack]:
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from tools.train import Training as BaseTraining
from keras.callbacks import Callback
from keras.datasets import mnist

import time

class Training(BaseTraining, Callback):
    """

    This keras Training class implements a keras Callback, allowing
    to get some information on the training process.

    """
    
    def __init__(self, count_mode='samples'):
        BaseTraining.__init__(self)
        Callback.__init__(self)
        if count_mode == 'samples':
            self.use_steps = False
        elif count_mode == 'steps':
            self.use_steps = True
        else:
            raise ValueError('Unknown `count_mode`: ' + str(count_mode))

    def start(self):
        # This will start the keras training loop.
        # As a result, the on_training_begin callback will be called,
        # which will also notify the observers.
        self._model.train(self._x_train, self._x_test,
                          epochs=self._epochs,
                          batch_size=self._batch_size,
                          progress=self)

    def stop(self):
        print("Stopping Training")
        print(self.model == self._model)
        # This will cause the keras training loop to stop.
        # As a result, the on_training_end callback will be called,
        # which will also notify the observers.
        self.model.stop_training = True

    def old_disfunct_star_stop(self):
        if self._autoencoder:
            model = self._autoencoder._vae
            print(f"onTrainModel-1: {type(model)}")
            if hasattr(model, 'callback_model') and model.callback_model:
                model = model.callback_model
            print("onTrainModel-2: {type(model)}")
            print("onTrainModel-3: {hasattr(model, 'stop_training')}")
            if not hasattr(model, 'stop_training') or not model.stop_training:
                import util # FIXME[hack]
                print("onTrainModel-4a: start training")
            else:
                print("onTrainModel-4b: stop training")
                model.stop_training = True

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
        self._start = time.time()
        super().start()

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
        super().stop()

    @property
    def batch_duration(self):
        return self._batch_duration

    @property
    def eta(self):
        """Estimated time to arival (ETA).
        """
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


    def set_data(self, x_train, x_test, y_train, y_test):
        self.notifyObservers('data_changed')

    def hack_load_mnist(self):

        """Initialize the dataset.
        This will set the self._x_train, self._y_train, self._x_test, and
        self._y_test variables. Although the actual autoencoder only
        requires the x values, for visualization the y values (labels)
        may be interesting as well.

        The data will be flattened (stored in 1D arrays), converted to
        float32 and scaled to the range 0 to 1. 
        """
        #
        # The dataset
        #
        
        # load the MNIST dataset
        x, y = mnist.load_data()
        (self._x_train, self._y_train) = x
        (self._x_test, self._y_test) = y

        input_shape = self._x_train.shape[1:]
        original_dim = input_shape[0] * input_shape[1]
        self._x_train = np.reshape(self._x_train, [-1, original_dim])
        self._x_test = np.reshape(self._x_test, [-1, original_dim])
        self._x_train = self._x_train.astype('float32') / 255
        self._x_test = self._x_test.astype('float32') / 255


from packaging import version
import keras
if version.parse(keras.__version__) >= version.parse('2.0.0'):
    from keras.layers import Conv2D
else:
    from keras.layers import Convolution2D

def conv_2d(filters, kernel_shape, strides, padding, input_shape=None):
    """
    Defines the right convolutional layer according to the
    version of Keras that is installed.
    :param filters: (required integer) the dimensionality of the output
                  space (i.e. the number output of filters in the
                  convolution)
    :param kernel_shape: (required tuple or list of 2 integers) specifies
                       the strides of the convolution along the width and
                       height.
    :param padding: (required string) can be either 'valid' (no padding around
                  input or feature map) or 'same' (pad to ensure that the
                  output feature map size is identical to the layer input)
    :param input_shape: (optional) give input shape if this is the first
                      layer of the model
    :return: the Keras layer
    """
    if version.parse(keras.__version__) >= version.parse('2.0.0'):
        if input_shape is not None:
            return Conv2D(filters=filters, kernel_size=kernel_shape,
                          strides=strides, padding=padding,
                          input_shape=input_shape)
        else:
            return Conv2D(filters=filters, kernel_size=kernel_shape,
                          strides=strides, padding=padding)
    else:
        if input_shape is not None:
            return Convolution2D(filters, kernel_shape[0], kernel_shape[1],
                                 subsample=strides, border_mode=padding,
                                 input_shape=input_shape)
        else:
            return Convolution2D(filters, kernel_shape[0], kernel_shape[1],
                                 subsample=strides, border_mode=padding)


# plot_model requires that pydot is installed
#from keras.utils import plot_model
import tensorflow as tf
from keras import backend as K

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class KerasModel:  # FIXME[concept]: this is actually a Keras Tensorflow(!) Model

    def __init__(self):
        from tensorflow.python.client import device_lib
        logger.info(device_lib.list_local_devices())

        #self._session = K.get_session()
        self._graph = tf.Graph()
        self._session = tf.Session(graph=self._graph)
        K.set_session(self._session)
        
        self._model = None
        self._snapshot = None

    def __del__(self):
        self._session.close()
   
    def load(self, filename: str):
        with self._graph.as_default():
            with self._session.as_default():
                self._vae.load_weights(filename)
                logger.info(f'loaded autoencoder from "{filename}"')

    def save(self, filename: str):
        with self._graph.as_default():
            with self._session.as_default():
                self._vae.save_weights(filename)
                logger.info(f'saved autoencoder to "{filename}"')

    def snapshot(self):
        """Make a snapshot of the current model state (weights).
        """
        self._snapshot = self._model.get_weights()

    def reset(self):
        """Reset the model parameters (weights) to the last snapshot.
        """
        if self._snapshot is not None:
            self._keras_model.set_weights(self._snapshot)
            self._sess.run(tf.global_variables_initializer())

    @property
    def graph(self):
        return self._graph

    @property
    def session(self):
        return self._session

    @property
    def model(self):
        return self._model

class KerasClassifier(KerasModel):  # FIXME[concept]: this is actually a Keras Tensorflow(!) Classifier

    def __init__(self):
        super().__init__()
        self._input_placeholder = None
        self._label_placeholder = None
        self._predictions = None

    @property
    def input(self):
        return self._input_placeholder

    @property
    def label(self):
        return self._label_placeholder

    @property
    def predictions(self):
        return self._predictions

