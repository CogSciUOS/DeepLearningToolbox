"""
File: logging.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
"""
# FIXME[hack]: this is just using a specific keras network as proof of
# concept. It has to be modularized and integrated into the framework

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QPushButton, QSpinBox, QVBoxLayout, QHBoxLayout)

from .panel import Panel
from qtgui.widgets.matplotlib import QMatplotlib

import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf


class AutoencoderPanel(Panel):
    """A panel displaying autoencoders.

    Attributes
    ----------
    _autoencoder: Network
        A network trained as autoencoder.

    _x_train
    _x_test

    _y_train
    _y_test

    """

    def __init__(self, parent=None):
        """Initialization of the LoggingPael.

        Parameters
        ----------
        parent  :   QWidget
                    The parent argument is sent to the QWidget constructor.
        """
        super().__init__(parent)
        self._initDataset()


        # h5 model trained weights
        self._weights_file = 'vae_mlp_mnist.h5'

        # Training parameters
        self._batch_size = 128
        self._epochs = 50

        self._initUI()

    def _initDataset(self):
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


    def _initUI(self):
        """Add the UI elements

            * The ``QLogHandler`` showing the log messages

        """
        #
        # Controls
        #
        self._buttonCreateModel = QPushButton("Create")
        self._buttonTrainModel = QPushButton("Train")
        self._buttonLoadModel = QPushButton("Load")
        self._buttonSaveModel = QPushButton("Save")
        self._buttonPlotModel = QPushButton("Plot Model")
        self._buttonPlotResults = QPushButton("Plot Results")

        self._spinboxEpochs = QSpinBox()
        self._spinboxEpochs.setRange(1, 50)

        #
        # Plots
        #
        self._trainingPlot = QMatplotlib()
        self._resultPlot1 = QMatplotlib()
        self._resultPlot2 = QMatplotlib()

        self._layoutComponents()
        self._connectComponents()

    def _connectComponents(self):
        self._buttonCreateModel.clicked.connect(self._onCreateModel)
        self._buttonTrainModel.clicked.connect(self._onTrainModel)
        self._buttonLoadModel.clicked.connect(self._onLoadModel)
        self._buttonSaveModel.clicked.connect(self._onSaveModel)
        self._buttonPlotModel.clicked.connect(self._onPlotModel)
        self._buttonPlotResults.clicked.connect(self._onPlotResults)

    def _layoutComponents(self):
        """Layout the UI elements.

            * The ``QLogHandler`` displaying the log messages

        """
        plotBar = QHBoxLayout()
        plotBar.addWidget(self._trainingPlot)
        plotBar.addWidget(self._resultPlot1)
        plotBar.addWidget(self._resultPlot2)

        buttonBar = QHBoxLayout()
        buttonBar.addWidget(self._buttonCreateModel)
        buttonBar.addWidget(self._spinboxEpochs)
        buttonBar.addWidget(self._buttonTrainModel)
        buttonBar.addWidget(self._buttonLoadModel)
        buttonBar.addWidget(self._buttonSaveModel)
        buttonBar.addWidget(self._buttonPlotModel)
        buttonBar.addWidget(self._buttonPlotResults)

        layout = QVBoxLayout()
        layout.addLayout(plotBar)
        layout.addLayout(buttonBar)
        self.setLayout(layout)

    def _onLoadModel(self):
        self._autoencoder.load(self._weights_file)
        
    def _onSaveModel(self):
        self._autoencoder.save(self._weights_file)

    def _onCreateModel(self):
        # Initialize the network
        #
        # Network parameters
        #
        original_dim = self._x_train.shape[1]
        intermediate_dim = 512
        latent_dim = 2
        import util
        util.runner.runTask(self._createModel, original_dim)

    def _createModel(self, original_dim):
        #self._graph = tf.Graph()
        #tf_config = tf.ConfigProto()
        #self._session = tf.Session(graph=self._graph, config=tf_config)
        #K.set_session(self._session)
        #with tf.device("/device:GPU:0"):
        #with tf.Session(graph=tf.Graph()) as sess:
        #K.set_session(sess)
        #with tf.device("/device:GPU:0"):
        self._autoencoder = KerasAutoencoder(original_dim)
        #self._autoencoder.train(self._x_train, self._x_test,
        #                        epochs=self._epochs,
        #                        batch_size=self._batch_size)

    def _onTrainModel(self):
        # train the autoencoder
        # FIXME[problem]: we have to run this in the same thread
        # where the autoencoder was initialized!
        # https://stackoverflow.com/questions/42322698/tensorflow-keras-multi-threaded-model-fitting
        import util # FIXME[hack]
        util.runner.runTask(self._trainModel)

    def _trainModel(self):
        #K.set_session(self._session)
        #with tf.device("/device:GPU:0"):
        self._epochs = self._spinboxEpochs.value()
        self._autoencoder.train(self._x_train, self._x_test,
                                epochs=self._epochs,
                                batch_size=self._batch_size)

    def _onPlotModel(self):
        # plot_model requires pydot 
        #plot_model(self._vae, to_file='vae_mlp.png', show_shapes=True)
        pass

    def _onPlotResults(self):
        data = (self._x_test, self._y_test)
        batch_size = self._batch_size
        self.plot_results1(data, batch_size=batch_size)

        # display a 30x30 2D manifold of digits
        n = 30
        digit_size = 28
        self.plot_results2(n, digit_size)

    def plot_results1(self, data, batch_size=128, model_name="vae_mnist"):
        """Plots labels and MNIST digits as function of 2-dim latent vector
        
        Arguments
        ---------
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
        """
        filename = os.path.join(model_name, "vae_mean.png")

        x_test, y_test = data
        os.makedirs(model_name, exist_ok=True)

        # display a 2D plot of the digit classes in the latent space
        z_mean = self._autoencoder.encode(x_test, batch_size)
        plt = self._resultPlot1
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
        #plt.colorbar()
        #plt.xlabel("z[0]")
        #plt.ylabel("z[1]")
        #plt.savefig(filename)
        #plt.show()

    def plot_results2(self, n, digit_size, model_name="vae_mnist"):
        filename = os.path.join(model_name, "digits_over_latent.png")

        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = self._autoencoder.decode(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit

        start_range = digit_size // 2

        end_range = n * digit_size + start_range + 1
        pixel_range = np.arange(start_range, end_range, digit_size)

        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)

        plt = self._resultPlot2
        #plt.xticks(pixel_range, sample_range_x)
        #plt.yticks(pixel_range, sample_range_y)
        #plt.xlabel("z[0]")
        #plt.ylabel("z[1]")
        plt.imshow(figure, cmap='Greys_r')
        #plt.savefig(filename)
        #plt.show()


import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
# plot_model requires that pydot is installed
#from keras.utils import plot_model

from distutils.version import LooseVersion
import keras
assert LooseVersion(keras.__version__) >= LooseVersion("2.2.0"), "Keras version to old, need at least 2.2.x"

from keras import backend as K
print(f"Backend: {K.backend()}")
assert K.backend() == "tensorflow", f"Keras should use the tensorflow backend, not {K.backend()}"

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class KerasAutoencoder:
    """A (variational) autoencoder implemented in Keras.

    Attributes
    ----------
    _vae
    _encoder
    _decoder

    _inputs
    _outputs

    
    _epochs
    _batch_size
    
    _weights_file
    _mse

    """


    def __init__(self, original_dim, intermediate_dim = 512, latent_dim = 2,
                 loss='mse'):
        """Construct a new, fully connected (dense) autoencoder.
        Both, encoder and decoder, will have one itermediate layer
        of the given dimension.
        """
        print(f"New VAE: {original_dim}/{intermediate_dim}/{latent_dim}")

        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())

        # network parameters
        input_shape = (original_dim, )

        #self._session = K.get_session()
        print("HALLO-1")
        self._graph = tf.Graph()
        print("HALLO-2")
        self._session = tf.Session(graph=self._graph)
        print("HALLO-3")
        K.set_session(self._session)
        print("HALLO-4")


        # VAE model = encoder + decoder
        with self._graph.as_default():

            print("HALLO-5")
            #
            # (1) build encoder model
            #
            self._inputs = Input(shape=input_shape, name='encoder_input')
            x = Dense(intermediate_dim, activation='relu')(self._inputs)
            self._z_mean = Dense(latent_dim, name='z_mean')(x)
            self._z_log_var = Dense(latent_dim, name='z_log_var')(x)

            # use reparameterization trick to push the sampling out as input
            # note that "output_shape" isn't necessary with the TensorFlow backend
            z = Lambda(self._sampling, output_shape=(latent_dim,),
                       name='z')([self._z_mean, self._z_log_var])

            # instantiate encoder model
            self._encoder = Model(self._inputs, [self._z_mean, self._z_log_var, z],
                                  name='encoder')
            self._encoder.summary()
            # plot_model requires pydot 
            #plot_model(self._encoder, to_file='vae_mlp_encoder.png',
            #           show_shapes=True)

            #
            # (2) build decoder model
            #
            latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
            x = Dense(intermediate_dim, activation='relu')(latent_inputs)
            self._outputs = Dense(original_dim, activation='sigmoid')(x)

            # instantiate decoder model
            self._decoder = Model(latent_inputs, self._outputs, name='decoder')
            self._decoder.summary()
            # plot_model require pydot installed
            #plot_model(self._decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

            #
            # (3) instantiate VAE model
            #
            self._outputs = self._decoder(self._encoder(self._inputs)[2])
            self._vae = Model(self._inputs, self._outputs, name='vae_mlp')

            if loss == 'mse':
                reconstruction_loss = mse(self._inputs, self._outputs)
            else:
                reconstruction_loss = binary_crossentropy(self._inputs,
                                                          self._outputs)
            # VAE loss = mse_loss or xent_loss + kl_loss
            reconstruction_loss *= original_dim
            kl_loss = (1 + self._z_log_var -
                       K.square(self._z_mean) - K.exp(self._z_log_var))
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            vae_loss = K.mean(reconstruction_loss + kl_loss)
            self._vae.add_loss(vae_loss)
            self._vae.compile(optimizer='adam')
            self._vae.summary()
        
    def __del__(self):
        self._session.close()
        
    # use the reparameterization trick:
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # z = z_mean + sqrt(var)*eps
    def _sampling(self, args):
        """Reparameterization trick by sampling fr an isotropic unit
        Gaussian.
        
        Arguments
        ---------
        args (tensor): mean and log of variance of Q(z|X)

        Returns
        -------
        z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

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

    def train(self, data, validation, epochs, batch_size):
        with self._graph.as_default():
            with self._session.as_default():
                self._vae.fit(data,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_data=(validation, None))

    def encode(self, data, batch_size):
        with self._graph.as_default():
            with self._session.as_default():
                z_mean, _, _ = self._encoder.predict(data,
                                                     batch_size=batch_size)
        return z_mean

    def decode(self, data, batch_size=1):
        with self._graph.as_default():
            with self._session.as_default():
                x_decoded = self._decoder.predict(data,
                                                  batch_size=batch_size)
        return x_decoded


from keras.callback import Callback
from observer import Observable

class KerasProgressCallback(Callback, Observable):
    ATTRIBUTES = ['epoch_changed', 'batch_changed']

    """Callback that prints metrics to stdout.

    Arguments:
      count_mode: One of "steps" or "samples".
          Whether the progress bar should
          count samples seens or steps (batches) seen.

    Raises:
      ValueError: In case of invalid `count_mode`.
    """


    """Inherits the following properties from keras.callbacks.Callback:

      params: dict. Training parameters
          (eg. verbosity, batch size, number of epochs...).
          can be set with  set_params(params):
      model: instance of `keras.models.Model`.
          Reference of the model being trained.

    The `logs` dictionary that callback methods take as argument will
    contain keys for quantities relevant to the current batch or
    epoch.

    Currently, the `.fit()` method of the `Sequential` model class
    will include the following quantities in the `logs` that it passes
    to its callbacks:

      on_epoch_end: logs include `acc` and `loss`, and
          optionally include `val_loss`
          (if validation is enabled in `fit`), and `val_acc`
          (if validation and accuracy monitoring are enabled).
      on_batch_begin: logs include `size`,
          the number of samples in the current batch.
      on_batch_end: logs include `loss`, and optionally `acc`
          (if accuracy monitoring is enabled).

    """

  def __init__(self):
    self.validation_data = None

  def set_params(self, params):
    self.params = params

  def set_model(self, model):
    self.model = model

    """
    
    
    def __init__(self, count_mode='samples'):
        super().__init__()
        if count_mode == 'samples':
            self.use_steps = False
        elif count_mode == 'steps':
            self.use_steps = True
        else:
            raise ValueError('Unknown `count_mode`: ' + str(count_mode))

    @change
    def on_train_begin(self, logs=None):
        self.verbose = self.params['verbose']
        self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose:
            print('Epoch %d/%d' % (epoch + 1, self.epochs))
            if self.use_steps:
                target = self.params['steps']
            else:
                target = self.params['samples']
            self.target = target
            self.progbar = Progbar(target=self.target, verbose=self.verbose)
        self.seen = 0

    def on_batch_begin(self, batch, logs=None):
        if self.seen < self.target:
            self.log_values = []

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        if self.use_steps:
            self.seen += 1
        else:
            self.seen += batch_size

        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        if self.verbose and self.seen < self.target:
            self.progbar.update(self.seen, self.log_values)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
        if self.verbose:
            self.progbar.update(self.seen, self.log_values, force=True)
