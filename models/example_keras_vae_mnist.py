import os

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import tensorflow as tf

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras

from packaging import version
assert version.parse(keras.__version__) >= version.parse("2.2.0"), \
    "Keras version too old for the autoencoder, need at least 2.2.x"

from keras import backend as K
logger.info(f"Backend: {K.backend()}")
assert K.backend() == "tensorflow", f"Keras should use the tensorflow backend, not {K.backend()}"

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy
# plot_model requires that pydot is installed
#from keras.utils import plot_model

from toolbox import toolbox

class Autoencoder:

    def __init__(self):
        self._encoder = None
        self._decoder =None

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

class VariationalAutoencoder:

    def sampleCode(self, n=1):
        pass

    def sampleData(self, n=1):
        pass

    def sampleCodeFor(self, input, n=1):
        pass

    def sampleDataFor(self, input, n=1):
        pass


from network.keras import KerasModel

class KerasAutoencoder(Autoencoder, KerasModel):
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
        logger.info(f"New VAE: {original_dim}/{intermediate_dim}/{latent_dim}")
        Autoencoder.__init__(self)
        KerasModel.__init__(self)

        # network parameters
        input_shape = (original_dim, )

        # VAE model = encoder + decoder
        with self._graph.as_default():

            #
            # (1) build encoder model
            #
            self._inputs = Input(shape=input_shape, name='encoder_input')
            x = Dense(intermediate_dim, activation='relu')(self._inputs)
            self._z_mean = Dense(latent_dim, name='z_mean')(x)
            self._z_log_var = Dense(latent_dim, name='z_log_var')(x)

            # Use reparameterization trick to push the sampling out as
            # input (note that "output_shape" isn't necessary with the
            # TensorFlow backend)
            self._z = Lambda(self._sampling, output_shape=(latent_dim,),
                             name='z')([self._z_mean, self._z_log_var])

            # instantiate encoder model. It provides two outputs:
            #  - (z_mean, z_log_var): a pair describing the mean and (log)
            #    variance of the code variable z (for input x)
            #  - z: a value sampled from that distribution
            self._encoder = Model(self._inputs,
                                  [self._z_mean, self._z_log_var, self._z],
                                  name='encoder')
            self._encoder.summary(print_fn=self._print_fn)
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
            self._decoder.summary(print_fn=self._print_fn)
            # plot_model require pydot installed
            #plot_model(self._decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

            #
            # (3) define the loss function
            #
            self._outputs = self._decoder(self._encoder(self._inputs)[2])
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

            #
            # (4) instantiate VAE model
            #
            self._vae = Model(self._inputs, self._outputs, name='vae_mlp')
            self._vae.add_loss(vae_loss)
            self._vae.compile(optimizer='adam')
            self._vae.summary(print_fn=self._print_fn)


    def _print_fn(self, line):
        logger.info(line)

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

    def train(self, data, validation, epochs, batch_size, progress):
        toolbox.acquire()
        with self._graph.as_default():
            with self._session.as_default():
                self._vae.fit(data,
                              epochs=epochs,
                              verbose=0,
                              batch_size=batch_size,
                              validation_data=(validation, None),
                              callbacks=[progress])
        toolbox.release()

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


    def reconstruct(self, data, batch_size):
        with self._graph.as_default():
            with self._session.as_default():
                reconstruction = self._vae.predict(data, batch_size=batch_size)
        return reconstruction
        

    def sample_code(self, input=None, params=None, n=1):
        with self._graph.as_default():
            with self._session.as_default():
                feed_dict = {}
                if params is not None:
                    z_mean = params['z_mean']
                    if not instanceof(z_mean, np.ndarray):
                        z_mean =  np.full(n, z_mean)
                    feed_dict[self._z_mean] = z_mean
                    z_log_var = params['z_log_var']
                    if not instanceof(z_log_var, np.ndarray):
                        z_log_var =  np.full(n, z_log_var)
                    feed_dict[self._z_mean] = z_log_var
                    z = self._z.eval(feed_dict=feed_dict)
                elif input is not None:
                    _, _, z= self._encoder.predict(input)
        return z
