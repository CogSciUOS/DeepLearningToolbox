"""Tensorflow implementation of adversarial autoencoders.

Originally based on [1] with significant changers.



Demo (command line)
===================

conda activate tf114
python -m dltb.thirdparty.tensorflow.aae



Demo (interactive)
==================

# import tensorflow as tf
import tensorflow.compat.v1 as tf
from ulf import SupervisedAdversarialAutoencoder
aae = SupervisedAdversarialAutoencoder(code_dim=2)
aae.prepare()

saver = tf.train.Saver(name='aae', filename="my_test_model")
sess = tf.InteractiveSession()
aae.set_tensorflow_session(sess)
start_epoch = aae._tf_initialize_variables(sess, saver)

aae.plot_data_codes_2d(test_xs, labels=test_ys)

aae.plot_recoded_images(test_xs[:100], labels=test_ys[:100])

aae.plot_decoded_codespace_2d(labels=3)

aae.plot_analogical_decoding()

sess.close()


Data
====
# data are numpy.ndarray.
# xs: image pixel data
#     dtype=float64, min/max=0.0/1.0,
#     shape=(batch, height, width, channels)
# ys: one-hot encoded class labels
#     dtype=float64, min/max=0.0/1.0
#     shape=(batch, classes)

      sellf._data_semi_pipeline = data_pipeline(self._conf.data)
      self._valid_xs = self._valid_xs[:self._conf.num_samples]
      self._valid_ys = self._valid_ys[:self._conf.num_samples]


References
==========
[1] https://github.com/MINGUKKANG/Adversarial-AutoEncoder.git
"""

# pylint: disable=too-many-lines
# pylint: disable=fixme
# pylint: disable=unexpected-keyword-arg,no-value-for-parameter


# standard imports
from typing import Tuple, Optional, Sized, Union
import time
import datetime

# third-party imports
import numpy as np
from tqdm import tqdm

# toolbox imports
from dltb.datasource import Datasource
from dltb.base.data import add_noise
from dltb.util.plot import TilingPlotter
from dltb.util.distributions import gaussian, gaussian_mixture, swiss_roll
from . import v1 as tf
from .v1 import Utils as tf_helper
from .ae import Autoencoder


class AdversarialAutoencoder(Autoencoder):
    """Base class for the TensorFlow Adversarial Autoencoder (AAE)
    implementation.  The AAE extends the basic Autoencoder (AE) by
    introducing an additional training objective to force the encoder
    to produces code that is distributed according to a given target
    distribution ("prior").

    The method :py:class:`_sample_prior` allows to samples codes
    from the target distribution.  There are three different target
    distributions implemented by this class: `'gaussian'`,
    `'gaussian_mixture'`, and `'swiss_roll'`.

    Training
    --------
    Training the AAE consists of two parts: on the one hand, the full
    autoencoder stack (encoder + decoder) is trained to minimize the
    reconstruction error.  On the other hand, the encoder part is
    trained to fit the code distribution to a given target distribution
    ("prior").  This training employs an adversarial training procedure
    in which the encoder acts as generator that tries to mimic the
    target distribution.

    The autoencoder training aims to minimize the reconstruction loss,
    which is accessible via the property `_tf_loss_reconstruction`.  It is
    interpreted as the "negative log likelihood".  Computing this loss
    only requires the input data values.

    The adversarial training process consist of two parts: on the one
    hand it aims to improve the discriminator in its ability to
    discriminate codes output by the encode from codes sampled from
    the target distribution. On the other hand it wants the encoder to
    improve to create codes that follow the target distribution.

    Arguments
    ---------
    data:
        The dataset to use (either 'MNIST' or '')

    """

    def __init__(self, data: str = 'MNIST',
                 prior: str = 'gaussian', **kwargs) -> None:
        super().__init__(**kwargs)
        self._conf.data = data
        self._conf.prior = prior

        # tensorflow placeholders
        self._tf_prior = None

        # The model
        self._tf_loss_discriminator = None
        self._tf_loss_generator = None

        # Optimizers
        self._tf_optimize_discriminator = None
        self._tf_optimize_generator = None

        # loss values (per batch)
        self._loss_discriminator_value = None
        self._loss_generator_value = None

    def _prepare_tensors(self) -> None:
        """
        """
        if self._tf_prior is None:
            self._tf_prior = \
                tf.placeholder(tf.float32, shape=[None, self.code_dim],
                               name="z_prior")

        super()._prepare_tensors()

    @staticmethod
    def _sample_prior(prior: str, zdim: int, nclasses: int, batch_size: int,
                      use_label_info: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Results
        -------
        prior_style:
        prior_label_onehot:
            A 2-dimensional numpy.ndarray of type float holding `prior_label`
            in one-hot-encoding. The shape is (batch_size, nclasses).
        """
        if prior == 'gaussian':
            prior_style, prior_label = \
                gaussian(batch_size, labels=nclasses)

        elif prior == 'gaussian_mixture':
            prior_style, prior_label = \
                gaussian_mixture(batch_size, components=nclasses, labels=True)

        elif prior == 'swiss_roll':
            prior_style, prior_label = \
                swiss_roll(batch_size, labels=nclasses)

        else:
            raise ValueError(f"Unknow prior: '{prior}'. Known values are: "
                             "gaussian, gaussian_mixture, swiss_roll")

        prior_label_onehot = np.eye(nclasses, dtype=np.float32)[prior_label]
        return prior_style, prior_label_onehot

    def _tf_define_optimizers(self, variables) -> None:
        """Prepare the training process. Define optimizers.
        """
        super()._tf_define_optimizers(variables)

        var_generator = [var for var in variables
                         if "encoder" in var.name]
        var_discriminator = [var for var in variables
                             if "discriminator" in var.name]

        self._tf_optimize_discriminator = \
            tf.train.AdamOptimizer(learning_rate=self._tf_learning_rate/5).\
            minimize(self._tf_loss_discriminator,
                     global_step=self._tf_global_step,
                     var_list=var_discriminator)
        self._tf_optimize_generator = \
            tf.train.AdamOptimizer(learning_rate=self._tf_learning_rate).\
            minimize(self._tf_loss_generator,
                     global_step=self._tf_global_step,
                     var_list=var_generator)

    def _tf_train_step(self, sess, feed_dict) -> None:
        """Perform one training step.
        """
        super()._tf_train_step(sess, feed_dict)
        # Optmize the reconstruction loss. This requires
        # 'inputs' and 'outputs' to be given in the 'feed_dict'
        # _negative_log_likelihood_value, _, _g = \
        self._loss_reconstruction_value, _, _g = \
            sess.run([self._tf_loss_reconstruction,
                      self._tf_optimize_reconstruction,
                      self._tf_global_step],
                     feed_dict=feed_dict)

        # Discriminator phase
        self._loss_discriminator_value, _ = \
            sess.run([self._tf_loss_discriminator,
                      self._tf_optimize_discriminator],
                     feed_dict=feed_dict)

        # Generator phase
        self._loss_generator_value, _ = \
            sess.run([self._tf_loss_generator,
                      self._tf_optimize_generator],
                     feed_dict=feed_dict)


class LabeledAutoencoder(Autoencoder):
    """A labeled autoencoder splits the code interpretation into two
    parts: one part representing the class label (as one-hot encoded
    vector) while the other part should hold other information
    (sometimes referred to as "style").

    A labeled autoencoder provides methods to compute and access the
    two code parts (style and label) independently.

    In a `LabeledAutoencoder`, the property :py:prop:`code_dim` refers
    to the dimensionality of the style part of the code, while the
    property :py:prop:`nclasses` contains the number of classes.
    That is, the combined code vector (style + label) has dimensionality
    `code_dim + nclasses`.
    """

    def __init__(self, nclasses: Optional[int] = None, **kwargs) -> None:
        super().__init__(**kwargs)

        # properties
        self._nclasses = nclasses

        # tensorflow placeholders
        self._tf_labels = None

    @property
    def nclasses(self) -> int:
        """The number of classes, that is the number of different labels,
        in this :py:class:`LabeledAutoencoder`.
        """
        return self._nclasses

    def _prepare_tensors(self) -> None:
        """
        """
        self._tf_labels = \
            tf.placeholder(dtype=tf.float32, shape=[None, self.nclasses],
                           name="Input_labels")

        super()._prepare_tensors()

    def one_hot(self, labels: np.ndarray,
                length: Optional[int] = None) -> np.ndarray:
        """Get the given labels in one-hot encoding.
        """
        if isinstance(labels, Sized):
            if length is not None and len(labels) != length:
                raise ValueError(f"Labels have length {len(labels)} but "
                                 f"should have lenght {length}.")
            if isinstance(labels, np.ndarray):
                if labels.ndim == 2:
                    return labels
                if labels.ndim == 1:
                    return np.eye(self.nclasses, dtype=np.float32)[labels]
            elif isinstance(labels, list):
                return np.eye(self.nclasses, dtype=np.float32)[labels]
        elif isinstance(labels, int):
            if length is None:
                raise ValueError("You need to specify a length for "
                                 "the one-hot vector")
            return np.eye(self.nclasses, dtype=np.float32)[[labels] * length]
        raise TypeError("Unexpected type {type(labels)} for labels.")


class LabeledAdversarialAutoencoder(LabeledAutoencoder,
                                    AdversarialAutoencoder):
    """A labeled adversarial autoencoder combines the ideas of the labeled
    autoencoder and the adversarial autoencoder.  It aims at forcing
    both, the distribution of the style part of code and the
    distribution of the label part of the code towards two given
    priors using adversarial training techniques.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # tensorflow placeholders
        self._tf_prior_style = None
        self._tf_prior_label = None

    def _prepare_tensors(self) -> None:
        """
        """
        self._tf_prior_style = \
            tf.placeholder(tf.float32, shape=[None, self.code_dim],
                           name="z_prior")
        self._tf_prior_label = \
            tf.placeholder(tf.float32, shape=[None, self.nclasses],
                           name="prior_labels")

        self._tf_prior = \
            tf.concat([self._tf_prior_style, self._tf_prior_label], axis=1)

        super()._prepare_tensors()

    def plot_analogical_decoding(self, style: Union[np.ndarray, int] = 10,
                                 plotter: Optional[TilingPlotter] = None,
                                 **kwargs) -> None:
        """Plot the analogical reasoning results. For a given style,
        generate analogical exemplas for each of the classes.

        Arguments
        ---------
        style:
            The styles to be used for generation. If an `int`
            this will specify the number of random styles to
            to be choosen.
        """
        # generate random code style samples (styles)
        if isinstance(style, int):  # number of examples
            style = np.random.rand(style, self.code_dim)
        examples = len(style)
        style = np.repeat(style, self.nclasses, axis=0)

        labels = np.arange(self.nclasses)
        labels = np.tile(labels, examples)
        one_hot = self.one_hot(labels)

        codes = np.concatenate((style, one_hot), axis=1)

        # use decoder to generate output data from the combined code.
        data = self.decode(codes)

        # plot the results
        if plotter is None:
            plotter = TilingPlotter()
        plotter.plot_tiling(data, rows=examples, columns=self.nclasses,
                            **kwargs)


class SupervisedAdversarialAutoencoder(LabeledAdversarialAutoencoder):
    """Specialized sublasse for a supervised Adversarial Autoencoder (AAE)
    implementation.

    Notice that the :py:class:`SupervisedAdversarialAutoencoder` is
    not really an autoencoder, as the code does not contain sufficient
    information to reconstruct the input.  Instead the decoder needs
    additional label information.

    The encoder encodes a given input just in the "style" part.  This
    style code has to be combined with the label information to obtain
    the full code.

    The encoder uses `tf_inputs` to compute the style code which is
    provided in `tf_encoded_style`.  This can be combined with
    information from `tf_labels` to obtain the full code vector in
    `tf_encoded`. The decoder uses `tf_encoded` to reconstruct the
    data and provides it in `tf_decoded`.

    This specific behavior results in a slight redesign of the
    autoencoder interface: the methods :py:meth:`encode`,
    :py:meth:`decode`, and :py:meth:`recode` accept an additional
    argument `labels` that allows to provide the required class
    labels.

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._conf.model = 'supervised'

        self._tf_encoded_style = None

    def encode(self, data: np.ndarray, batch_size: int = 128,
               labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Encode data using the encoder part of the autoencoder.

        Arguments
        ---------
        data:
            The data to be encoded.

        Result
        ------
        code:
            The codes obtained from the data. If no label information
            is available, this will only be the style part of the code.
        """
        length = len(data)
        code = np.ndarray((length, self.code_dim))

        offset = 0
        batches = self._np_batcher(data, batch_size=batch_size)
        code_batches = \
            self._tf_encode_batches(batches, tf_encoded=self._tf_encoded_style)
        for batch in code_batches:
            end = offset + len(batch)
            code[offset: end] = batch
            offset = end

        if labels is None:
            return code

        labels_one_hot = self.one_hot(labels, length=length)
        return np.concatenate((code, labels_one_hot), axis=1)

    def decode(self, code: np.ndarray, batch_size: int = 128,
               labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Decode given code values into the data space using the decoder
        part of the autoencoder.

        Arguments
        ---------
        code:
            The codes to be decoded.

        Result
        ------
        data:
            The reconstructed data.
        """
        if code.shape[1] == self.code_dim:
            if labels is None:
                raise ValueError("The supervised autoencoder needs explicit "
                                 "label information for decoding.")
            labels_one_hot = self.one_hot(labels, length=len(code))
            code = np.concatenate((code, labels_one_hot), axis=1)
        return super().decode(code, batch_size=batch_size)

    def recode(self, data: np.ndarray, batch_size: int = 128,
               labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Reconstruct data values using the autoencoder, that is first
        encode the data and the decode it back into the data space.

        Arguments
        ---------
        data:
            The data to be recoded.

        Result
        ------
        recoded:
            The reconstructed data.
        """
        # FIXME[hack]: just encode + decode. A full tensorflow
        # recoding would probably be more efficient
        code = self.encode(data, batch_size=batch_size, labels=labels)
        return self.decode(code, batch_size=batch_size)

    def _prepare_tensors(self) -> None:
        """Setup TensorFlow properties for the supervised adversarial
        autoencoder.

        Prepare the tensorflow network (computational graph) realizing
        the supervised adversarial autoencoder.

        The constructed graph looks as follows:

                 [inputs]           [labels]
                    |                  |
               inputs_flat             |
                    |                  |
          (encoder) |                  |
                    V                  |
             *encoded_style*           |      [prior_style]  [prior_label]
                    |                  |             |            |
                    +------------------+             +------+-----+
                             |                              |
                             V                              V
                         *encoded*                        prior
                             |                              |
                    +----------------+                      |
          (decoder) |                |                      |
                    |                V                      V
                    V             D_fake_logits        D_real_logits
                *decoded*          |         |              |
                    |              V         V              V
                    +-----+     G_loss   D_loss_fake   D_loss_true
                          |        |           |              |
            [outputs]     |        V           +-------+------+
                |         |   *loss_generator*         V
            outputs_flat  |                   *loss_discriminator*
                V         V
             *loss_reconstruction*

        [...]: input values (tf.placeholder)
                 inputs
                 outputs
                 code
                 prior_style
                 prior_label
        *...*: output values (tf.tensor)
                 encoded: encoder output (just style, no label)
                 decoded: decoder output (flat data)

        Tensorflow properties are prefixed with '_tf_'.
        """
        # Super class sets up a standard autoencoder from placeholders
        # 'inputs', 'outputs', providing 'encoded', 'decoded' and
        # 'loss_reconstruction'.
        super()._prepare_tensors()

        #
        # The discriminators
        #
        prior = \
            tf.concat([self._tf_prior_style, self._tf_prior_label], axis=1)
        discriminator_real_logits = \
            self._tf_discriminator(prior, self._tf_keep_prob)
        discriminator_fake_logits = \
            self._tf_discriminator(self._tf_encoded, self._tf_keep_prob)

        discriminator_fake_labels = tf.zeros_like(discriminator_fake_logits)
        discriminator_loss_fake = tf.nn.\
            sigmoid_cross_entropy_with_logits(logits=discriminator_fake_logits,
                                              labels=discriminator_fake_labels)
        discriminator_real_labels = tf.ones_like(discriminator_real_logits)
        discriminator_loss_true = tf.nn.\
            sigmoid_cross_entropy_with_logits(logits=discriminator_real_logits,
                                              labels=discriminator_real_labels)

        generator_fake_labels = tf.ones_like(discriminator_fake_logits)
        generator_loss = tf.nn.\
            sigmoid_cross_entropy_with_logits(logits=discriminator_fake_logits,
                                              labels=generator_fake_labels)

        self._tf_loss_discriminator = \
            tf.reduce_mean(discriminator_loss_fake) + \
            tf.reduce_mean(discriminator_loss_true)
        self._tf_loss_generator = tf.reduce_mean(generator_loss)

    def _tf_encoder(self, data, keep_prob):
        """Encoder for an autoencoder.
        """
        self._tf_encoded_style = super()._tf_encoder(data, keep_prob)
        return tf.concat([self._tf_encoded_style, self._tf_labels], axis=1)

    def train(self,
              labeled_train_data: Datasource,
              labeled_display_data: Datasource,
              sess: tf.Session, saver) -> None:
        """Train the network.
        """
        # display_data: (images, images_noised, labels)
        # A batch of data used for plotting intermediate results
        # during training (taken from the validation dataset)
        # Each is a numpy array of length 100 and appropriate shape.
        display_data = labeled_display_data[:100, ('array', 'array', 'label')]

        #
        # prepare the optimizers
        #

        # obtain the variables used in the model
        total_vars = tf.trainable_variables()
        # var_ae = [var for var in total_vars
        #           if "encoder" in var.name or "decoder" in var.name]

        # Optimizers
        self._tf_define_optimizers(total_vars)

        #
        # Start the training
        #
        start_epoch = self._tf_initialize_variables(sess, saver)
        start_time = time.time()

        total_batch = len(labeled_train_data) // self._conf.batch_size
        labeled_batch_iterator = \
            labeled_train_data(batch_size=self._conf.batch_size, loop=True,
                               attributes=('array', 'noisy', 'label'))

        for epoch in tqdm(range(start_epoch, self._conf.n_epoch),
                          initial=start_epoch, total=self._conf.n_epoch):
            likelihood = 0
            discriminator_value = 0
            generator_value = 0

            # Adapt the learning rate depending on the epoch
            lr_value = self._learning_rate_schedule(epoch)

            for _batch_idx in tqdm(range(total_batch)):
                batch_xs, batch_noised_xs, batch_ys = \
                    next(labeled_batch_iterator)

                # Sample from the prior distribution
                prior_style, prior_label_onehot = \
                    self._sample_prior(self._conf.prior, zdim=self.code_dim,
                                       nclasses=self.nclasses,
                                       batch_size=self._conf.batch_size,
                                       use_label_info=True)

                feed_dict = {
                    self._tf_inputs: batch_noised_xs,
                    self._tf_outputs: batch_xs,
                    self._tf_labels: batch_ys,
                    self._tf_prior_style: prior_style,
                    self._tf_prior_label: prior_label_onehot,
                    self._tf_learning_rate: lr_value,
                    self._tf_keep_prob: self._conf.keep_prob
                }

                # AutoEncoder phase
                self._tf_train_step(sess, feed_dict)

                # Summary
                likelihood += \
                    self._loss_reconstruction_value/total_batch
                discriminator_value += \
                    self._loss_discriminator_value/total_batch
                generator_value += \
                    self._loss_generator_value/total_batch

            # every 5th epoch (except the last) plot the manifold canvas
            if epoch % 5 == 0 or epoch == (self._conf.n_epoch - 1):
                name = f"Manifold_canvas_{epoch}"
                self.plot_recoded_images(display_data[1],
                                         targets=display_data[0],
                                         labels=display_data[2],
                                         filename=name)

            # output end of epoch information
            runtime = time.time() - start_time
            print(f"Epoch: {epoch:3d}, "
                  f"global step: {sess.run(self._tf_global_step)}, "
                  f"Time: {datetime.timedelta(seconds=runtime)}")
            print(f"             lr_AE: {lr_value:.5f}"
                  f"   loss_AE: {likelihood:.4f}   ")
            print(f"             lr_D: {lr_value/5:.5f}"
                  f"   loss_D: {discriminator_value:.4f}")
            print(f"             lr_G: {lr_value:.5f}"
                  f"   loss_G: {generator_value:.4f}\n")

            if saver is not None:
                saver.save(sess, 'checkpoints/my_test_model',
                           global_step=self._tf_global_step,
                           write_meta_graph=False)
                print(f"Saver: {saver.last_checkpoints}")


class SemisupervisedAdversarialAutoencoder(LabeledAdversarialAutoencoder):
    """Specialized sublasse for a semi-supervised
    Adversarial Autoencoder (AAE) implementation.

    The main differences to the fully supervised AAE are
    the following:
    * the encoder (generator) also outputs class labels. That is the latent
      representation is split into two parts: the continuous z and the one-hot
      encoded label information y.
    * there now are two discriminators, one for each part of the latent
      representation, and two loss functions training them:
      `_loss_discriminator_style` and `_loss_discriminator_label`.
      Training the z part requires z value output from the encoder/generator
      ()
      as well as real z values sampled from the target distribution.
      Training the y part requires the label output from the encoder/generator

    * the produced class labels can be used as additional training objective
      to train the encoder (generator) to minimize crossentropy loss, if
      ground truth label are available (supervised case).  This loss function
      is stored under the name `_crossentropy_labels`.  The training process
      requires input data with real class labels.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._conf.model = 'semi_supervised'

        #
        # data related properties
        #

        # model related properties
        self._style = None
        self._crossentropy_labels = None

        # loss functions
        self._tf_loss_discriminator_label = None
        self._tf_loss_discriminator_style = None

        # optimizers
        self._op_z_discriminator = None
        self._op_y_discriminator = None
        self._op_generator = None
        self._op_crossentropy_labels = None

        self._l_z_discriminator = None
        self._l_y_discriminator = None
        self._l_generator = None
        self._crossentropy = None

    def _prepare_tensors(self) -> None:
        """Setup TensorFlow properties for the supervised adversarial
        autoencoder.

        """
        super()._prepare_tensors()

        # placeholders
        # FIXME[semi]: self._z_cat
        # Y_cat = tf.placeholder(dtype=tf.float32,
        #                        shape=[None, n_cls], name="labels_cat")
        # labels_cat = self._tf_prior_label

        # FIXME[coding]: code duplication
        flat_data_length = \
            self._data_shape[0] * self._data_shape[1] * self._data_shape[2]
        inputs_flat = tf.reshape(self._tf_inputs, [-1, flat_data_length])
        outputs_flat = tf.reshape(self._tf_outputs, [-1, flat_data_length])

        # the encoder
        self._style, labels_softmax = \
            self._tf_semi_encoder(inputs_flat, self._tf_keep_prob,
                                  semi_supervised=False)
        _, labels_generated = \
            self._tf_semi_encoder(inputs_flat, self._tf_keep_prob,
                                  semi_supervised=True)
        latent_inputs = tf.concat([self._style, labels_softmax], axis=1)

        # the decoder
        self._tf_decoded = \
            self._tf_semi_decoder(latent_inputs, self._tf_keep_prob)

        #
        # the discriminators
        #

        discriminator_label_fake = \
            self._tf_semi_y_discriminator(labels_softmax,
                                          self._tf_keep_prob)
        discriminator_label_real = \
            self._tf_semi_y_discriminator(self._tf_prior_label,
                                          self._tf_keep_prob)

        discriminator_style_fake = \
            self._tf_semi_z_discriminator(self._style,
                                          self._tf_keep_prob)
        discriminator_style_real = \
            self._tf_semi_z_discriminator(self._tf_prior_style,
                                          self._tf_keep_prob)

        #
        # loss functions
        #
        self._tf_loss_reconstruction = \
            tf.reduce_mean(tf.squared_difference(self._tf_decoded,
                                                 outputs_flat))

        discriminator_label_zeros = tf.zeros_like(discriminator_label_fake)
        discriminator_label_ones = tf.ones_like(discriminator_label_real)
        discriminator_loss_label_real = tf.nn.\
            sigmoid_cross_entropy_with_logits(logits=discriminator_label_real,
                                              labels=discriminator_label_ones)
        discriminator_loss_label_fake = tf.nn.\
            sigmoid_cross_entropy_with_logits(logits=discriminator_label_fake,
                                              labels=discriminator_label_zeros)
        self._tf_loss_discriminator_label = \
            tf.reduce_mean(discriminator_loss_label_real) + \
            tf.reduce_mean(discriminator_loss_label_fake)

        discriminator_style_zeros = tf.zeros_like(discriminator_style_fake)
        discriminator_style_ones = tf.ones_like(discriminator_style_real)
        discriminator_loss_style_real = tf.nn.\
            sigmoid_cross_entropy_with_logits(logits=discriminator_style_real,
                                              labels=discriminator_style_ones)
        discriminator_loss_style_fake = tf.nn.\
            sigmoid_cross_entropy_with_logits(logits=discriminator_style_fake,
                                              labels=discriminator_style_zeros)
        self._tf_loss_discriminator_style = \
            tf.reduce_mean(discriminator_loss_style_real) + \
            tf.reduce_mean(discriminator_loss_style_fake)

        loss_generator_label = tf.nn.\
            sigmoid_cross_entropy_with_logits(logits=discriminator_label_fake,
                                              labels=discriminator_label_ones)
        loss_generator_style = tf.nn.\
            sigmoid_cross_entropy_with_logits(logits=discriminator_style_fake,
                                              labels=discriminator_style_ones)
        self._tf_loss_generator = \
            tf.reduce_mean(loss_generator_style) + \
            tf.reduce_mean(loss_generator_label)

        crossentropy_labels = tf.nn.\
            softmax_cross_entropy_with_logits(logits=labels_generated,
                                              labels=self._tf_labels)
        self._crossentropy_labels = tf.reduce_mean(crossentropy_labels)

    def _tf_semi_encoder(self, data, keep_prob, semi_supervised=False):
        """Encoder for semi-supervised AAE.

        Arguments
        ---------
        """
        with tf.variable_scope("semi_encoder", reuse=tf.AUTO_REUSE):
            net = tf_helper.dense_layer(data, self._conf.semi_n_hidden,
                                        name="dense_1", keep_prob=keep_prob)
            net = tf_helper.dense_layer(net, self._conf.semi_n_hidden,
                                        name="dense_2", keep_prob=keep_prob)
            style = tf_helper.dense(net, self.code_dim, name="style")

            if semi_supervised is False:
                labels_generated = \
                    tf.nn.softmax(tf_helper.dense(net, self.nclasses,
                                                  name="labels"))
            else:
                labels_generated = \
                    tf_helper.dense(net, self.nclasses,
                                    name="label_logits")

        return style, labels_generated

    def _tf_semi_decoder(self, code, keep_prob):
        """Decoder for semi-supervised AAE.

        Result
        ------
        decoder:
            A flat tensor holding the decoded data.
        """
        flat_data_length = \
            self._data_shape[0] * self._data_shape[1] * self._data_shape[2]
        with tf.variable_scope("semi_decoder", reuse=tf.AUTO_REUSE):
            net = tf_helper.dense_layer(code, self._conf.semi_n_hidden,
                                        name="dense_1", keep_prob=keep_prob)
            net = tf_helper.dense_layer(net, self._conf.semi_n_hidden,
                                        name="dense_2", keep_prob=keep_prob)
            net = tf.nn.sigmoid(tf_helper.dense(net, flat_data_length,
                                                name="dense_3"))
        return net

    def _tf_semi_z_discriminator(self, style, keep_prob):
        """Discriminator for style codes.
        """
        with tf.variable_scope("semi_z_discriminator", reuse=tf.AUTO_REUSE):
            net = tf_helper.dense_layer(style, self._conf.semi_n_hidden,
                                        name="dense_1", keep_prob=keep_prob)
            net = tf_helper.dense_layer(net, self._conf.semi_n_hidden,
                                        name="dense_2", keep_prob=keep_prob)
            logits = tf_helper.dense(net, 1, name="dense_3")
        return logits

    def _tf_semi_y_discriminator(self, label, keep_prob):
        """Discriminator for class labels.
        """
        with tf.variable_scope("semi_y_discriminator", reuse=tf.AUTO_REUSE):
            net = tf_helper.dense_layer(label, self._conf.semi_n_hidden,
                                        name="dense_1", keep_prob=keep_prob)
            net = tf_helper.dense_layer(net, self._conf.semi_n_hidden,
                                        name="dense_2", keep_prob=keep_prob)
            logits = tf_helper.dense(net, 1, name="dense_3")
        return logits

    def _tf_define_optimizers(self, variables) -> None:
        """Prepare the training process. Define optimizers.
        """
        super()._tf_define_optimizers(variables)

        # FIXME[semi]: z and y discriminator
        var_z_discriminator = [var for var in variables
                               if "z_discriminator" in var.name]
        var_y_discriminator = [var for var in variables
                               if "y_discriminator" in var.name]
        var_generator = [var for var in variables
                         if "encoder" in var.name]

        self._op_z_discriminator = \
            tf.train.AdamOptimizer(learning_rate=self._tf_learning_rate/5).\
            minimize(self._tf_loss_discriminator,
                     global_step=self._tf_global_step,
                     var_list=var_z_discriminator)
        self._op_y_discriminator = \
            tf.train.AdamOptimizer(learning_rate=self._tf_learning_rate/5).\
            minimize(self._tf_loss_discriminator,
                     global_step=self._tf_global_step,
                     var_list=var_y_discriminator)
        self._op_generator = \
            tf.train.AdamOptimizer(learning_rate=self._tf_learning_rate).\
            minimize(self._tf_loss_generator,
                     global_step=self._tf_global_step,
                     var_list=var_generator)

        # optimizer for supervised data: minimize cross-entropy between
        # real and fake labels.
        self._op_crossentropy_labels = \
            tf.train.AdamOptimizer(learning_rate=self._tf_learning_rate).\
            minimize(self._crossentropy_labels,
                     global_step=self._tf_global_step,
                     var_list=var_generator)

    def _tf_train_step(self, sess, feed_dict) -> None:
        """Perform one training step.
        """
        # AutoEncoder phase
        super()._tf_train_step(sess, feed_dict)

        # Discriminator phase
        # FIXME[semi]: optimize both discriminators
        self._l_z_discriminator, _ = \
            sess.run([self._tf_loss_discriminator_style,
                      self._op_z_discriminator],
                     feed_dict=feed_dict)
        self._l_y_discriminator, _ = \
            sess.run([self._tf_loss_discriminator_label,
                      self._op_y_discriminator],
                     feed_dict=feed_dict)

        # Generator phase
        self._l_generator, _ = \
            sess.run([self._tf_loss_generator, self._op_generator],
                     feed_dict=feed_dict)

    def _tf_train_supervised_step(self, sess, feed_dict) -> None:
        # Cross_Entropy phase
        self._crossentropy, _ = \
            sess.run([self._crossentropy_labels,
                      self._op_crossentropy_labels],
                     feed_dict=feed_dict)

    def train(self,
              unlabeled_train_data: Datasource,
              labeled_train_data: Datasource,
              labeled_display_data: Datasource,
              sess: tf.Session, saver) -> None:
        """Train the network.
        """
        # display_data: (images, images_noised, labels)
        # A batch of data used for plotting intermediate results
        # during training (taken from the validation dataset)
        # Each is a numpy array of length 100 and appropriate shape.
        display_data = labeled_display_data[:100, ('array', 'array', 'label')]

        # obtain the variables used in the model
        total_vars = tf.trainable_variables()

        # Optimizers
        self._tf_define_optimizers(total_vars)

        #
        # Start the training
        #
        start_epoch = self._tf_initialize_variables(sess, saver)
        start_time = time.time()

        total_batch = len(labeled_train_data) // self._conf.batch_size
        labeled_batch_iterator = \
            labeled_train_data(batch_size=self._conf.batch_size,
                               attributes=('array', 'array', 'label'))
        unlabeled_batch_iterator = \
            unlabeled_train_data(batch_size=self._conf.batch_size,
                                 attributes=('array', 'noisy'))

        for epoch in tqdm(range(start_epoch, self._conf.n_epoch),
                          initial=start_epoch, total=self._conf.n_epoch):
            likelihood = 0
            discriminator_z_value = 0
            discriminator_y_value = 0
            generator_value = 0
            crossentropy_value = 0

            # Adapt the learning rate depending on the epoch
            lr_value = self._learning_rate_schedule(epoch)

            for _batch_idx in tqdm(range(total_batch)):

                #
                # Part 1: unsupervised learning
                #
                batch_xs, batch_noised_xs = next(unlabeled_batch_iterator)

                # FIXME[semi]:
                real_cat_labels = \
                    np.random.randint(low=0, high=self.nclasses,
                                      size=self._conf.batch_size)
                real_cat_labels = np.eye(self.nclasses)[real_cat_labels]

                # Sample from the prior distribution
                prior_style, _tf_prior_label_onehot = \
                    self._sample_prior(self._conf.prior, zdim=self.code_dim,
                                       nclasses=self.nclasses,
                                       batch_size=self._conf.batch_size,
                                       use_label_info=False)

                feed_dict = {
                    self._tf_inputs: batch_noised_xs,
                    self._tf_outputs: batch_xs,
                    # self._tf_labels: batch_ys,
                    self._tf_prior_style: prior_style,
                    # FIXME[semi]: real_cat_labels instead of
                    # prior_label_onehot
                    self._tf_prior_label: real_cat_labels,
                    self._tf_learning_rate: lr_value,
                    self._tf_keep_prob: self._conf.keep_prob
                }

                # perform the actual training
                self._tf_train_step(sess, feed_dict)

                #
                # Part 2: supervised training with labels
                #

                # Obtain a batch of labeled validation data to train
                # the encoder to predict correct class labels
                # (minimizing crossentropy)
                batch_semi_xs, batch_noised_semi_xs, batch_semi_ys = \
                    next(labeled_batch_iterator)

                feed_dict_semi = {
                    self._tf_inputs: batch_noised_semi_xs,
                    self._tf_outputs: batch_semi_xs,
                    self._tf_labels: batch_semi_ys,
                    # FIXME[semi]: _tf_prior_label was Y_cat
                    self._tf_prior_label: real_cat_labels,
                    self._tf_learning_rate: lr_value,
                    self._tf_keep_prob: self._conf.keep_prob
                }

                self._tf_train_supervised_step(self, sess, feed_dict_semi)

                # Summary
                likelihood += \
                    self._loss_reconstruction_value/total_batch
                discriminator_z_value += self._l_z_discriminator/total_batch
                discriminator_y_value += self._l_y_discriminator/total_batch
                generator_value += self._l_generator/total_batch
                crossentropy_value += self._crossentropy/total_batch

            # every 5th epoch (except the last) plot the manifold canvas
            if epoch % 5 == 0 or epoch == (self._conf.n_epoch - 1):
                name = f"Manifold_semi_canvas_{epoch}"
                self.plot_recoded_images(display_data[1],
                                         targets=display_data[0],
                                         filename=name)

            # output end of epoch information
            runtime = time.time() - start_time
            print(f"Epoch: {epoch:3d}, "
                  f"global step: {sess.run(self._tf_global_step)}, "
                  f"Time: {datetime.timedelta(seconds=runtime)}")
            print(f"             lr_AE: {lr_value:.5f}"
                  f"   loss_AE: {likelihood:.4f}")
            print(f"             lr_D: {lr_value/5:.5f}"
                  f"   loss_z_D: {discriminator_z_value:.4f},"
                  f"   loss_y_D: {discriminator_y_value:.4f}")
            print(f"             lr_G: {lr_value:.5f}"
                  f"   loss_G: {generator_value:.4f},"
                  f"   loss_CE: {crossentropy_value:.4f}\n")


def main() -> None:
    """The main program.
    """
    ModelClass = SupervisedAdversarialAutoencoder
    # ModelClass = SemiSupervisedAdversarialAutoencoder

    datasource_train = Datasource(module='mnist', one_hot=True)
    datasource_train.add_postprocessor(add_noise)
    datasource_test = Datasource(module='mnist', section='test', one_hot=True)

    aae = ModelClass(shape=datasource_train.shape, code_dim=2,
                     nclasses=len(datasource_train.label_scheme))
    aae.prepare()

    saver = tf.train.Saver(name='aae', filename="my_test_model")
    # print(f"\n\n##### Saver: {saver.last_checkpoints}\n\n")
    # print("tf.train.latest_checkpoint(): "
    #       f"{tf.train.latest_checkpoint('checkpoints')}")
    # print(f"\n\n##### Saver: {saver.last_checkpoints}\n\n")

    with tf.Session() as sess:
        aae.set_tensorflow_session(sess)
        aae.train(datasource_train, datasource_test, sess, saver)

        if aae.code_dim == 2:
            print("-" * 80)
            print("plot 2D Scatter Result")
            aae.plot_data_codes_2d(datasource_test._array,
                                   labels=datasource_test._labels,
                                   filename='2D_latent_space.png')

        if aae.code_dim and aae.conf.flag_plot_mlr:
            print("-" * 80)
            print("plot Manifold Learning Result")
            # filename = "PMLR/PMLR"
            aae.plot_decoded_codespace_2d(labels=5)

        if aae.conf.flag_plot_arr:
            print("-"*80)
            print("plot analogical reasoning result")
            aae.plot_analogical_decoding()


if __name__ == '__main__':
    main()
