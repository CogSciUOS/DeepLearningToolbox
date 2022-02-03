"""Tensorflow autoencoder implementation.

Originally based on [1] with significant changers.


Demo (command line)
===================

conda activate tf114
python -m dltb.thirdparty.tensorflow.ae


Demo inference (interactive)
----------------------------

from dltb.datasource import Datasource
from dltb.thirdparty.tensorflow.ae import Autoencoder

test_data = Datasource(module='mnist', section='test', one_hot=True)
ae = Autoencoder(shape=test_data.shape, code_dim=2)

batch = test_data[:100]  # FIXME[bug]: ae.encode(batch) does not work
codes = ae.encode(batch)

data, labels = test_data[:1000, ('array', 'label')]
ae.plot_data_codes_2d(data, labels=labels)

Demo training (interactive)
----------------------------

from dltb.datasource import Datasource
from dltb.thirdparty.tensorflow.ae import Autoencoder
from dltb.tool.train import Trainer
from tqdm import tqdm

datasource_train = Datasource(module='mnist', one_hot=True)
ae = Autoencoder(shape=datasource_train.shape, code_dim=2)
trainer = Trainer(trainee=ae, training_data=datasource_train)
trainer.train(epochs=5, restore=False, progress=tqdm)

References
-----------
[1] https://github.com/MINGUKKANG/Adversarial-AutoEncoder.git
"""

# standard imports
from typing import Tuple, Optional
import time
import datetime

# third-party imports
import numpy as np
from tqdm import tqdm

# toolbox imports
from dltb.base.run import runnable
from dltb.datasource import Datasource
from dltb.base.data import Data, add_noise
from dltb.tool.autoencoder import Autoencoder as AutoencoderBase
from .v1 import tf, Utils as tf_helper, TensorflowBase


class Autoencoder(AutoencoderBase, TensorflowBase):
    """Base class for autoencoders.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # TensorFlow placeholders
        self._tf_inputs = None
        self._tf_outputs = None

        # TensorFlow placeholders for hyperparamters
        self._tf_keep_prob = None
        self._tf_learning_rate = None

        # The model
        self._tf_encoded = None
        self._tf_decoded = None
        self._tf_loss_reconstruction = None  # "negative_log_likelihood"

        # Optimizers
        self._tf_optimize_reconstruction = None
        self._loss_reconstruction_value = None

        # TensorFlow session to run this Autoencoder
        self._tf_session = None

    #
    # Implementation of Preparable
    #

    def _prepared(self) -> bool:
        return self._tf_inputs is not None and super()._prepared()

    def _tf_prepare(self) -> None:
        """Prepare this :py:class:`Autoencoder`. Proparation
        consists of preparing the dataset to be used for training and
        evaluation as well as setting up the network structure.
        """
        super()._tf_prepare()
        self._prepare_tensors()

    def _learning_rate_schedule(self, epoch: int) -> float:
        """Adapt the learning rate depending on the epoch.
        """
        if epoch <= 50:
            return self._conf.lr_start
        if epoch <= 100:
            return self._conf.lr_mid

        return self._conf.lr_end

    #
    # Tensorflow code
    #

    def _prepare_tensors(self) -> None:
        """

        The constructed graph looks as follows:

                 [inputs]
                    |
               inputs_flat
                    |
          (encoder) |
                    V
                *encoded*
                    |
          (decoder) |             [outputs]
                    V                 |
                *decoded*        outputs_flat
                    |                 |
                    +--------+--------+
                             |
                             V
                    *loss_reconstruction*

        [...]: input values (tf.placeholder)
                 inputs: the input data to encode
                 outputs: output data (for training). Can be same
                     as inputs, but inputs could also be distorted
                     with noise while outputs should always be clean.
        *...*: output values (tf.tensor)
                 encoded: encoder output (just style, no label)
                 decoded: decoder output (flat data)
                 loss_reconstruction: the reconstruction loss

        Tensorflow properties are prefixed with '_tf_'.
        """

        # Inputs: the data input (images)
        input_shape = (None, ) + self._data_shape
        self._tf_inputs = \
            tf.placeholder(dtype=tf.float32, shape=input_shape,
                           name="Inputs_noised")
        self._tf_outputs = \
            tf.placeholder(dtype=tf.float32, shape=input_shape,
                           name="Inputs")

        # Hyperparameters:
        self._tf_keep_prob = \
            tf.placeholder(dtype=tf.float32, name="dropout_rate")
        self._tf_learning_rate = \
            tf.placeholder(dtype=tf.float32, name="learning_rate")

        #
        # Setting up the actual network
        #

        flat_data_length = \
            self._data_shape[0] * self._data_shape[1]
        if len(self._data_shape) > 2:
            flat_data_length *= self._data_shape[2]
        inputs_flat = tf.reshape(self._tf_inputs, [-1, flat_data_length])
        outputs_flat = tf.reshape(self._tf_outputs, [-1, flat_data_length])

        self._tf_encoded = \
            self._tf_encoder(inputs_flat, self._tf_keep_prob)
        self._tf_decoded = \
            self._tf_decoder(self._tf_encoded, self._tf_keep_prob)

        self._tf_loss_reconstruction = \
            tf.reduce_mean(tf.squared_difference(self._tf_decoded,
                                                 outputs_flat))

    def _tf_encoder(self, data, keep_prob):
        """Encoder for an autoencoder.
        """
        with tf.variable_scope("sup_encoder", reuse=tf.AUTO_REUSE):
            net = tf_helper.dense_layer(data, self._conf.super_n_hidden,
                                        name="dense_1", keep_prob=keep_prob)
            net = tf_helper.dense_layer(net, self._conf.super_n_hidden,
                                        name="dense_2", keep_prob=keep_prob)
            net = tf_helper.dense(net, self.code_dim, name="dense_3")
        return net

    def _tf_decoder(self, code, keep_prob):
        """Decoder for an autoencoder.
        """
        flat_data_length = np.prod(self._data_shape)
        with tf.variable_scope("sup_decoder", reuse=tf.AUTO_REUSE):
            net = tf_helper.dense_layer(code, self._conf.super_n_hidden,
                                        name="dense_1", keep_prob=keep_prob)
            net = tf_helper.dense_layer(net, self._conf.super_n_hidden,
                                        name="dense_2", keep_prob=keep_prob)
            net = tf.nn.sigmoid(tf_helper.dense(net, flat_data_length,
                                                name="dense_3"))
        return net

    def _tf_discriminator(self, code, keep_prob):
        """Discriminator for supervised AAE.
        """
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            net = tf_helper.dense_layer(code, self._conf.super_n_hidden,
                                        name="dense_1", keep_prob=keep_prob)
            net = tf_helper.dense_layer(net, self._conf.super_n_hidden,
                                        name="dense_2", keep_prob=keep_prob)
            logits = tf_helper.dense(net, 1, name="dense_3")
        return logits

    #
    # Coding API
    #

    @staticmethod
    def _np_batcher(data: np.ndarray, batch_size: int = 128):
        if isinstance(data, Data):
            data = data.array
        length = len(data)
        for offset in range(0, length, batch_size):
            yield data[offset: min(offset + batch_size, length)]

    def _encode(self, data: np.ndarray, batch_size: int = 128) -> np.ndarray:
        length = len(data)
        code = np.ndarray((length, self.code_dim))

        offset = 0
        batches = self._np_batcher(data, batch_size=batch_size)
        for batch in self._tf_encode_batches(batches):
            end = offset + len(batch)
            code[offset: end] = batch
            offset = end
        return code

    def _decode(self, code: np.ndarray, batch_size: int = 128) -> np.ndarray:
        length = len(code)
        data = np.ndarray((length,) + self.data_shape)

        offset = 0
        batches = self._np_batcher(code, batch_size=batch_size)
        for batch in self._tf_decode_batches(batches):
            end = offset + len(batch)
            data[offset: end] = batch
            offset = end
        return data

    def _recode(self, data: np.ndarray, batch_size: int = 128) -> np.ndarray:
        length = len(data)
        recoded = np.ndarray((length,) + self.data_shape)

        offset = 0
        batches = self._np_batcher(data, batch_size=batch_size)
        for batch in self._tf_recode_batches(batches):
            end = offset + len(batch)
            recoded[offset: end] = batch
            offset = end
        return recoded
    
    def _tf_encode_batch(self, batch):
        """Encode a single batch of data values given as numpy array into a
        batch of corresponding code points, using the encoder part of
        the autoencoder.

        Arguments
        ---------
        data:
            A a batch of data.  It is assumed that the batch has
            an appropriate shape.

        Result
        ------
        code:
            A batch of corresponding code values.
        """
        if self._tf_session is None:
            raise RuntimeError("No session.")

        feed_dict = {
            self._tf_inputs: batch,
            self._tf_keep_prob: 1.0
        }
        return self._tf_session.run(self._tf_encoded, feed_dict=feed_dict)

    def _tf_encode_batches(self, batches, tf_encoded=None):
        """Encode batched data from an iterator.
        """
        if self._tf_session is None:
            raise RuntimeError("No session.")

        if tf_encoded is None:
            tf_encoded = self._tf_encoded

        feed_dict = {
            self._tf_keep_prob: 1.0
        }
        for batch in batches:
            feed_dict[self._tf_inputs] = batch
            yield self._tf_session.run(tf_encoded, feed_dict=feed_dict)

    def _tf_decode_batch(self, code: np.ndarray) -> np.ndarray:
        """Decode a batch of code values given as numpy array
        into a batch of datapoints, using the decoder part of
        the autoencoder.

        Arguments
        ---------
        code:
            A batch of code values.  It is assumed that the
            batch has an appropriate shape.

        Result
        ------
        data:
            A batch containing the decoded data points. Batch size
            and order of elements corresponds to the code values.
        """
        if self._tf_session is None:
            raise RuntimeError("No session.")

        feed_dict = {
            self._tf_encoded: code,
            self._tf_keep_prob: 1.0
        }
        data = self._tf_session.run(self._tf_decoded, feed_dict=feed_dict)
        data = np.reshape(data, (-1,) + self._data_shape)
        return data

    def _tf_decode_batches(self, batches):
        """Decode batched codes from an iterator.
        """
        if self._tf_session is None:
            raise RuntimeError("No session.")

        feed_dict = {
            self._tf_keep_prob: 1.0
        }
        for batch in batches:
            feed_dict[self._tf_encoded] = batch
            data = self._tf_session.run(self._tf_decoded, feed_dict=feed_dict)
            yield np.reshape(data, (-1,) + self._data_shape)

    def _tf_recode_batch(self, data: np.ndarray) -> np.ndarray:
        """Recode a batch of data values given as numpy array
        """
        if self._tf_session is None:
            raise RuntimeError("No session.")

        feed_dict = {
            self._tf_inputs: data,
            self._tf_keep_prob: 1.0
        }
        recoded = self._tf_session.run(self._tf_decoded, feed_dict=feed_dict)
        return np.reshape(recoded, (-1,) + self._data_shape)

    def _tf_recode_batches(self, batches):
        """Recode batched data from an iterator.
        """
        if self._tf_session is None:
            raise RuntimeError("No session.")

        feed_dict = {
            self._tf_keep_prob: 1.0
        }
        for batch in batches:
            feed_dict[self._tf_inputs] = batch
            # FIXME[todo]: the supervised autoencoder needs some
            # labels here!
            recoded = \
                self._tf_session.run(self._tf_decoded, feed_dict=feed_dict)
            yield np.reshape(recoded, (-1,) + self._data_shape)

    def _tf_define_optimizers(self, variables) -> None:
        """Prepare the training process. Define optimizers.
        """
        var_ae = [var for var in variables
                  if "encoder" in var.name or "decoder" in var.name]
        self._tf_optimize_reconstruction = \
            tf.train.AdamOptimizer(learning_rate=self._tf_learning_rate).\
            minimize(self._tf_loss_reconstruction,
                     global_step=self._tf_global_step,
                     var_list=var_ae)

    def _tf_prepare_training(self) -> int:
        """Prepare training by defining the optimizers.
        """
        super()._tf_prepare_training()

        # obtain the variables used in the model
        total_vars = tf.trainable_variables()

        # define the optimizers
        self._tf_define_optimizers(total_vars)

    def _tf_train_step(self, sess, feed_dict) -> None:
        """Perform one training step.
        """
        # Optmize the reconstruction loss. This requires
        # 'inputs' and 'outputs' to be given in the 'feed_dict'
        # _negative_log_likelihood_value, _, _g = \
        self._loss_reconstruction_value, _, _g = \
            sess.run([self._tf_loss_reconstruction,
                      self._tf_optimize_reconstruction,
                      self._tf_global_step],
                     feed_dict=feed_dict)

    #
    # Training
    #

    def _tf_train_batch(self, inputs, outputs,
                        learning_rate: float, sess: tf.Session) -> float:
        feed_dict = {
            self._tf_inputs: inputs,
            self._tf_outputs: outputs,
            self._tf_learning_rate: learning_rate,
            self._tf_keep_prob: self._conf.keep_prob
        }

        # AutoEncoder phase
        self._tf_train_step(sess, feed_dict)

        return self._loss_reconstruction_value

    def train_batch(self, data: Data, epoch: int = None):
        batch_xs, batch_noised_xs = data.array, data.array

        lr_value = self._learning_rate_schedule(epoch)

        loss_reconstruction = \
            self._tf_train_batch(batch_noised_xs, batch_xs,
                                 lr_value, self._tf_session)

        # Summary
        # avg_loss_reconstruction += \
        #     loss_reconstruction/total_batch
        return loss_reconstruction


def train(model,
          training_data: Datasource,
          display_data: Datasource,
          sess: tf.Session, _saver) -> None:
    """Train the network
    """
    # display_data: (images, images_noised, labels)
    # A batch of data used for plotting intermediate results
    # during training (taken from the validation dataset)
    # Each is a numpy array of length 100 and appropriate shape.
    display_data = display_data[:100, ('array', 'array', 'label')]

    model.prepare_training()
    start_epoch = \
        model.get_hyperparamter('global_step') // model.conf.batch_size

    start_time = time.time()

    total_batch = len(training_data) // model.conf.batch_size
    batch_iterator = \
        training_data(batch_size=model.conf.batch_size, loop=True,
                      attributes=('array', 'noisy', 'label'))

    for epoch in tqdm(range(start_epoch, model.conf.n_epoch),
                      initial=start_epoch, total=model.conf.n_epoch):
        avg_loss_reconstruction = 0

        # Adapt the learning rate depending on the epoch
        lr_value = model._learning_rate_schedule(epoch)

        for _batch_idx in tqdm(range(total_batch)):
            batch_xs, batch_noised_xs, _batch_ys = \
                next(batch_iterator)

            loss_reconstruction = \
                model._tf_train_batch(batch_noised_xs, batch_xs,
                                      lr_value, sess)

            # Summary
            avg_loss_reconstruction += \
                loss_reconstruction/total_batch

        # every 5th epoch (except the last) plot the manifold canvas
        if epoch % 5 == 0 or epoch == (model.conf.n_epoch - 1):
            name = f"Manifold_canvas_{epoch}"
            model.plot_recoded_images(display_data[1],
                                      targets=display_data[0],
                                      # labels=display_data[2],
                                      filename=name)

        # output end of epoch information
        runtime = time.time() - start_time
        print(f"Epoch: {epoch:3d}, "
              f"global step: {sess.run(model._tf_global_step)}, "
              f"Time: {datetime.timedelta(seconds=runtime)}")
        print(f"             lr_AE: {lr_value:.5f}"
              f"   loss_AE: {avg_loss_reconstruction:.4f}   ")

        model.store_checkpoint()


def main() -> None:
    """The main program.
    """

    datasource_train = Datasource(module='mnist', one_hot=True)
    datasource_train.add_postprocessor(add_noise)
    datasource_test = Datasource(module='mnist', section='test', one_hot=True)

    autoencoder = Autoencoder(shape=datasource_train.shape, code_dim=2)
    print(autoencoder.prepared)

    saver = tf.train.Saver(name='aa', filename="my_test_model")

    with tf.Session() as sess:
        train(autoencoder, datasource_train, datasource_test, sess, saver)

        if autoencoder.code_dim == 2:
            print("-" * 80)
            print("plot 2D Scatter Result")
            autoencoder.plot_data_codes_2d(datasource_test._array,
                                           labels=datasource_test._labels,
                                           filename='2D_latent_space.png')

        if autoencoder.code_dim and autoencoder.conf.flag_plot_mlr:
            print("-" * 80)
            print("plot Manifold Learning Result")
            # filename = "PMLR/PMLR"
            autoencoder.plot_decoded_codespace_2d(labels=5)

        if autoencoder.conf.flag_plot_arr:
            print("-"*80)
            print("plot analogical reasoning result")
            autoencoder.plot_analogical_decoding()


if __name__ == '__main__':
    main()
