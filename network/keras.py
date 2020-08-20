
# standard imports
from __future__ import absolute_import
from typing import List
from collections import OrderedDict

# third party imports

# toolbox imports
from . import Network as BaseNetwork, Classifier as BaseClassifier
from .exceptions import ParsingError
from . import layers
from dltb.thirdparty.keras import keras


class Layer(layers.Layer):
    """A keras :py:class:`Layer` implements the abstract layer class
    based on keras layer object.

    A `py:class:`Layer` may bundle multiple keras layer objects of the
    unterlying Keras model, e.g., when convolution and activation
    function are realized be separate Keras layers.
    """

    def __init__(self, network: BaseNetwork,
                 keras_layer_objs: List[keras.layers.Layer]) -> None:
        super().__init__(network)
        if len(keras_layer_objs) > 6:  # FIXME[hack]: was 2 not 6
            raise ValueError('A layer should at most contain 2 keras layer'
                             'objects: normal plus activation')
        self._keras_layer_objs = keras_layer_objs

    @property
    def input_shape(self):
        """The input shape is obtained from the first Keras layer bundled
        in this `py:class:`Layer`.
        """
        return self._keras_layer_objs[0].input_shape

    @property
    def output_shape(self):
        """The output shape is obtained from the last Keras layer bundled
        in this `py:class:`Layer`.
        """
        return self._keras_layer_objs[-1].output_shape


class NeuralLayer(Layer, layers.NeuralLayer):
    """A keras `NeuralLayer` consists of one or two keras.layers.Layer
    objects.
    """

    @property
    def parameters(self):
        return self._keras_layer_objs[0].get_weights()

    @property
    def num_parameters(self):
        return self._keras_layer_objs[0].count_params()

    @property
    def weights(self):
        return self._keras_layer_objs[0].get_weights()[0]

    @property
    def bias(self):
        return self._keras_layer_objs[0].get_weights()[1]


class StridingLayer(Layer, layers.StridingLayer):

    @property
    def strides(self):
        return self._keras_layer_objs[0].strides

    @property
    def padding(self):
        return self._keras_layer_objs[0].padding


class Dense(NeuralLayer, layers.Dense):
    pass


class Conv2D(NeuralLayer, StridingLayer, layers.Conv2D):

    @property
    def kernel_size(self):
        return self._keras_layer_objs[0].kernel_size

    @property
    def filters(self):
        return self._keras_layer_objs[0].filters


class MaxPooling2D(StridingLayer, layers.MaxPooling2D):

    @property
    def pool_size(self):
        return self._keras_layer_objs[0].pool_size


class Dropout(Layer, layers.Dropout):
    pass


class Flatten(Layer, layers.Flatten):
    pass


class Network(BaseNetwork):
    """Abstract base class for the keras networks for the specific backends.

    Implements only the functionality that can be efficiently
    implemented in pure Keras.
    """

    def __init__(self, *args, model_file: str = None, model=None, **kwargs):
        """
        Load Keras model.
        Parameters
        ----------
        model_file
            Path to the .h5 model file.
        """
        print("**keras.Network:", kwargs)
        kwargs['data_format'] = keras.backend.image_data_format()

        super().__init__(*args, **kwargs)
        self._model_file = model_file
        self._model = model

    def _prepare(self) -> None:
        # Set learning phase to train in case setting to test would
        # affect gradient computation.
        #
        # FIXME[todo]: Set it to test, to eliminate dropout, check for
        # gradient computation later.
        keras.backend.set_learning_phase(0)

        # Note: we have to load the model before calling super()._prepare(),
        # as the model is required to provide the layers, requested
        # by super()._prepare().
        if self._model is None:
            self._prepare_model()
        else:
            print("network.keras: INFO: model graph was already set")
        super()._prepare()

    def _prepare_model(self) -> None:
        if self._model_file is not None:
            self._model = keras.models.load_model(self._model_file)

    def _unprepare(self) -> None:
        if self._model_file is not None:
            self._model = None
        super()._unprepare()

    def _prepared(self) -> bool:
        return (self._model is not None) and super()._prepared()

    #
    # Layers
    #

    # Missing layer types:
    # - InputLayer
    # - ZeroPadding2D
    # - BatchNormalization

    # A mapping of keras.layer.Layer class names to
    # network.keras.Layer class names
    _layer_types_to_classes = {
        'Conv2D': Conv2D,
        'Dense': Dense,
        'MaxPooling2D': MaxPooling2D,
        'Dropout': Dropout,
        'Flatten': Flatten
    }

    # Layer types that encapsulate the inner product, bias add,
    # activation pattern.
    _neural_layer_types = {'Dense', 'Conv2D'}

    # Layer types that just map input to output without trainable
    # parameters.
    _transformation_layer_types = {'MaxPooling2D', 'Flatten', 'Dropout'}

    def _compute_layer_ids(self):
        """

        Returns
        -------

        """
        # Apparently config can be sometimes a dict or
        # simply a list of `layer_specs`.
        config = self._model.get_config()
        if isinstance(config, dict):
            layer_specs = config['layers']
        elif isinstance(config, list):
            layer_specs = config
        else:
            raise ValueError('Config is neither a dict nor a list.')
        return [layer_spec['config']['name'] for layer_spec in layer_specs]

    def _create_layer_dict(self) -> OrderedDict:
        """
        """

        layer_ids = self._compute_layer_ids()
        layer_dict = OrderedDict()
        last_layer_id = ''
        last_layer_type = None

        # current_layers: Layers that could not be wrapped in objects
        # yet, because they were missing the activation function.
        current_layers = []

        for i, layer_id in enumerate(layer_ids):
            keras_layer_obj = self._model.get_layer(layer_id)
            layer_type = keras_layer_obj.__class__.__name__
            if (layer_type not in self._layer_types_to_classes and
                layer_type not in self._neural_layer_types and
                layer_type not in self._transformation_layer_types and
                layer_type not in {'Activation', 'BatchNormalization',
                                   'ZeroPadding2D', 'Add', 'InputLayer',
                                   'GlobalAveragePooling2D'}):
                print(f"({i}) {layer_id} ({layer_type})")
        #
        for layer_id in layer_ids:
            keras_layer_obj = self._model.get_layer(layer_id)
            layer_type = keras_layer_obj.__class__.__name__

            # Check whether the layer type is considered a real layer.
            if layer_type in self._layer_types_to_classes:

                # Check that if the layer is a neural layer it
                # contains an activation function.  If so the layer
                # can be added savely.  Otherwise take the next
                # activation function and merge the layers
                if ((layer_type in self._neural_layer_types and
                     keras_layer_obj.activation.__name__ != 'linear') or
                    (layer_type in self._transformation_layer_types)):
                    # Check that there is no unfinished layer with
                    # missing activation function.  If not add the
                    # layer.
                    # FIXME[hack]: unfinished layer can be longer than 1 element
                    if True or not current_layers:
                        cls = self._layer_types_to_classes[layer_type]
                        layer_dict[layer_id] = cls(self, [keras_layer_obj])
                    else:
                        raise ParsingError("Missing activation function: "
                                           f"{current_layers}")
                else:
                    # Check that there is no other unfinished layer
                    # before that one.
                    # FIXME[hack]: there may be more than one layer
                    # without activation functions, e.g.
                    #   ZeroPadding2D
                    #   Conv2D
                    #   BatchNormalization
                    #   Activation
                    # or
                    #   Conv2D
                    #   Conv2D
                    #   BatchNormalization
                    #   BatchNormalization
                    #   Add
                    #   Activation
                    # Work this out in more detail!
                    if True or not current_layers:
                        current_layers.append(keras_layer_obj)
                        last_layer_id = layer_id
                        last_layer_type = layer_type
                    else:
                        raise ParsingError("Two consectutive layers with no "
                                           "activation function:"
                                           f"{current_layers} and {layer_type}")

            elif layer_type == 'Activation':
                # Check that there was a layer before without
                # activation function and merge.
                if current_layers:
                    current_layers.append(keras_layer_obj)
                    cls = self._layer_types_to_classes[last_layer_type]
                    layer_dict[last_layer_id] = cls(self, current_layers)
                    current_layers = []
                else:
                    raise ParsingError("Two activation layers "
                                       "after each other.")
            elif layer_type in {'InputLayer', 'BatchNormalization',
                                'ZeroPadding2D', 'Add', 'InputLayer',
                                'GlobalAveragePooling2D'}:
                # FIXME[hack]
                current_layers.append(keras_layer_obj)
            else:
                raise ParsingError(f"Not sure how to deal with {layer_type} "
                                   f"with current layers ({current_layers}).")

        return OrderedDict(layer_dict)

    #
    # Trainer
    #

    def get_trainer(self, training):
        return Trainer(training, self)

    #
    # FIXME[old]: this was network.keras.KerasModel
    #

    def load(self, filename: str):
        with self._graph.as_default():
            with self._session.as_default():
                self._vae.load_weights(filename)
                LOG.info(f'loaded autoencoder from "{filename}"')

    def save(self, filename: str):
        with self._graph.as_default():
            with self._session.as_default():
                self._vae.save_weights(filename)
                LOG.info(f'saved autoencoder to "{filename}"')

    _snapshot = None

    def snapshot(self):
        """Make a snapshot of the current model state (weights).
        """
        self._snapshot = self._model.get_weights()

    @property
    def model(self):
        return self._model

    # plot_model requires that pydot is installed
    #from keras.utils import plot_model
    def plot(self):  # FIXME[hack]: plot_model allows for more parameters ...
        plot_model(self._model)


from .network import Trainer as BaseTrainer
from keras.callbacks import Callback

import time


class Trainer(BaseTrainer, Callback):
    """

    This keras Trainer class implements a keras Callback, allowing
    to get some information on the training process.

    """

    def __init__(self, training, network, count_mode='samples'):
        # FIXME[question]: can we use real python multiple inheritance here?
        # (that is just super().__init__(*args, **kwargs))
        BaseTraining.__init__(self, training, network)
        Callback.__init__(self)
        if count_mode == 'samples':
            self.use_steps = False
        elif count_mode == 'steps':
            self.use_steps = True
        else:
            raise ValueError('Unknown `count_mode`: ' + str(count_mode))

    def _train(self):

        # This will start the keras training loop.
        # As a result, the on_training_begin callback will be called,
        # which will also notify the observers.
        x_train = self._training._x_train
        x_test = self._training._x_test
        epochs = self._training.epochs
        batch_size = self._batch_size
        keras_model = self._training.network
        keras_model.train(x_train, x_test, epochs=epochs,
                          batch_size=batch_size, progress=self)

    def stop(self):
        print("Stopping Training")
        # This will cause the keras training loop to stop.
        # As a result, the on_training_end callback will be called,
        # which will also notify the observers.
        keras_model = self._training.network
        keras_model.stop_training = True

    def on_train_begin(self, logs=None):
        """Called on the beginning of training.
        """
        #self._epochs = self.params['epochs']
        #self._batch_size = self.params['batch_size']
        #self._samples = self.params['samples']
        #self._batches = self._samples // self._batch_size
        self._training.epoch = 0
        self._training.batch = 0
        self._batch_duration = 0.
        self._start = time.time()
        super().start()

    def on_epoch_begin(self, epoch, logs=None):
        self._training.epoch = epoch
        self.notifyObservers('epoch_changed')

    def on_batch_begin(self, batch, logs=None):
        """
        on_batch_begin: logs include `size`,
          the number of samples in the current batch.
        """
        self._training.batch = batch
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

    def set_data(self, x_train, y_train, x_test, y_test):
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self.notifyObservers('data_changed')


from packaging import version
if version.parse(keras.__version__) >= version.parse('2.0.0'):
    from keras.layers import Conv2D
else:
    from keras.layers import Convolution2D

def conv_2d(filters, kernel_shape, strides, padding, input_shape=None):
    """
    Defines the right convolutional layer according to the
    version of Keras that is installed.
    :param filters: (required integer) the dimensionality of the output
        space (i.e. the number output of filters in the convolution)
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


# FIXME[hack]: what is this supposed to do?
class Classifier(Network, BaseClassifier):

    def __init__(self, **kwargs):
        print("**keras_tensorflow.Classifier:", kwargs, self.__class__.__mro__)
        super().__init__(**kwargs)
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
