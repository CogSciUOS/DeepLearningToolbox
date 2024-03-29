"""Implementation of the :py:class:`dltb.network.Network` interface
based on Keras.
"""

# standard imports
from types import ModuleType, FunctionType
from typing import List
from collections import OrderedDict

# third party imports
import numpy as np
import keras.layers
from keras import backend as K

# toolbox imports
from dltb.util.importer import import_module
from dltb.network import Network as BaseNetwork
from dltb.network import NetworkParsingError as ParsingError
from dltb.network import Classifier
from . import keras
from .layer import Layer, Conv2D, Dense, MaxPooling2D, Dropout, Flatten


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
        print("**network.keras.Network:", kwargs)
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

    def _get_activations(self, input_samples: np.ndarray,
                         layer_ids: list,) -> List[np.ndarray]:
        """To be implemented by subclasses.
        Computes a list of activations from a list of layer ids.
        """

        # input placeholder
        inputs = [self._model.input, K.learning_phase()]

        # all layer outputs except first (input) layer
        outputs = [self[layer]._keras_layer_objs[-1].output
                   for layer in layer_ids]

        # activations function
        activations_functor = K.function(inputs, outputs)

        # do the actual computation (0.=training, 1.=predict)
        activations = activations_functor([input_samples, 1.])

        return activations


from network.network import Trainer as BaseTrainer
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
        BaseTrainer.__init__(self, training, network)
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
        self.notify_observers('epoch_changed')

    def on_batch_begin(self, batch, logs=None):
        """
        on_batch_begin: logs include `size`,
          the number of samples in the current batch.
        """
        self._training.batch = batch
        self._batch_size = logs['size']
        self._batch_start = time.time()
        self.notify_observers('batch_changed')

    def on_batch_end(self, batch, logs=None):
        """
        on_batch_end: logs include `loss`, and optionally `acc`
        """
        self._batch_duration = time.time() - self._batch_start
        self._loss = logs['loss']
        if 'acc' in logs:
            self._accuracy = logs['acc']
        self.notify_observers('metrics_changed')
    
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
        self.notify_observers('metrics_changed')

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
        self.notify_observers('data_changed')


class ApplicationsNetwork(Network, Classifier):
    """One of the predefined and pretrained networks from the
    :py:mod:`keras.applications`.


    Parameters
    ----------
    model: str
        The name of the model, e.g. `ResNet50`.
    module: str
        The python module in which the model is defined, e.g.,
        `tensorflow.keras.applications`.
    """
    KERAS_APPLICATIONS = 'tensorflow.keras.applications'

    def __init__(self, model: str, module: str = None, **kwargs) -> None:
        super().__init__(scheme='ImageNet', **kwargs)

        # evaluate the module argument
        if module is None:
            self._module, self._module_name = None, None
        elif isinstance(module, ModuleType):
            self._module, self._module_name = module, module.__name__
        elif isinstance(module, str):
            self._module = None
            self._module_name = (module if '.' in module else
                                 '.'.join((self.KERAS_APPLICATIONS, module)))
        else:
            raise TypeError("Argument 'module' should by module "
                            "or module name")

        # evaluate the model argument
        module_name = None
        self._model, self._model_name, self._model_function = None, None, None
        if isinstance(model, str):
            # from tensorflow.keras.applications import ResNet50
            # from tensorflow.keras.applications.resnet50 import ResNet50
            if "." in model:
                module_name, self._model_name = module.rsplit('.')
            else:
                self._model_name = model
                if self._module_name is None:
                    self._module_name = ".".join((self.KERAS_APPLICATIONS,
                                                  model.lower()))
        elif isinstance(model, FunctionType):
            # model.__module__ = 'tensorflow.python.keras.applications'
            # model.__name__ = 'wrapper'
            if model.__name__ == 'wrapper':
                self._model_name = model.__closure__[0].cell_contents.__name__
            else:
                self._model_name = model.__name
            self._model_function = model
        elif model is None:
            raise ValueError("No Model provided for ApplicationsNetwork")
        else:
            raise TypeError("Instantiation of ApplicationsNetwork with "
                            "bad model type ({type(model)}).")

        # Consistency Check
        if (module_name and self._module_name and
                module_name != self._module_name):
            # Inconsistent module names
            raise ValueError("Inconsistent module name: "
                             "'{self._module_name}' should be '{module_name}'")

    def _prepared(self) -> bool:
        """Check if this :py:class:`ApplicationsNetwork` has been prepared.

        """
        return self._model is not None and super()._prepared()

    def _prepare(self) -> None:
        if self._module is None:
            self._module = import_module(self._module_name)

        self.preprocess_input = getattr(self._module, 'preprocess_input')
        self.decode_predictions = getattr(self._module, 'decode_predictions')

        if self._model is None:
            if self._model_function is None:
                self._model_function = getattr(self._module, self._model_name)

            # Initialize the model (will download data if not yet present)
            self._model = self._model_function(weights='imagenet')
            # FIXME[warning]: (tensorflow.__version__ = '1.15.0')
            # From tensorflow_core/python/ops/resource_variable_ops.py:1630:
            #   calling BaseResourceVariable.__init__ (from
            #   tensorflow.python.ops.resource_variable_ops) with constraint
            #   is deprecated and will be removed in a future version.
            # Instructions for updating:
            # If using Keras pass *_constraint arguments to layers.

        super()._prepare()

    #
    # Preprocessing input
    #

    # Some models use images with values ranging from 0 to 1. Others
    # from -1 to +1. Others use the "caffe" style, that is not
    # normalized, but is centered.

    # From the source code, Resnet is using the caffe style.
    #

    # You don't need to worry about the internal details of
    # preprocess_input. But ideally, you should load images with the
    # keras functions for that (so you guarantee that the images you
    # load are compatible with preprocess_input).

    # Example: https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet/preprocess_input

    # i = tf.keras.layers.Input([None, None, 3], dtype = tf.uint8)
    # x = tf.cast(i, tf.float32)
    # x = tf.keras.applications.mobilenet.preprocess_input(x)
    # core = tf.keras.applications.MobileNet()
    # x = core(x)
    # model = tf.keras.Model(inputs=[i], outputs=[x])
    #
    # image = tf.image.decode_png(tf.io.read_file('file.png'))
    # result = model(image)

    # https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet/preprocess_input
    #
    # The images are converted from RGB to BGR, then each color
    # channel is zero-centered with respect to the ImageNet dataset,
    # without scaling.


# FIXME[hack]: what is this supposed to do?
class KerasClassifier(Network, Classifier):

    def __init__(self, **kwargs):
        print("**network.keras_tensorflow.Classifier:",
              kwargs, self.__class__.__mro__)
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
