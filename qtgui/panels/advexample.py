"""
File: logging.py
Author: Ulf Krumnack
Email: krumnack@uni-osnabrueck.de
Github: https://github.com/krumnack
"""


from toolbox import Toolbox

from .panel import Panel
from qtgui.widgets.matplotlib import QMatplotlib
from qtgui.widgets.training import QTrainingBox
from qtgui.utils import QObserver

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QPushButton, QSpinBox,
                             QVBoxLayout, QHBoxLayout)


class AdversarialExamplePanel(Panel, QObserver, Toolbox.Observer):
    """A panel displaying adversarial examples.

    Attributes
    ----------
    _network: NetworkView
        A network trained as autoencoder.
    """

    def __init__(self, parent=None):
        """Initialization of the AdversarialExamplePanel.

        Parameters
        ----------
        parent: QWidget
            The parent argument is sent to the QWidget constructor.
        """
        super().__init__(parent)
        self._controller = None

        self._initUI()
        self._layoutComponents()

        self.setController(AdversarialExampleController())
        

    def _initUI(self):
        
        #
        # Controls
        #
        self._buttonCreateModel = QPushButton("Create")
        self._buttonTrainModel = QPushButton("Train")
        self._buttonLoadModel = QPushButton("Load")
        self._buttonSaveModel = QPushButton("Save")
        self._buttonResetModel = QPushButton("Reset")
        self._buttonPlotModel = QPushButton("Plot Model")
        self._buttonShowExample = QPushButton("Show")
        self._buttonShowExample.clicked.connect(self._onShowExample)

        #
        # Plots
        #
        self._trainingBox = QTrainingBox()
        self._pltOriginal = QMatplotlib()
        self._pltAdversarial = QMatplotlib()

    def _layoutComponents(self):
        """Layout the UI elements.
        """
        plotBar = QHBoxLayout()
        plotBar.addWidget(self._trainingBox)
        plotBar.addWidget(self._pltOriginal)
        plotBar.addWidget(self._pltAdversarial)

        buttonBar = QHBoxLayout()
        buttonBar.addWidget(self._buttonCreateModel)
        buttonBar.addWidget(self._buttonTrainModel)
        buttonBar.addWidget(self._buttonLoadModel)
        buttonBar.addWidget(self._buttonSaveModel)
        buttonBar.addWidget(self._buttonResetModel)
        buttonBar.addWidget(self._buttonPlotModel)
        buttonBar.addWidget(self._buttonShowExample)

        layout = QVBoxLayout()
        layout.addLayout(plotBar)
        layout.addLayout(buttonBar)
        self.setLayout(layout)

    # FIXME[hack]: no quotes!
    def setController(self, controller: 'AdversarialExampleController') -> None:
        self._controller = controller
        self._buttonCreateModel.clicked.connect(controller.create_model)
        self._buttonTrainModel.clicked.connect(controller.train_model)
        self._buttonLoadModel.clicked.connect(controller.load_model)
        self._buttonSaveModel.clicked.connect(controller.save_model)
        self._buttonResetModel.clicked.connect(controller.reset_model)
        self.observe(controller)

    def _enableComponents(self, running=False):
        print(f"enable components: {running}")
        available = self._controller is not None and not running
        self._buttonCreateModel.setEnabled(not running)
        for w in (self._buttonTrainModel,
                  self._buttonLoadModel, self._buttonSaveModel,
                  self._buttonPlotModel, 
                  self._buttonShowExample):
            w.setEnabled(available)


    def _onShowExample(self):
        if self._controller is None:
            self._pltOriginal.noData()
            self._pltAdversarial.noData()
        else:
            example_data, example_label, example_prediction = \
                self._controller.get_example()
            with self._pltOriginal as ax:
                ax.imshow(example_data[:,:,0], cmap='Greys_r')
                ax.set_title(f"Label = {example_label.argmax()}, "
                             f"Prediction = {example_prediction.argmax()}")

            adversarial_data, adversarial_prediction = \
                self._controller.get_adversarial_example()
            with self._pltAdversarial as ax:
                ax.imshow(adversarial_data[:,:,0], cmap='Greys_r')
                ax.set_title(f"Prediction = {adversarial_prediction.argmax()}")

    def adversarialControllerChanged(self, controller, change):
        if 'busy_changed' in change:
            self._enableComponents(controller.busy)

# FIXME[hack]: this is just using a specific keras network as proof of
# concept. It has to be modularized and integrated into the framework

import os

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import tensorflow as tf
from tensorflow import keras
import numpy as np

print(f"numpy: {np.__version__}")
print(f"tensorflow: {tf.__version__}")

from base.observer import Observable
import util

import cleverhans
from cleverhans.attacks import FastGradientMethod
from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.train import train
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_tf import model_eval

print(f"cleverhans: {cleverhans.__version__}")
print(f"keras: {keras.__version__} (backend: {keras.backend.backend()}, dim_ordering: {keras.backend.image_data_format()})")

assert keras.backend.image_data_format() == 'channels_last', "this tutorial requires keras to be configured to channels_last format"


from network.keras import KerasClassifier
# FIXME[hack]
from models.example_keras_advex_mnist import KerasMnistClassifier


class AdversarialExampleController(Observable,
                                   method='adversarialControllerChanged',
                                   changes=['busy_changed', 'data_changed',
                                            'parameter_changed']):
    """

    Attributes
    ----------

    _model: leverhans.model.Model
        The cleverhans model used to 
    """

    _nb_epochs                  = 6
    _batch_size                 = 128
    _learning_rate              = 0.001
    _train_dir                  = 'train_dir'
    _filename                   = 'mnist.ckpt'
    _testing                    = False
    _label_smoothing            = 0.1


    # FIXME[todo]: this needs to be initialized ...
    _runner: Runner = None

    def __init__(self):
        super().__init__()

        self._model = None
        self._loss = None

        self._input_placeholder = None
        self._label_placeholder = None
        self._preds = None

        self._graph = None
        self._sess = None

        self._busy = False

        self.load_mnist()  # FIXME[hack]

        # FIXME[old]: check what is still needed from the following code
        # Object used to keep track of (and return) key accuracies
        self._report = AccuracyReport()

        # Set numpy random seed to improve reproducibility
        self._rng = np.random.RandomState([2017, 8, 30])

        # Set TF random seed to improve reproducibility
        tf.set_random_seed(1234)

        self._train_params = {
            'nb_epochs': self._nb_epochs,
            'batch_size': self._batch_size,
            'learning_rate': self._learning_rate,
            'train_dir': self._train_dir,
            'filename': self._filename
        }
        self._eval_params = {
            'batch_size': self._batch_size
        }

        if not os.path.exists(self._train_dir):
            os.mkdir(self._train_dir)

        self._ckpt = tf.train.get_checkpoint_state(self._train_dir)
        print(f"train_dir={self._train_dir}, chheckpoint={self._ckpt}")
        self._ckpt_path = False if self._ckpt is None else self._ckpt.model_checkpoint_path

    def init_from_keras_classifier(self, keras_classifier: KerasClassifier):
        self._graph = keras_classifier.graph
        self._sess = keras_classifier.session
        self._input_placeholder = keras_classifier.input
        self._label_placeholder = keras_classifier.label
        self._preds = keras_classifier.predictions
        
        self._model = KerasModelWrapper(keras_classifier.model)
        self._loss = CrossEntropy(self._model, smoothing=self._label_smoothing)

        with self._graph.as_default():
            fgsm = FastGradientMethod(self._model, sess=self._sess)
            fgsm_params = {
                'eps': 0.3,
                'clip_min': 0.,
                'clip_max': 1.
                }
            adv_x = fgsm.generate(self._input_placeholder, **fgsm_params)

            # Consider the attack to be constant
            self._adv_x = tf.stop_gradient(adv_x)

            # model predictions for adversarial examples
            self._preds_adv = keras_classifier.model(adv_x)

        self._keras_classifier = keras_classifier  # FIXME[hack]: we need to keep a reference to the KerasClassifier to prevent the session from being closed

    def create_model(self):
        keras_classifier = KerasMnistClassifier()  # FIXME[hack]
        self.init_from_keras_classifier(keras_classifier)

    def dump_model(self):
        with self._graph.as_default():
            layer_names = self._model.get_layer_names()
            print(f"Model has {len(layer_names)} layers: {layer_names}")
            for n in layer_names:
                print(f"  {n}: {self._model.get_layer(self._input_placeholder, n)}")

            model_layers = self._model.fprop(self._input_placeholder)
            print(f"Model has {len(model_layers)} layers:")
            for n, l in model_layers.items():
                print(f"  {n}: {l}")


    def train_model(self):
        """Train the model using the current training data.
        """
        logging.info("Training Cleverhans model from scratch.")
        # FIXME[todo]: self._runner is not initialized yet!
        #self._runner.runTask(self._train_model)
        self._train_model()  # FIXME[hack]

    def _train_model(self):
        self._busy = True
        self.change('busy_changed')

        def evaluate():
            self.evaluate_model(self._x_test, self._y_test)

        # now use the cleverhans train method (this will optimize the
        # loss function, and hence the model):
        # FIXME[problem]: there seems to be no way to get some progress
        #   report from this train method. The only callback we can
        #   register is 'evaluate', which can be used for arbitrary
        #   operations, but which is only called after every epoch
        with self._graph.as_default():
            train(self._sess, self._loss, self._x_train, self._y_train,
                  evaluate=evaluate, args=self._train_params,
                  rng=self._rng)

        self._busy = False
        self.change('busy_changed')

    def evaluate_model(self, data, label):
        """Evaluate the accuracy of the MNIST model.
        """
        # use cleverhans' model_eval function:
        with self._graph.as_default():
            accuracy = model_eval(self._sess, self._input_placeholder,
                                  self._label_placeholder, self._preds,
                                  data, label, args=self._eval_params)
        print(f"MNIST model accurace: {accuracy:0.4f}")


    def load_model(self):
        if self._ckpt_path:
            with self._graph.as_default():
                saver = tf.train.Saver()
                print(self._ckpt_path)
                saver.restore(self._sess, self._ckpt_path)
                print(f"Model loaded from: {format(self._ckpt_path)}")
            self.evaluate_model(self._x_test, self._y_test)
        else:
            print("Model was not loaded.")

    def save_model(self):
        print("Model was not saved.")

    def reset_model(self):
        print("Model was not reset.")

    def load_mnist(self):
        """Load the training data (MNIST).
        """
        # Get MNIST data
        train_start, train_end = 0, 60000
        test_start, test_end = 0, 10000
        mnist = MNIST(train_start=train_start,
                      train_end=train_end,
                      test_start=test_start,
                      test_end=test_end)
        
        self._x_train, self._y_train = mnist.get_set('train')
        self._x_test, self._y_test = mnist.get_set('test')

        # Use Image Parameters
        self._img_rows, self._img_cols, self._nchannels = \
             self._x_train.shape[1:4]
        self._nb_classes = self._y_train.shape[1]

        print(f"len(train): {len(self._x_train)} / {len(self._y_train)}")
        print(f"len(test):  {len(self._x_test)} / {len(self._y_test)}")
        print(f"img_rows x img_cols x nchannels: {self._img_rows} x {self._img_cols} x {self._nchannels}")
        print(f"nb_classes: {self._nb_classes}")

    def get_example(self, index: int=None):
        if index is None:
            index = np.random.randint(len(self._x_test))
        #batch = np.arange(self._batch_size)
        batch = np.asarray([index])
        self._x_sample = self._x_train[batch]
        self._y_sample = self._y_train[batch]
        with self._graph.as_default():
            feed_dict = {self._input_placeholder: self._x_sample}
            preds_sample = \
                self._preds.eval(feed_dict=feed_dict, session=self._sess)
        
        return self._x_sample[0], self._y_sample[0], preds_sample[0]

    def get_adversarial_example(self, index: int=None):

        with self._graph.as_default():
            feed_dict = {self._input_placeholder: self._x_sample}
            x_adversarial = \
                self._adv_x.eval(feed_dict=feed_dict, session=self._sess)
            feed_dict = {self._input_placeholder: x_adversarial}
            preds_adversarial = \
                self._preds_adv.eval(feed_dict=feed_dict, session=self._sess)
        return x_adversarial[0], preds_adversarial[0]

    @property
    def busy(self):
        return self._busy

