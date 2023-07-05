import os

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

from dltb.network import Classifier
from dltb.thirdparty.keras.tensorflow import Network as KerasNetwork
from dltb.thirdparty.keras.layer import conv_2d


class KerasMnistClassifier(KerasNetwork, Classifier):
    """A simple feed forward network implemented in Keras.

    Attributes
    ----------
    _inputs
    _outputs
   
    _epochs
    _batch_size
    
    _weights_file

    """

    def __init__(self, **kwargs):
        """Construct a new, convolutional network.
        """
        logger.info(f"New Keras MNIST model")
        super().__init__(scheme=10, **kwargs)

    def _prepare(self) -> None:
        super()._prepare()

        # FIXME[hack]: why are we doing this here?
        # we get an error as the graph is empty:
        # RuntimeError: The Session graph is empty.
        #   Add operations to the graph before calling run().
        self.snapshot()       
        for l in self._snapshot:
            print(l.shape)
        self._model.summary(print_fn=logger.info)

    def _prepare_graph(self):
        super()._prepare_graph()
        with self._graph.as_default():
            self._create_model()
       
            
    def _create_model(self, img_rows=28, img_cols=28, nchannels=1,
                      nb_classes=10):
        # Create TF session and set as Keras backend session
        #self._sess = tf.Session()
        #keras.backend.set_session(self._sess)
        print("\n\nKerasMnistClassifier: create_model\n\n")
        
        # Define input TF placeholder
        input_shape = (None, img_rows, img_cols, nchannels)
        label_shape = (None, nb_classes)
        self._input_placeholder = tf.placeholder(tf.float32, shape=input_shape)
        self._label_placeholder = tf.placeholder(tf.float32, shape=label_shape)

        # img_rows: number of row in the image (e.g. 28 for MNIST)
        # img_cols: number of columns in the image (e.g. 28 for MNIST)
        # channels: number of color channels (e.g., 1 for MNIST)
        # nb_filters: number of convolutional filters per layer
        nb_filters = 64
        # nb_classes: the number of output classes (e.g. 10 for MNIST)

        self._model = Sequential()

        # Define the layers successively 
        layers = [
            conv_2d(nb_filters, (8, 8), (2, 2), "same",
                    input_shape=input_shape[1:]),
            Activation('relu'),
            conv_2d((nb_filters * 2), (6, 6), (2, 2), "valid"),
            Activation('relu'),
            conv_2d((nb_filters * 2), (5, 5), (1, 1), "valid"),
            Activation('relu'),
            Flatten(),
            Dense(nb_classes)
        ]

        for layer in layers:
            self._model.add(layer)

        self._logits_tensor = self._model(self._input_placeholder)

        self._model.add(Activation('softmax'))
        self._predictions = self._model(self._input_placeholder)
