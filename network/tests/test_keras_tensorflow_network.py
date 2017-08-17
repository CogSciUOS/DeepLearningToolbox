import keras.backend as K
from unittest import TestCase
from network import KerasTensorFlowNetwork
import numpy as np
from keras.datasets import mnist


class TestKerasTensorFlowNetwork(TestCase):


    def setUp(self):
        self.network = KerasTensorFlowNetwork('../../models/example_keras_mnist_model_with_dropout.h5')
        self.data = mnist.load_data()[1][0]

    def test_get_layer_id_list(self):
        self.assertEqual(self.network.layer_ids,
                         ['conv2d_1',
                          'max_pooling2d_1',
                          'conv2d_2',
                          'dropout_1',
                          'flatten_1',
                          'dense_1',
                          'dropout_2',
                          'dense_2'])

    def test_get_layer_input_shape(self):
        self.assertEqual((None, 13, 13, 32) , self.network.get_layer_input_shape('conv2d_2'))

    def test_get_layer_output_shape(self):
        self.assertEqual((None, 11, 11, 32), self.network.get_layer_output_shape('conv2d_2'))

    def test_get_activations(self):
        input_image = self.data[0:1, :, :, np.newaxis]
        self.assertTrue(
            (
            self.network._model.predict(input_image) ==
            self.network.get_activations(['dense_2'], input_image)
            ).all()
        )

    def test_get_layer_weights(self):
        self.assertTrue(
            (self.network._model.get_weights()[2] ==
             self.network.get_layer_weights('conv2d_2')[0]).all()
        )