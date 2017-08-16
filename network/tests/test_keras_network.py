from unittest import TestCase
from network.keras_network import KerasNetwork
import numpy as np

class TestKerasNetwork(TestCase):


    def setUp(self):
        self.network = KerasNetwork('22x2_model.h5')

    def test_get_layer_id_list(self):
        self.assertEqual(self.network.get_layer_id_list(),
                         ['conv2d_22',
                          'conv2d_23',
                          'max_pooling2d_7',
                          'conv2d_24',
                          'max_pooling2d_8',
                          'dropout_13',
                          'flatten_8',
                          'dense_14',
                          'activation_14',
                          'dropout_14',
                          'dense_15',
                          'activation_15'])

    def test_get_layer_input_shape(self):
        self.assertEqual((None, 60, 60, 32), self.network.get_layer_input_shape('conv2d_23'))

    def test_get_layer_output_shape(self):
        self.assertEqual((None, 56, 56, 32), self.network.get_layer_output_shape('conv2d_23'))

    def test_get_activations(self):
        images = np.load('2x2data.npy')
        #TODO: don't assume tensorflow backend for image format
        self.assertTrue(
            (
            self.network._model.predict(images[0:1, :, :, np.newaxis]) ==
            self.network.get_activations('activation_15', images[0:1, :, :])
            ).all()
        )

    def test_get_layer_weights(self):
        self.assertTrue(
            (self.network._model.get_weights()[2] ==
             self.network.get_layer_weights('conv2d_23')).all()
        )