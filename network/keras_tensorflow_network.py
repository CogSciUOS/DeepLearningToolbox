from keras_network import KerasNetwork
from layers import keras_tensorflow_layers
import keras
import numpy as np

from collections import OrderedDict
from frozendict import FrozenOrderedDict
from network.exceptions import ParsingError


class KerasTensorFlowNetwork(KerasNetwork):

    _layer_types_to_classes = {
        'Conv2D': keras_tensorflow_layers.KerasTensorFlowConv2D,
        'Dense': keras_tensorflow_layers.KerasTensorFlowDense,
        'MaxPooling2D': keras_tensorflow_layers.KerasTensorFlowMaxPooling2D,
        'Dropout': keras_tensorflow_layers.KerasTensorFlowDropout,
        'Flatten': keras_tensorflow_layers.KerasTensorFlowFlatten
    }

    # Layer types that encapsulate the inner product, bias add, activation pattern.
    _neural_layer_types = {'Dense', 'Conv2D'}
    # Layer types that just map input to output without trainable parameters.
    _transformation_layer_types = {'MaxPooling2D', 'Flatten', 'Dropout'}

    def __init__(self, **kwargs):
        """
        Load Keras model.
        Parameters
        ----------
        modelfile_path
            Path to the .h5 model file.
        """
        super().__init__(**kwargs)
        self._sess = keras.backend.get_session()

    def _create_layer_dict(self):

        layer_ids = self._compute_layer_ids()
        layer_dict = OrderedDict()
        last_layer_id = ''
        last_layer_type = None
        current_layers = [] # Layers that could not be wrapped in objects yet, because they were missing the activation function.
        for layer_id in layer_ids:
            keras_layer_obj = self._model.get_layer(layer_id)
            layer_type = keras_layer_obj.__class__.__name__
            # Check whether the layer type is considered a real layer.
            if layer_type in self._layer_types_to_classes.keys():
                # Check that if the layer is a neural layer it contains an activation function.
                # If so the layer can be added savely.
                # Otherwise take the next activation function and merge the layers
                if ((layer_type in self._neural_layer_types and keras_layer_obj.activation.__name__ != 'linear') or
                    (layer_type in self._transformation_layer_types)):
                    # Check that there is no unfinished layer with missing activation function.
                    # If not add the layer.
                    if not current_layers:
                        layer_dict[layer_id] = self._layer_types_to_classes[layer_type](self, [keras_layer_obj])
                    else:
                        raise ParsingError('Missing activation function.')
                else:
                    # Check that there is no other unfinished layer before that one.
                    if not current_layers:
                        current_layers.append(keras_layer_obj)
                        last_layer_id = layer_id
                        last_layer_type = layer_type
                    else:
                        raise ParsingError('Two consectutive layers with no activation function.')

            elif layer_type == 'Activation':
                # Check that there was a layer before without activation function and merge.
                if current_layers:
                    current_layers.append(keras_layer_obj)
                    layer_dict[last_layer_id] = self._layer_types_to_classes[last_layer_type](self, current_layers)
                    current_layers = []
                else:
                    raise ParsingError('Two activation layers after each other.')
            else:
                raise ParsingError('Not sure how to deal with that layer type at that position.')


        return FrozenOrderedDict(layer_dict)



    def _compute_activations(self, layer_ids: list, input_samples: np.ndarray):
        """Gives activations values of the loaded_network/model
        for a given layername and an input (inputsample).
        Parameters
        ----------
        layer_ids: The layers the activations should be fetched for.
        input_samples: Array of samples the activations should be computed for.

        Returns
        -------

        """
        activation_tensors  = [self.layer_dict[layer_id].output for layer_id in layer_ids]
        return self._feed_input(activation_tensors, input_samples)

    def _compute_net_input(self, layer_ids: list, input_samples: np.ndarray):
        ops = self._sess.graph.get_operations()
        net_input_tensors = []
        # To get the net input of a layer we take the second to last operation as the net input.
        # The last operation corresponds usually to the addition of the bias.
        for layer_id in layer_ids:
            output_op = self.layer_dict[layer_id].output.op
            net_input_op = ops[ops.index(output_op) - 2]
            # Assumes the second to last op is the matmaul / convolution operation.
            net_input_tensors.append(net_input_op.outputs[0])
            # Sanity check.
            if net_input_op.type not in {'Conv2D', 'MatMul'}:
                raise ValueError('Operation of type {} does not provided net input.'.format(net_input_op.type))
        return self._feed_input(net_input_tensors, input_samples)

    def _feed_input(self, fetches: list, input_samples: np.ndarray):
        network_input_tensor = self._sess.graph.get_operations()[0].outputs[0] # Assuming the first op is the input.
        print('input_shape', network_input_tensor.get_shape())
        return self._sess.run(fetches=fetches, feed_dict={network_input_tensor: input_samples})


