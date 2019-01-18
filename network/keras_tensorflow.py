from __future__ import absolute_import

import numpy as np

import sys
import os

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


from collections import OrderedDict
from frozendict import FrozenOrderedDict
from network.exceptions import ParsingError

from .keras import Network as KerasNetwork
from .layers import keras_tensorflow_layers

class Network(KerasNetwork):

    _layer_types_to_classes = {
        'Conv2D': keras_tensorflow_layers.KerasTensorFlowConv2D,
        'Dense': keras_tensorflow_layers.KerasTensorFlowDense,
        'MaxPooling2D': keras_tensorflow_layers.KerasTensorFlowMaxPooling2D,
        'Dropout': keras_tensorflow_layers.KerasTensorFlowDropout,
        'Flatten': keras_tensorflow_layers.KerasTensorFlowFlatten
    }

    # Layer types that encapsulate the inner product, bias add,
    # activation pattern.
    _neural_layer_types = {'Dense', 'Conv2D'}

    # Layer types that just map input to output without trainable
    # parameters.
    _transformation_layer_types = {'MaxPooling2D', 'Flatten', 'Dropout'}


    @classmethod
    def import_framework(cls, cpu=True):
        # The only way to configure the keras backend appears to be
        # via environment variable. We thus inject one for this
        # process. Keras must be loaded after this is done
        os.environ['KERAS_BACKEND'] = 'tensorflow'

        # TF_CPP_MIN_LOG_LEVEL: Control the amount of TensorFlow log
        # message displayed on the console.
        #  0 = INFO
        #  1 = WARNING
        #  2 = ERROR
        #  3 = FATAL
        #  4 = NUM_SEVERITIES
        # Defaults to 0, so all logs are shown.
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

        # TF_CPP_MIN_VLOG_LEVEL brings in extra debugging information
        # and actually works the other way round: its default value is
        # 0 and as it increases, more debugging messages are logged
        # in.
        # Remark: VLOG messages are actually always logged at the INFO
        # log level. It means that in any case, you need a
        # TF_CPP_MIN_LOG_LEVEL of 0 to see any VLOG message.
        #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        
        if cpu:
            # unless we do this, TF still checks and finds gpus (not
            # sure if it actually uses them)
            #
            # UPDATE: TF now still loads CUDA, there seems to be no
            # way around this
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

            # FIXME[todo]: the following setting causes a TensorFlow
            # error: failed call to cuInit: CUDA_ERROR_NO_DEVICE
            #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            logger.info("Running in CPU-only mode.")
            import tensorflow as tf
            from multiprocessing import cpu_count
            num_cpus = cpu_count()
            config = tf.ConfigProto(intra_op_parallelism_threads=num_cpus,
                                    inter_op_parallelism_threads=num_cpus,
                                    allow_soft_placement=True,
                                    device_count={'CPU': num_cpus, 'GPU': 0})
            session = tf.Session(config=config)

        global K
        from keras import backend as K
        # image_dim_ordering:
        #   'tf' - "tensorflow":
        #   'th' - "theano":
        K.set_image_dim_ordering('tf')
        logger.info(f"image_dim_ordering: {K.image_dim_ordering()}")
        logger.info(f"image_data_format: {K.image_data_format()}")

        if cpu:
            K.set_session(session)

        super(Network, cls).import_framework()


    def __init__(self, **kwargs):
        """
        Load Keras model.
        Parameters
        ----------
        modelfile_path
            Path to the .h5 model file.
        """
        super().__init__(**kwargs)
        self._sess = K.get_session()

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
        # To get the net input of a layer we take the second to last
        # operation as the net input.  This operation corresponds
        # usually to the addition of the bias.
        for layer_id in layer_ids:
            output_op = self.layer_dict[layer_id].output.op
            # Assumes the input to the activation is the net input.
            net_input_tensors.append(output_op.inputs[0])
        return self._feed_input(net_input_tensors, input_samples)

    def _feed_input(self, fetches: list, input_samples: np.ndarray):
        network_input_tensor = self._model.layers[0].input
        return self._sess.run(fetches=fetches, feed_dict={network_input_tensor: input_samples})

