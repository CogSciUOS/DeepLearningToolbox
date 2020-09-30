# standard imports
from typing import List
from collections import OrderedDict

# third party imports
import numpy as np
import caffe
from caffe.proto import caffe_pb2

import google.protobuf.text_format
from tempfile import NamedTemporaryFile

# toolbox imports
from network.exceptions import ParsingError

from . import Network as BaseNetwork
from .layers import caffe_layers


class Network(BaseNetwork):

    _LAYER_TYPES_TO_IGNORE = {
        'Data',
        'Input'
    }

    _LAYER_TYPES = {
        'activation_functions': {'ReLU', 'PReLU', 'ELU', 'Sigmoid', 'TanH', 'Softmax'},
        'net_input_layers': {'Convolution', 'InnerProduct'},
        'input': {'Data', 'Input'}
    }

    _LAYER_TYPES_TO_CLASSES = {
        'Convolution': caffe_layers.CaffeConv2D,
        'InnerProduct': caffe_layers.CaffeDense,
        'Pooling': caffe_layers.CaffeMaxPooling2D,
        'Dropout': caffe_layers.CaffeDropout,
        'Flatten': caffe_layers.CaffeFlatten
    }

    _TRANSFORMATION_LAYER_TYPES = {'Pooling', 'Flatten', 'Dropout'}


    def __init__(self, **kwargs):
        """
        Load Caffe model.
        Parameters
        ----------
        **kwargs
            model_def
                Path to the .prototxt model definition file.
            model_weights
                Path the .caffemodel weights file.
        """
        # Caffe uses channels first as data format.
        kwargs['data_format'] = 'channels_first'
        self.protonet = self._remove_inplace(kwargs['model_def'])
        # Write the new protobuf model definition to a file, so it can be read to create a caffe net.
        # This would probably not be necessary with boost > 1.58, see https://stackoverflow.com/a/34172374/4873972
        with NamedTemporaryFile(mode='w', delete=True) as fp:
            fp.write(str(self.protonet))
            fp.seek(0)
            self._caffenet = caffe.Net(fp.name,
                                       kwargs['model_weights'],
                                       caffe.TEST)
        self.layer_dict = self._create_layer_dict()

        super().__init__(**kwargs)


    def _create_layer_dict(self):
        layer_dict = OrderedDict()

        num_flatten_layers = 0

        # Make sure that the first layer after input is a net input layer.
        if (self.protonet.layer[0].type in self._LAYER_TYPES['input'] and
            self.protonet.layer[1].type in self._LAYER_TYPES['net_input_layers']):

            last_caffe_layer = self._caffenet.layers[1]
            last_proto_layer = self.protonet.layer[1]
        else:
            raise ParsingError('First layer is not net input layer.')


        for caffe_layer, proto_layer in zip(self._caffenet.layers, self.protonet.layer):

            # Check whether the layer is considered a layer or just input.
            if caffe_layer.type in self._LAYER_TYPES_TO_IGNORE:
                continue

            # Check that if we have a Pooling layer, we are really dealing with a max pooling layer.
            # Allow no other types of pooling right now.
            if caffe_layer.type == 'Pooling':
                if not proto_layer.pooling_param.MAX == 0:
                    raise ParsingError('Only max pooling is allowed, but was not used in layer {}'.format(str(proto_layer)))
            # Check whether the layer has a separate activation function, as to whether net input can
            # be computed or not.
            if caffe_layer.type in self._TRANSFORMATION_LAYER_TYPES:
                # Transformation layers are just by themselves, they can be added right away.
                layer_dict[proto_layer.name] = self._LAYER_TYPES_TO_CLASSES[caffe_layer.type](
                    self, caffe_layer, proto_layer
                )
            elif caffe_layer.type in self._LAYER_TYPES['net_input_layers']:
                # Check whether there occurs an implicit flatten operation between two layers and
                # and artificial flatten layer has to be added.
                # That is the case if we now have a dense layer and the previous layer still had NCHW dimensions.
                if (caffe_layer.type == 'InnerProduct' and
                    len(self._caffenet.blobs[last_proto_layer.name].data.shape) > 2):
                    # Keep a count of the number of added Flatten layers, to name them properly.
                    num_flatten_layers += 1
                    layer_dict['flatten_' + str(num_flatten_layers)] = self._LAYER_TYPES_TO_CLASSES['Flatten'](
                        self, last_caffe_layer, last_proto_layer, caffe_layer, proto_layer
                    )
                # Conv and Dense layers should wait for there activation function.
                # If however the last layer was a net input layer as well it needs to be added now.
                # The activation in this case would be the layer itself.
                if last_caffe_layer.type in self._LAYER_TYPES['net_input_layers']:
                    layer_dict[last_proto_layer.name] = self._LAYER_TYPES_TO_CLASSES[last_caffe_layer.type](
                        self, last_caffe_layer, last_proto_layer, last_caffe_layer, last_proto_layer
                    )
            elif caffe_layer.type in self._LAYER_TYPES['activation_functions']:
                # If we are dealing with an activation function, last layer before needs to be a net input layer.
                if last_caffe_layer.type in self._LAYER_TYPES['net_input_layers']:
                    layer_dict[last_proto_layer.name] = self._LAYER_TYPES_TO_CLASSES[last_caffe_layer.type](
                        self, last_caffe_layer, last_proto_layer, caffe_layer, proto_layer,
                    )
                else:
                    raise ParsingError('Activation function not after conv or dense layer.')



            # Save the values of the current iteration to compare with the next.
            last_caffe_layer = caffe_layer
            last_proto_layer = proto_layer

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

        activation_layers = []
        # Try to find the activation layer for each layer id. If it cannot be found the layer only has one output.
        for layer_id in layer_ids:
            try:
                activation_layers.append(self.layer_dict[layer_id].activation_layer_name)
            except AttributeError:
                activation_layers.append(self.layer_dict[layer_id].layer_name)

            # maybe_activation_layer = getattr(self.layer_dict[layer_id], 'activation_layer_name', None)
            # if maybe_activation_layer is not None:
            #     activation_layers.append(maybe_activation_layer)
            # else:
            #     activation_layers.append(self.layer_dict[layer_id].layer_name)


        return self._feed_input(activation_layers, input_samples)

    def _compute_net_input(self, layer_ids: list, input_samples: np.ndarray):
        # Use the default output as this corresponds to the net input for neural layers
        net_input_layers = [self.layer_dict[layer_id].layer_name for layer_id in layer_ids]
        return self._feed_input(net_input_layers, input_samples)



    def _feed_input(self, fetches: list, input_samples: np.ndarray) -> List[np.ndarray]:
        # Assuming the first layer is the input layer.
        input_blob = next(iter(self._caffenet.blobs.values()))
        # Reshape the loaded_network. Change only the batch size.
        # The batch size is otherwise fixed in the model definition.
        old_input_shape = input_blob.data.shape
        new_input_shape = list(old_input_shape)
        new_input_shape[0] = input_samples.shape[0]
        input_blob.reshape(*new_input_shape)
        self._caffenet.reshape()
        # Feed the input into the loaded_network and forward it.
        input_blob.data[...] = input_samples
        self._caffenet.forward()

        outputs = [self._caffenet.blobs[fetch].data for fetch in fetches]
        return outputs

    def _remove_inplace(self, model_def):
        """Remove inplace operations from caffe protobuf model definition, so net input
        and activations and be retrieved separately.

        Parameters
        ----------
        model_def: Path to the model definition.

        Returns
        -------

        """
        protonet = caffe_pb2.NetParameter()
        with open(model_def, 'r') as fp:
            google.protobuf.text_format.Parse(str(fp.read()), protonet)

        replaced_tops = {}
        for layer in protonet.layer:
            # Check whether bottoms were renamed.
            for i in range(len(layer.bottom)):
                if layer.bottom[i] in replaced_tops.keys():
                    layer.bottom[i] = replaced_tops[layer.bottom[i]]

            if layer.bottom == layer.top:
                for i in range(len(layer.top)):
                    # Retain the mapping from the old to the new layer_name.
                    new_top = layer.top[i] + '_' + layer.name
                    replaced_tops[layer.top[i]] = new_top
                    # Redefine layer.top
                    layer.top[i] = new_top

        return protonet

