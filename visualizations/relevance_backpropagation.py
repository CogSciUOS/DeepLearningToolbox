from network.network import Network as BaseNetwork
import numpy as np

class RelevanceBackpropagation:
    """
    Relevance backpropagation limited to the first maxpool layer.
    """
    def __init__(self, network: BaseNetwork):
        self._network = network

    def visualize(self, input_sample, eps=1) -> np.ndarray:
        last_layer_id = self._network.get_layer_id_list()[-1]
        activations = []
        for idx, layer_id in reversed(list(enumerate(self._network.get_layer_id_list()))):
            activations.append(self._network.get_activations(layer_id, input_sample))

        return self._compute_relevance_heatmap(activations)



    def _compute_relevance_heatmap(self, activations, relevance):

        relevances = [activations[0]]
        print(reversed(list(enumerate(self._network.get_layer_id_list()))))
        for idx, layer in reversed(list(enumerate(self._model.layers))):

            # print("Propagating relevance backwards from layer {} to layer {} ({}).".format(idx+1,idx,type(layer)))

            if isinstance(layer, Dense):
                # Dense layer (stores weights from previous layer to this layer)
                W, B = layer.get_weights()

                # compute local pre-activation matrix,
                # i.e. z_{ij} in [1], formula (50)
                local_pre_activation = activations[idx-1][:, np.newaxis] * W

                # compute global pre-activation vector (that is the net input
                # to activation function), i.e. z_{j} in [1], formula (51)
                global_pre_activation = local_pre_activation.sum(0) + B

                # Check: compare this to
                # net_input = np.dot(activation[0],W) + B

                # compute the relevance matrix
                print(local_pre_activation.shape)
                print(global_pre_activation.shape)

                relevance_matrix = local_pre_activation / (global_pre_activation + eps) * relevances[0]

                # compute new relevance vector for previous layer
                relevance = relevance_matrix.sum(1)

            elif isinstance(layer, Conv2D):
                # 2D convolution. Compute a "backward convolution"

                # layer.input_shape[1:] is (channels, nb_row, nb_col)
                relevance = np.zeros(layer.input_shape[1:])

                # Get weights and bias values of the kernels.
                # weights are a 4-tuple:
                #   (nb_filter, channels, filter_row, filter_col)
                #   nb_filter is depth in output, channels is depth in input
                # biases are just a vector (nb_filter,)
                #   i.e. one bias per filter
                W, B = layer.get_weights()

                # get activation in previous layer
                # Shape: (channels, nb_row, nb_col) like input_shape
                activation = activations[idx-1]

                # Now loop through all positions in the output array and
                # consider the "receptive field" for each position (and all
                # filters).  For this sub-problem perform the standard (fully
                # connected) method to compute the relevance values (that is,
                # relevance of output pixel in position (r,c) will only
                # contribute to the relevance in its receptive field).

                # FIXME[todo]: care for the border_mode ('valid', 'same' or 'full')
                # For now we simply ignore boundary pixels
                if layer.padding != 'valid':
                    print("warning: Ignoring the border_mode ({}) - may get implemented in the future ...".format(
                        layer.border_mode), file=sys.stderr)

                # layer.output_shape[2:] is (out_rows, out_cols)

                for r, c in np.ndindex(layer.output_shape[1:3]):
                    # print(layer.output_shape)


                    # compute local pre-activation matrix:
                    # extract activation for the receptive field
                    # from the full activation matrix
                    # (and add one dimension to enumerate the filters).
                    # Multiply with connections weights.
                    # Shape: (nb_filter, channel, filter_row, filter_col)
                    # local_pre_activation = activation[np.newaxis,:,r:r+layer.nb_row,c:c+layer.nb_col] * W   for the theano
                    print(W.shape)
                    print(activation[r:r + layer.kernel_size[0], c:c + layer.kernel_size[1], np.newaxis, :].shape)
                    print(local_pre_activation.shape)
                    local_pre_activation = activation[r:r + layer.kernel_size[0], c:c + layer.kernel_size[1], :,
                                           np.newaxis] * W

                    # compute the total net activation for each filter by
                    # summing up over channels (1), rows (2), and columns (3)
                    # and adding the bias.
                    # Shape: (nb_filter,)
                    # global_pre_activation = local_pre_activation.sum((1,2,3)) + B
                    global_pre_activation = local_pre_activation.sum((0, 1, 2)) + B

                    # a local relevance matrix for the current filter and
                    # position providing relevance values for all pixels
                    # that may get relevance from the current output pixel
                    # (its "receptive field").
                    # Shape: (nb_filter, channel, filter_row, filter_col)
                    # print(local_pre_activation.shape)
                    # print(relevances[0][r,c,:].shape)
                    # print(global_pre_activation.shape)
                    # relevance_matrix = local_pre_activation * (relevances[0][r,c,:]/(global_pre_activation+eps))[:,np.newaxis,np.newaxis,np.newaxis]
                    relevance_matrix = local_pre_activation * (relevances[0][r, c, :] / (global_pre_activation + eps))[
                                                              np.newaxis, np.newaxis, np.newaxis, :]

                    # update relevance values for the "receptive field", by
                    # summing the relevance up the relevances of all filters:
                    # print(relevance_matrix.shape)
                    # print(relevance.shape)
                    relevance[r:r + layer.kernel_size[0], c:c + layer.kernel_size[1], :] += relevance_matrix.sum(3)



            elif isinstance(layer, Flatten):
                # Flatten changes the shape of the node array. This does not
                # affect the relevance values. We just have to adapt their
                # shape.
                relevance = relevances[0].reshape(layer.input_shape[1:])
            elif isinstance(layer, MaxPooling2D):
                print(idx-1)
                print(layer)
                relevance = relevances[0]
                break
            elif isinstance(layer, Dropout):
                # Dropout does not change the output shape and should not
                # affect the relevance vmodel = load_model(model_file)alue. Hence we simply keep the old
                # relevance value.
                relevance = relevances[0]

            else:
                # Other types of layer are not supported yet.
                print("error: Layers of type {} are not supported yet. Sorry!".format(type(layer)), file=sys.stderr)
                sys.exit(1)

            relevances.insert(0, relevance)

        return relevances
