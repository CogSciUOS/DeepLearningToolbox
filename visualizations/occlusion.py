import numpy as np

def get_occlussion_map(self, input_sample: np.ndarray, kernel_size: int) -> np.ndarray:
    """gives a heatmap of oclussion algorithm for maskes defined by kernel_shape

    """
    input_shape = input_sample.shape
    heatmap = np.empty([input_shape[1] * input_shape[2]])

    padded_input = np.lib.pad(input_sample,
                              pad_width=((0, 0), (kernel_size, kernel_size), (kernel_size, kernel_size), (0, 0)),
                              mode='constant')

    maskmulti = np.ones(
        [input_shape[1] * input_shape[2], input_shape[1] + (kernel_size) * 2, input_shape[2] + (kernel_size) * 2, 1])
    maskadd = np.zeros(
        [input_shape[1] * input_shape[2], input_shape[1] + (kernel_size) * 2, input_shape[2] + (kernel_size) * 2, 1])

    for i in range(input_shape[1] * input_shape[2]):
        maskmulti[i,
        i // input_shape[2] + kernel_size - (kernel_size):i // input_shape[2] + kernel_size + (kernel_size + 1),
        i % input_shape[2] + kernel_size - (kernel_size):i % input_shape[2] + kernel_size + (kernel_size + 1), 0] = 0
        maskadd[i,
        i // input_shape[2] + kernel_size - (kernel_size):i // input_shape[2] + kernel_size + (kernel_size + 1),
        i % input_shape[2] + kernel_size - (kernel_size):i % input_shape[2] + kernel_size + (kernel_size + 1), 0] = 1
    occluded_list = np.multiply(padded_input, maskmulti) + maskadd

    heatmap = self.get_activations(occluded_list[:, kernel_size:-kernel_size, kernel_size:-kernel_size],
                                   [list(self.layer_dict.keys())[-3]])
    # print(np.asarray(heatmap).shape)
    # print(self.get_layer_weights(self.layer_dict.keys()[-1])[0].shape)


    heatmap = np.dot(np.asarray(heatmap)[0, :, :], self.get_layer_weights(self.layer_ids[-1]))[:,
              np.argmax(self._model.predict(input_sample))]
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / (np.max(heatmap) - np.min(heatmap))

    heatmap = heatmap * 255
    heatmap = (255 - heatmap).astype(np.uint8)
    # heatmap=((heatmap-np.min(heatmap))*255/(np.max(heatmap)-np.min(heatmap))).astype(int)

    return np.reshape(heatmap, input_shape, order='A')
