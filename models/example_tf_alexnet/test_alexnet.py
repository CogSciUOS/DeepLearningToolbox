from network.tensorflow import Network as TensorFlowNetwork
import tensorflow as tf
import os.path
import sys

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
#print(tf.list_devices())
print(tf.test.is_built_with_cuda())

checkpoint = 'bvlc_alexnet.ckpt'
checkpoint = os.path.join('models', 'example_tf_alexnet',
                           'bvlc_alexnet.ckpt')
loaded_network = TensorFlowNetwork(checkpoint=checkpoint)

#print(loaded_network._sess.graph.get_operations())
print(loaded_network.layer_dict)

import numpy as np
from scipy.misc import imread
# "laska.png", "poodle.png"
images = []
for arg in sys.argv[1:]:
    im = (imread(arg)[:,:,:3]).astype(np.float32)
    im = im - im.mean()
    im[:, :, 0], im[:, :, 2] = im[:, :, 2], im[:, :, 0]
    images.append(im)

# Assuming the first op is the input.
network_input_tensor = loaded_network._sess.graph.get_operations()[0].outputs[0]
network_output_tensor = loaded_network.layer_dict['dense_3'].activation_tensor

from imagenet_classes import class_names


if images:
    in_op = None
    out_op = None
    for op in loaded_network._sess.graph.get_operations():
        #print(op.type)
        if op.type == 'Placeholder':
            in_op = op;
            print("Heureka: in!")
        if op.type == 'Softmax':
            out_op = op;
            print("Heureka: out!")
            break

    if out_op:
        output = loaded_network._sess.run(network_output_tensor, feed_dict = {network_input_tensor:images})


    for input_im_ind in range(output.shape[0]):
        inds = np.argsort(output)[input_im_ind,:]
        print("Image", input_im_ind)
        for i in range(5):
            print("  {}: {} ({})".format(i,class_names[inds[-1-i]],
                                         output[input_im_ind, inds[-1-i]]))
