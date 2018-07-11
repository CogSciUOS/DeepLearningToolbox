from network.tensorflow import Network as TensorFlowNetwork
import tensorflow as tf
import os.path
import sys

#if tf.test.gpu_device_name():
#    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
#else:
#    print("Please install GPU version of TF")
#print(tf.list_devices())
#print(tf.test.is_built_with_cuda())

model_path = os.getenv('ALEXNET_MODEL', '.')
checkpoint = os.path.join(model_path, 'bvlc_alexnet.ckpt')

loaded_network = TensorFlowNetwork(checkpoint=checkpoint)
print("network input shape: {}".format(loaded_network.get_input_shape()))

for id in loaded_network.layer_dict:
    print(id)

#print("Operations in graph:")
#for op in loaded_network._sess.graph.get_operations():
#    print(" op: {} ({})".format(op.name, op.type))


import numpy as np
from scipy.misc import imread, imresize
# "laska.png", "poodle.png"
images = []
for arg in sys.argv[1:]:
    im = (imread(arg)[:,:,:3]).astype(np.float32)
    im = im - im.mean()
    im[:, :, 0], im[:, :, 2] = im[:, :, 2], im[:, :, 0]
    images.append(im)


import os, random
prefix = os.path.join(os.getenv('IMAGENET_DATA', '.'),'train')
for i in range(3):
    img_dir = os.path.join(prefix, random.choice(os.listdir(prefix)))
    img_file = os.path.join(img_dir, random.choice(os.listdir(img_dir)))
    print(img_file)
    im = (imread(img_file)[:,:,:3]).astype(np.float32)
    im = imresize(im, (227,227))
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
            network_input_tensor = op.outputs[0];
            print("Heureka: in! ({})".format(network_input_tensor))
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
