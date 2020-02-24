# Classify input data using a network.
#

# FIXME[clean]: This code is taken from
# "models/example_tf_alexent/test_alexnet.py". Remove that file once
# this test script is working.


# FIXME[todo]: we need the following in the PYTHONPATH
#  - the project root dir (to load TensorFlowNetwork from the network package)

# FIXME[todo]: we should allow for other networks than the hard coded
# AlexNet!

# FIXME[concept]: there should be a connection between network
#   and suitable datasets/labels, i.e., AlexNet should provide
#   class labels, even if not applied to ImageNet data

import os
import sys

from network.tensorflow import Network as TensorFlowNetwork
from network.loader import load_alexnet

# import tensorflow as tf
#if tf.test.gpu_device_name():
#    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
#else:
#    print("Please install GPU version of TF")
#print(tf.list_devices())
#print(tf.test.is_built_with_cuda())

from datasources import DataFiles, ImageNet


# FIXME[hack]: instead of prepare_input_image use the network.resize
# API once it is finished!
import numpy as np
from util.image import imresize

def prepare_input_image(input_data):
    im = (input_data[0][:,:,:3]).astype(np.float32)
    im = imresize(im, (227,227)) # FIXME[hack]: ImageNet/AlexNet!
    im = im - im.mean()
    im[:, :, 0], im[:, :, 2] = im[:, :, 2], im[:, :, 0]
    return im

#import cv2

def main():

    #
    # Load the network
    #

    network = load_alexnet()

    # Output network information
    print("network input shape: {}".format(network.get_input_shape()))
    for id in network.layer_dict:
        print(id)

    print(type(network._input_placeholder))
    import tensorflow as tf
    tensor = tf.get_default_graph().get_tensor_by_name('xw_plus_b:0')
    print("tensor is ", tensor)
    print("tensor[0] is ", tensor[0])


    #
    # Load input data
    #
    images = []
    if len(sys.argv) > 1:
        # "laska.png", "poodle.png"
        data_files = DataFiles(sys.argv[1:])
        images.extend(map(prepare_input_image, data_files))

    if 'IMAGENET_DATA' in os.environ:
        imagenet = ImageNet()
        for i in range(3):
            images.append(prepare_input_image(imagenet.random()))

    print(len(images))


    #
    # Classify
    #
    if len(images) > 0:
        # The following requires OpenCV to be compiled against
        # some GUI toolkit, which is not the case for a conda
        # installation.
        #for index, image in enumerate(images):
        #    cv2.imshow('image' + str(index), image)

        network.classify_top_n(images, 5)

        #cv2.waitKey(0)
        #cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
