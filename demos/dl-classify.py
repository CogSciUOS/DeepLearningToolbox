#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A command line interface for running and testing classifiers.

.. moduleauthor:: Ulf Krumnack

"""

# FIXME[clean]: This code is taken from
# "models/example_tf_alexent/test_alexnet.py". Remove that file once
# this test script is working.


# standard imports
import os
import sys
import logging
import argparse

# third party imports

# toolbox imports
from network import Network
from network import argparse as NetworkArgparse
from datasource.imagenet import ImageNet
from dltb.base.data import Data
from dltb.tool.classifier import Classifier

# logging
LOG = logging.getLogger(__name__)
del logging


correct = 0
error = 0

def accuracy(classifier: Classifier, data: Data, top: int = None) -> float:
    image = classifier._image_as_batch(data.filename)
    scores = classifier.class_scores(image)
    label, confidence = classifier.top_classes(scores)
    label, confidence = label[0], confidence[0]  # FIXME[hack]: extend to batch ...
    rank, score = classifier.class_rank(scores[0], data.label)  # FIXME[hack]: extend to batch ...
    print(f"{data.filename}: "
          f"{label.label('text')} vs. {data.label.label('text')}, "
          f"rank={rank}")
    global correct, error
    correct += int(label == data.label)
    error += int(label != data.label)
    print(f"correct={correct}, error={error}, accuracy={correct/(correct+error)*100:.2f}%")


def main():

    parser = \
        argparse.ArgumentParser(description='Deep-learning based classifiers')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate the classifier on the given datasource')
    NetworkArgparse.prepare(parser)
    args = parser.parse_args()

    network = NetworkArgparse.network(args)
    if network is None:
        print("No network was specified.")
        return

    if args.evaluate:
        imagenet = ImageNet()
        imagenet.prepare()

        # for _ in range(3):
        while True:
            try:
                data = imagenet.get_random()
                if len(data.shape) != 3:
                    continue
                accuracy(network, data)
            except KeyboardInterrupt:
                print("Keyboard interrupt")
                break
    else:
        filenames = ['images/elephant.jpg', 'dog.jpg']

        for filename in filenames:
            label = network.classify_image(filename)
            print(f"{filename}: {label.label('text')}")

            label, score = network.classify_image(filename, confidence=True)
            print(f"{filename}: {label.label('text'), score}")

            labels = network.classify_image(filename, top=5)
            print(f"{filename}: {[label.label('text') for label in labels]}")

            labels, scores = network.classify_image(filename, top=5,
                                                    confidence=True)
            for i, (label, score) in enumerate(zip(labels, scores)):
                print(f"({i+1}) {label.label('text')} ({score*100:.2f}%)")

# import tensorflow as tf
# if tf.test.gpu_device_name():
#    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
# else:
#    print("Please install GPU version of TF")
# print(tf.list_devices())
# print(tf.test.is_built_with_cuda())


from datasource.files import DataFiles


def main_old():
    #
    # Load the network
    #

    network = Network.register_initialize_key('alexnet-tf')

    # Output network information
    print("network input shape: {}".format(network.get_input_shape()))
    for layer_id in network.layer_dict:
        print(layer_id)

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
        images.extend(map(network.preprocess_image, data_files))

    if 'IMAGENET_DATA' in os.environ:
        imagenet = ImageNet()
        for _ in range(3):
            images.append(network.preprocess_image(imagenet.random()))

    print(len(images))

    #
    # Classify
    #
    if len(images) > 0:
        # The following requires OpenCV to be compiled against
        # some GUI toolkit, which is not the case for a conda
        # installation.
        # for index, image in enumerate(images):
        #    cv2.imshow('image' + str(index), image)

        network.classify(images, top=5)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
