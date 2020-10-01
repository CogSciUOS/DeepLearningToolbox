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
from datasource import Datasource
from datasource.imagenet import ImageNet
from dltb.base.data import Data
from dltb.tool.classifier import Classifier

# logging
LOG = logging.getLogger(__name__)
del logging

class Evaluator:
    class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        ENDC = '\033[0m'

    def __init__(self, classifier: Classifier = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._classifier = classifier
        self._correct = 0
        self._error = 0
        self._correct5= 0
        self._total = 0

    def accuracy(self, data: Data, top: int = None) -> float:
        # FIXME[todo]: batch processing
        scores = self._classifier.class_scores(data)
        label, confidence = self._classifier.top_classes(scores)
        rank, score = self._classifier.class_rank(scores, data.label)

        if label == data.label:
            text = (self.bcolors.OKGREEN + 'correct' + self.bcolors.ENDC +
                    f": '{label.label('text')}' (confidence={confidence:.3f})")
        elif rank < 5:
            text = (self.bcolors.OKBLUE + 'top-5' + self.bcolors.ENDC +
                    f": '{label.label('text')}' vs. '{data.label.label('text')}', "
                    f"rank={rank+1}")
        else:
            text = (self.bcolors.FAIL + 'error' + self.bcolors.ENDC +
                    f": '{label.label('text')}' vs. '{data.label.label('text')}', "
                    f"rank={rank+1}")
        print(f"{data.filename}: {text}")
        global correct, correct5, error, total
        self._correct += int(label == data.label)
        self._correct5 += rank <= 5
        self._error += int(label != data.label)
        self._total += 1
        print(f"total={self._total}, correct={self._correct}, "
              f"correct5={self._correct5}, error={self._error}, "
              f"accuracy={self._correct/self._total*100:.2f}%, "
              f"top-5 accuracy={self._correct5/self._total*100:.2f}%\r",
              end='')

    def evaluate(self, datasource: Datasource) -> None:
        # for _ in range(3):
        while True:
            try:
                data = datasource.get_random()
                if len(data.shape) != 3:
                    continue
                self.accuracy(data)
            except RuntimeError as error:
                print(f"error procesing {data.filename} {data.shape}: {error}")
                raise
            except KeyboardInterrupt:
                print(f"error procesing {data.filename} {data.shape}")
                print("Keyboard interrupt")
                break


def main():

    parser = \
        argparse.ArgumentParser(description='Deep-learning based classifiers')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate the classifier on the given datasource')
    NetworkArgparse.prepare(parser)
    parser.add_argument('image', metavar='IMAGE', nargs='*')
    args = parser.parse_args()

    network = NetworkArgparse.network(args)
    if network is None:
        print("No network was specified.")
        return

    if args.evaluate:
        evaluator = Evaluator(network)
        imagenet = ImageNet()
        imagenet.prepare()
        evaluator.evaluate(imagenet)

    else:
        # filenames = ['images/elephant.jpg', 'dog.jpg']

        for filename in args.image:
            label = network.classify(filename)
            print(f"classify('{filename}'): {label.label('text')}")

            label, score = network.classify(filename, confidence=True)
            print(f"classify('{filename}', confidence=True): "
                  f"{label.label('text'), score}")

            labels = network.classify(filename, top=5)
            print(f"classify('{filename}', top=5): "
                  f"{[label.label('text') for label in labels]}")

            labels, scores = network.classify(filename, top=5,
                                              confidence=True)
            print(f"classify('{filename}', top=5, confidence=True): ")
            for i, (label, score) in enumerate(zip(labels, scores)):
                print(f"({i+1}) {label.label('text')} ({score*100:.2f}%)")

            scores = network.class_scores(filename)
            print(f"class_scores('{filename}': {scores.shape}")

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
    for layer in network:
        print(layer.get_id())

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
