#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A command line interface for running and testing classifiers.

.. moduleauthor:: Ulf Krumnack

"""

# standard imports
import logging
import argparse

# third party imports

# toolbox imports
from datasource import Datasource
from datasource.imagenet import ImageNet
from dltb.base.data import Data
from dltb.tool.classifier import Classifier, ImageClassifier
from dltb.network import argparse as NetworkArgparse

# logging
LOG = logging.getLogger(__name__)


class Evaluator:
    """A :py:class:`Evaluator` for :py:class:`Classifier`.

    The :py:class:`Evaluator` counts corect and top-correct values,
    allowing to compute accuracy.
    """

    class Bcolors:
        # pylint: disable=too-few-public-methods
        """Escape codes for color output.
        """
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
        self._correct_top = 0
        self._total = 0

    def accuracy(self, data: Data, top: int = None) -> float:
        """Compute accuracy values for the given data and add them
        to the counters.
        """
        # FIXME[todo]: batch processing
        scores = self._classifier.class_scores(data)
        label, confidence = self._classifier.top_classes(scores, top=top)
        rank, _score = self._classifier.class_rank(scores, data.label)

        if label == data.label:
            text = (self.Bcolors.OKGREEN + 'correct' + self.Bcolors.ENDC +
                    f": '{label.label('text')}' (confidence={confidence:.3f})")
        elif top is not None and rank < top:
            text = (self.Bcolors.OKBLUE + f'top-{top}' + self.Bcolors.ENDC +
                    f": '{label.label('text')}' vs. "
                    f"'{data.label.label('text')}', "
                    f"rank={rank+1}")
        else:
            text = (self.Bcolors.FAIL + 'error' + self.Bcolors.ENDC +
                    f": '{label.label('text')}' vs. "
                    f"'{data.label.label('text')}', "
                    f"rank={rank+1}")
        print(f"{data.filename}: {text}")
        self._correct += int(label == data.label)
        self._error += int(label != data.label)
        self._total += 1
        if top is not None:
            self._correct_top += rank <= top

    def print_status(self, top: int = None, end: str = '') -> None:
        """Print the current status of this :py:class:`Evaluator`.
        """
        print(f"total={self._total}, correct={self._correct}, " +
              ('' if top is None else f"correct-{top}={self._correct_top}, ") +
              f"error={self._error}, "
              f"accuracy={self._correct/self._total*100:.2f}%, " +
              ('' if top is None else f"top-{top} accuracy="
               f"{self._correct_top/self._total*100:.2f}%") + "\r", end=end)

    def evaluate(self, datasource: Datasource, top: int) -> None:
        """Run an evaluation loop.
        """
        while True:
            try:
                data = datasource.get_random()
                if len(data.shape) != 3:
                    continue
                self.accuracy(data, top)
                self.print_status(top)
            except RuntimeError as error:
                print(f"error procesing {data.filename} {data.shape}: {error}")
                raise
            except KeyboardInterrupt:
                print(f"error procesing {data.filename} {data.shape}")
                print("Keyboard interrupt")
                self.print_status(top, end='\n')
                break


def main():
    """The main program.
    """

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
        # "laska.png", "poodle.png"
        #filename = 'images/elephant.jpg'
        #label = network.classify_image(filename)
        #print(filename, label.label('text'))
        print("{type(network).__name__} is subclass of ImageClassifier:",
              isinstance(network, ImageClassifier))

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


if __name__ == "__main__":
    main()
