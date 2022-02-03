#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A command line interface for running and testing classifiers.

.. moduleauthor:: Ulf Krumnack


Examples
--------

Classify an image (examples/cat.jpg) using the DenseNet image classifier.

   python demos/dl-classify.py --densenet examples/cat.jpg

Evaluate AlexNet image classifier (on the ImageNet validation set):

   python demos/dl-classify.py --densenet --evaluate

"""

# standard imports
from typing import Optional, Iterable
import logging
import argparse
import json
import csv

# third party imports
import numpy as np

# toolbox imports
import dltb.argparse as ToolboxArgparse
from dltb.base.data import Data
from dltb.tool.classifier import Classifier, SoftClassifier, ImageClassifier
from dltb.network import argparse as NetworkArgparse, Network
from dltb.datasource import Datasource
from dltb.util.terminal import Terminal
from dltb.thirdparty.datasource.imagenet import ImageNet

# logging
LOG = logging.getLogger(__name__)


class Evaluator:
    """A :py:class:`Evaluator` for :py:class:`Classifier`.

    The :py:class:`Evaluator` counts correct and top-correct values,
    allowing to compute accuracy.

    Storing results
    ---------------
    An :py:class:`Evaluator` can store results in different forms:
    * just summary metrics:
    * the confusion matrix
    * a detailed report of each probed datapoint:

    The detailed report may include an identifier for the datapoint
    (like a filename in case of a file datasource or the index in an
    :py:class:`Indexed` datasource).
    It may then include the correct class (label) besides the predicted
    class.  For a soft classifier, this may instead report the top-n
    predicted classes optionally with confidence score and/or probit
    value.

    """

    # FIXME[todo]: not only store accuracy, but also detailed results
    # that allow
    # - to compute a confusion matrix
    # - to identify "hard" data in a datasource
    # - allow to store data in a database

    def __init__(self, classifier: Classifier = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._classifier = classifier
        self._correct = 0
        self._error = 0
        self._correct_top = 0
        self._total = 0

        self._top = None

        number_of_classes = len(classifier.scheme)
        self._confusion = np.zeros((number_of_classes, number_of_classes),
                                   dtype=np.int32)

        self._result_log = []

    def evaluate(self, datasource: Datasource, top: Optional[int] = None,
                 terminal: Optional[Terminal] = None) -> None:
        """Run an evaluation loop.

        Arguments
        ---------
        datasource:
            The datasource providing (labeled) data for the evaluation.
        top:
            Compute the top-n accuracy in addition top-1 accuracy
        terminal:
            A terminal to report results.
        """
        self._top = top
        self._terminal = terminal

        while True:
            try:
                data = datasource.get_random()
                self += data
                self.output_status(top)
            except RuntimeError as error:
                print(f"error procesing {data.filename} {data.shape}: {error}")
                raise
            except KeyboardInterrupt:
                print(f"error procesing {data.filename} {data.shape}")
                print("Keyboard interrupt")
                self.output_status(top, end='\n')
                self._write_result_log()
                break

    def __iadd__(self, data: Data) -> 'Evaluator':
        """Add a new data point to this evaluator. 

        Arguments
        ---------
        data: Data
            The Data for which to compute the accuracy.

        """
        # FIXME[todo]: batch processing
        # FIXME[todo]: hard classifier (no scores))
        scores = self._classifier.class_scores(data)
        label, confidence = self._classifier.top_classes(scores, top=self._top)
        rank, _score = self._classifier.class_rank(scores, data.label)

        #
        # Update the counters
        #

        # label0: the predicted label
        label0 = label if self._top is None else label[0]
        self._correct += int(label0 == data.label)
        self._error += int(label0 != data.label)
        self._total += 1
        if self._top is not None:
            self._correct_top += rank <= self._top

        self._confusion[data.label, label0] += 1

        #
        # record the results
        #

        # FIXME[todo]: what should be recorded:
        #  - prediction
        #  - top predictions
        #  - top predicitons with scores
        #  - all scores
        record = (data.filename, data.label, label0)
        self._result_log.append(record)

        #
        # show the results
        #
        if self._terminal is not None:
            self._show_result(data, label, confidence, rank)

        return self

    def _show_result(self, data, label, confidence, rank) -> None:
        label0 = label if self._top is None else label[0]
        if label0 == data.label:
            # correct: data label (ground truuth) is equal to the
            # predicted label
            confidence0 = confidence if self._top is None else confidence[0]
            text = (self._terminal.status('correct', status='ok') +
                    f": '{label0['text']}' "
                    f"(confidence={confidence0:.3f})")
        elif self._top is not None and rank < self._top:
            # top-n correct: ground truth is within the top-n predictions
            # (only applies if a top values was provided)
            text = (self._terminal.status(f'top-{self._top}', status='ok2') +
                    f": '{label[rank]['text']}' vs. "
                    f"'{data.label['text']}', "
                    f"rank={rank+1}"
                    f"(confidence={confidence[rank]:.3f})")
        else:
            text = (self._terminal.status('error', status='fail') +
                    f": '{label0['text']}' vs. "
                    f"'{data.label['text']}', "
                    f"rank={rank+1}")
        self._terminal.output(f"{data.filename}: {text}")

    def output_status(self, top: int = None, end: str = '') -> None:
        """Print the current status of this :py:class:`Evaluator`.
        """
        status = (f"total={self._total}, correct={self._correct}, " +
                  ('' if top is None else
                   f"correct-{top}={self._correct_top}, ") +
                  f"error={self._error}, "
                  f"accuracy={self._correct/self._total*100:.2f}%" +
                  ('' if top is None else f", top-{top} accuracy="
                   f"{self._correct_top/self._total*100:.2f}%"))
        print(status + "\r", end=end)

    def _write_result_log(self) -> None:
        """Write the result log to the output file.
        """
        if False:  # json
            filename = "classification.json"
            print(f"Writing results to '{filename}' ...", end='')
            with open(filename, 'w') as outfile:
                json.dump(self._result_log, outfile, indent=4)
            print(" done.")

        if True:  # json
            filename = "classification.csv"
            print(f"Writing results to '{filename}' ...", end='')
            with open(filename, 'w') as outfile:
                csvwriter = csv.writer(outfile, delimiter=',')
                for record in self._result_log:
                    csvwriter.writerow(record)
            print(" done.")

    def _iterate_result_log(self) -> Iterable:
        """Iterate over the result log.
        """

        if False:  # json
            filename = "classification.json"
            with open(filename, 'w') as infile:
                records = json.load(infile)
            for record in records:
                yield record

        if True:  # csv
            filename = "classification.csv"
            with open(filename, 'w') as infile:
                csvreader = csv.reader(infile, delimiter=',')
                for record in csvreader:
                    yield record


def main():
    """The main program.
    """

    parser = \
        argparse.ArgumentParser(description='Deep-learning based classifiers')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate the classifier on the given datasource')
    parser.add_argument('--top', type=int, default=None,
                        help='evaluate top-n accuracy of classifier')
    parser.add_argument('--scores', '--no-scores', dest='scores',
                        action=ToolboxArgparse.NegateAction,
                        nargs=0, default=None,
                        help='output classification scores '
                        '(in case of soft classifier)')
    parser.add_argument('--classifier-info', action='store_true',
                        default=False,
                        help='output additional information on the classifier')
    ToolboxArgparse.add_arguments(parser, ('network',))
    parser.add_argument('image', metavar='IMAGE', nargs='*',
                        help='images to classify')
    args = parser.parse_args()
    ToolboxArgparse.process_arguments(parser, args)

    classifier = NetworkArgparse.network(parser, args)

    if classifier is None:
        print("No classifier was specified.")
        return

    if args.classifier_info:
        print(f"{type(classifier).__name__} is an ImageClassifier:",
              isinstance(classifier, ImageClassifier))
        print(f"{type(classifier).__name__} is a SoftClassifier:",
              isinstance(classifier, SoftClassifier))
        print(f"{type(classifier).__name__} is a Network:",
              isinstance(classifier, Network))

    if args.evaluate:
        #
        # Evaluate classifier on a (labeled) dataset
        #

        evaluator = Evaluator(classifier)
        terminal = Terminal()
        imagenet = ImageNet()
        imagenet.prepare()
        evaluator.evaluate(imagenet, top=args.top, terminal=terminal)

    else:
        #
        # Classify data given as command line arguments
        #

        if args.scores is None:
            args.scores = isinstance(classifier, SoftClassifier)
        elif args.scores and not isinstance(classifier, SoftClassifier):
            args.scores = False
            LOG.warning("Not reporting scores as %s is not a soft classifier",
                        classifier)

        if args.top is not None and not isinstance(classifier, SoftClassifier):
            args.top = None
            LOG.warning("Not listing top classes as %s is not a "
                        "soft classifier", classifier)

        for filename in args.image:

            if args.top is None:
                if args.scores:
                    label, score = \
                        classifier.classify(filename, confidence=True)
                    print(f"classify('{filename}', confidence=True): "
                          f"{label['text'], score}")
                else:
                    label = classifier.classify(filename)
                    print(f"classify('{filename}'): {label['text']}")
            else:
                if args.scores:
                    labels, scores = \
                        classifier.classify(filename, top=args.top,
                                            confidence=True)
                    print(f"classify('{filename}', top={args.top}, "
                          f"scores={args.scores}): ")
                    for i, (label, score) in enumerate(zip(labels, scores)):
                        print(f"({i+1}) {label['text']} ({score*100:.2f}%)")
                else:
                    labels = classifier.classify(filename, top=args.top)
                    print(f"classify('{filename}', top=args.top): "
                          f"{[label['text'] for label in labels]}")

    # else:
    # scores = classifier.class_scores(filename)
    #    print(f"class_scores('{filename}': {scores.shape}")


if __name__ == "__main__":
    main()
