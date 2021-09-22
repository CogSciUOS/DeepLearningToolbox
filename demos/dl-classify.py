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

    def accuracy(self, data: Data, top: int = None,
                 terminal: Terminal = None) -> float:
        """Compute accuracy values for the given data and add them
        to the counters.

        Arguments
        ---------
        data: Data
            The Data for which to compute the accuracy.
        top: int
            Compute the top-n accuracy in addition top-1 accuracy
        terminal: Terminal
            A terminal to report results.

        """
        # FIXME[todo]: batch processing
        scores = self._classifier.class_scores(data)
        label, confidence = self._classifier.top_classes(scores, top=top)
        rank, _score = self._classifier.class_rank(scores, data.label)

        label0 = label if top is None else label[0]
        self._correct += int(label0 == data.label)
        self._error += int(label0 != data.label)
        self._total += 1
        if top is not None:
            self._correct_top += rank <= top

        if terminal is not None:
            confidence0 = confidence if top is None else confidence[0]
            if label0 == data.label:
                # correct: data label (ground truuth) is equal to the
                # predicted label
                text = (terminal.status('correct', status='ok') +
                        f": '{label0['text']}' "
                        f"(confidence={confidence0:.3f})")
            elif top is not None and rank < top:
                # top-n correct: ground truth is within the top-n predictions
                # (only applies if a top values was provided)
                text = (terminal.status(f'top-{top}', status='ok2') +
                        f": '{label[rank]['text']}' vs. "
                        f"'{data.label['text']}', "
                        f"rank={rank+1}"
                        f"(confidence={confidence[rank]:.3f})")
            else:
                text = (terminal.status('error', status='fail') +
                        f": '{label0['text']}' vs. "
                        f"'{data.label['text']}', "
                        f"rank={rank+1}")
            terminal.output(f"{data.filename}: {text}")

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

    def evaluate(self, datasource: Datasource, top: int = None,
                 terminal: Terminal = None) -> None:
        """Run an evaluation loop.
        """
        while True:
            try:
                data = datasource.get_random()
                self.accuracy(data, top, terminal=terminal)
                self.output_status(top)
            except RuntimeError as error:
                print(f"error procesing {data.filename} {data.shape}: {error}")
                raise
            except KeyboardInterrupt:
                print(f"error procesing {data.filename} {data.shape}")
                print("Keyboard interrupt")
                self.output_status(top, end='\n')
                break


# FIXME[hack]: move this class to some other module
class NegateAction(argparse.Action):
    """An `argparse.Action` allowing to negate an option by prefixing
    it with `'no'`.
    """
    def __call__(self, parser, ns, values, option):
        setattr(ns, self.dest, option[2:4] != 'no')


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
                        action=NegateAction, nargs=0, default=None,
                        help='output classification scores '
                        '(in case of soft classifier)')
    parser.add_argument('--network-info', action='store_true', default=False,
                        help='output additional information on the network')
    ToolboxArgparse.add_arguments(parser)
    NetworkArgparse.prepare(parser)
    parser.add_argument('image', metavar='IMAGE', nargs='*',
                        help='images to classify')
    args = parser.parse_args()
    ToolboxArgparse.process_arguments(parser, args)

    network = NetworkArgparse.network(parser, args)
    if network is None:
        print("No network was specified.")
        return

    if args.network_info:
        print(f"{type(network).__name__} is an ImageClassifier:",
              isinstance(network, ImageClassifier))
        print(f"{type(network).__name__} is a SoftClassifier:",
              isinstance(network, SoftClassifier))
        print(f"{type(network).__name__} is a Network:",
              isinstance(network, Network))

    if args.evaluate:
        #
        # Evaluate classifier on a (labeled) dataset
        #

        evaluator = Evaluator(network)
        terminal = Terminal()
        imagenet = ImageNet()
        imagenet.prepare()
        evaluator.evaluate(imagenet, top=args.top, terminal=terminal)

    else:
        #
        # Classify data given as command line arguments
        #

        if args.scores is None:
            args.scores = isinstance(network, SoftClassifier)
        elif args.scores and not isinstance(network, SoftClassifier):
            args.scores = False
            LOG.warning("Not reporting scores as %s is not a soft classifier",
                        network)

        if args.top is not None and not isinstance(network, SoftClassifier):
            args.top = None
            LOG.warning("Not listing top classes as %s is not a "
                        "soft classifier", network)

        for filename in args.image:

            if args.top is None:
                if args.scores:
                    label, score = network.classify(filename, confidence=True)
                    print(f"classify('{filename}', confidence=True): "
                          f"{label['text'], score}")
                else:
                    label = network.classify(filename)
                    print(f"classify('{filename}'): {label['text']}")
            else:
                if args.scores:
                    labels, scores = network.classify(filename, top=args.top,
                                                      confidence=True)
                    print(f"classify('{filename}', top={args.top}, "
                          f"scores={args.scores}): ")
                    for i, (label, score) in enumerate(zip(labels, scores)):
                        print(f"({i+1}) {label['text']} ({score*100:.2f}%)")
                else:
                    labels = network.classify(filename, top=args.top)
                    print(f"classify('{filename}', top=args.top): "
                          f"{[label['text'] for label in labels]}")

    # else:
    # scores = network.class_scores(filename)
    #    print(f"class_scores('{filename}': {scores.shape}")


if __name__ == "__main__":
    main()
