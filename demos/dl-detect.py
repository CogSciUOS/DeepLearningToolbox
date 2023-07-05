#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A command line interface for running and testing (image) detectors.

.. moduleauthor:: Ulf Krumnack


Examples
--------

Detect faces in an image (examples/cat.jpg) .

   python demos/dl-detect.py --densenet examples/cat.jpg

Evaluate detector (on the ? validation set):

   python demos/dl-detect.py --densenet --evaluate

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
from dltb.base.image import Image
from dltb.tool.detector import Detector, ImageDetector
from dltb.tool.segmenter import ImageSegmenter
from dltb.tool.face import Detector as FaceDetector
from dltb.network import argparse as NetworkArgparse, Network
from dltb.datasource import Datasource
from dltb.util.terminal import Terminal
from dltb.util.image import imshow

# logging
LOG = logging.getLogger(__name__)



def main():
    """The main program.
    """

    parser = \
        argparse.ArgumentParser(description='Deep-learning based image detectors')
    parser.add_argument('--detector',
                        help='the detector to use')
    parser.add_argument('--scores', '--no-scores', dest='scores',
                        action=ToolboxArgparse.NegateAction,
                        nargs=0, default=None,
                        help='output detection scores '
                        '(in case of soft detector)')
    parser.add_argument('--detector-info', action='store_true',
                        default=False,
                        help='output additional information on the detector')
    parser.add_argument('--list-detectors', action='store_true',
                        default=False,
                        help='list registered detectors')
    ToolboxArgparse.add_arguments(parser, ('network',))
    parser.add_argument('image', metavar='IMAGE', nargs='*',
                        help='images to classify')
    args = parser.parse_args()
    ToolboxArgparse.process_arguments(parser, args)


    if args.list_detectors:
        ImageDetector.list_implementations()
        return  # exit program

    # 
    # detector = NetworkArgparse.network(parser, args)
    if args.detector:
        detector = ImageDetector(implementation=args.detector)

    if detector is None:
        print("No detector was specified.")
        return

    for img in args.image:
        image = Image(img)
        # detections = detector.detect(image)

        if isinstance(detector, ImageSegmenter):
            detector.segment_and_show(image)
        else:
            detector.detect_and_show(image)
        # imshow(image)

    
if __name__ == "__main__":
    main()
