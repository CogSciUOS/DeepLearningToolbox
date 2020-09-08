#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A command line interface for running and testing face tools.

.. moduleauthor:: Ulf Krumnack

"""

# standard imports
import os
import sys
import logging
from argparse import ArgumentParser

# third party imports

# toolbox imports
from dltb.tool import Tool

# logging
LOG = logging.getLogger(__name__)
del logging


def main():

    parser = ArgumentParser(description=
                            'Deep-learning based face processing')
    parser.add_argument('--detect', action='store_true', default=True,
                        help='Run face detection')
    args = parser.parse_args()

    if args.detect:
        detector = Tool.register_initialize_key('haar')
        print(f"Detector: {detector} [{detector.prepared}]: ",
              type(detector).__mro__)
        detector.prepare()
        print(f"Detector: {detector} [{detector.prepared}]")



        data = detector.process_image('images/129647068_1.jpg')
        print(data)
        image = data.data.copy()
        detections = detector.detections(data)
        print(detections)

        import matplotlib as mpl 
        # mpl.use('wxAgg')  # ModuleNotFoundError: No module named 'wx'
        import matplotlib.pyplot as plt

        subplots = plt.subplots(nrows=1, ncols=2)

        print(f"{detections.description}: {len(detections.regions)}")
        for index, region in enumerate(detections.regions):
            print(f"({index+1}) {region.location}")
            region.mark_image(image)
            plt.imshow(region.location.extract(image))
            plt.waitforbuttonpress()
            plt.cla()
        plt.imshow(image)
    else:
        print("No operation specified.")

if __name__ == "__main__":
    main()
