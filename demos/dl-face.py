#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""A command line interface for running and testing face tools.

.. moduleauthor:: Ulf Krumnack

"""

# standard imports
from argparse import ArgumentParser

# toolbox imports
from dltb.tool import Tool
from dltb.util.image import imshow


def main():
    """Main program: parse command line options and start face tools.
    """

    parser = ArgumentParser(description='Deep-learning based face processing')
    parser.add_argument('images', metavar='IMAGE', type=str, nargs='+',
                        help='an image to use')
    parser.add_argument('--detect', action='store_true', default=True,
                        help='run face detection')
    args = parser.parse_args()

    if args.detect:
        detector = Tool.register_initialize_key('haar')
        print(f"Detector: {detector} [{detector.prepared}]: ",
              type(detector).__mro__)
        detector.prepare()
        print(f"Detector: {detector} [{detector.prepared}]")

        for url in args.images:
            print(f"Applying detector to {url}")
            data = detector.process_image(url)
            print(data)
            image = data.data.copy()
            detections = detector.detections(data)
            print(detections)

            if detections:
                print(f"{detections.description}: {len(detections)}")
                for index, region in enumerate(detections.regions):
                    print(f"({index+1}) {region.location}")
                    region.mark_image(image)
                    imshow(region.location.extract(image), wait_for_key=True)
            else:
                print(f"{detections.description}: no detections")
            imshow(image, wait_for_key=True)
    else:
        print("No operation specified.")

if __name__ == "__main__":
    main()
