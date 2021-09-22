#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""A command line interface for running and testing face tools.

.. moduleauthor:: Ulf Krumnack

"""

# standard imports
from argparse import ArgumentParser
import os

# toolbox imports
import dltb.argparse as ToolboxArgparse
from dltb.base.data import Data
from dltb.base.image import ImageDisplay, Imagelike
from dltb.base.video import Webcam, Reader as VideoReader
from dltb.datasource import ImageDirectory
from dltb.tool.face import Detector as FaceDetector
from dltb.util.image import imshow
from dltb.thirdparty import implementations, import_class


def output_detections(detector: FaceDetector, data: Data,
                      extract: bool = False) -> None:
    """Output detections in textual and graphical form.
    """
    detections = detector.detections(data)
    marked_image = detector.marked_image(data)

    print(detections.description)
    if detections:
        print(f"{detections.description}: {len(detections)}")
        for index, region in enumerate(detections.regions):
            print(f"({index+1}) {region.location}")

    else:
        print(f"{detections.description}: no detections")

    imshow(marked_image, wait_for_key=True, timeout=5)

    if extract:
        extractions = detector.extractions(data)
        print(f"Showing {len(extractions)} extractions:")
        for index, extraction in enumerate(extractions):
            print(f"({index+1}) {extraction.shape}")
            imshow(extraction, wait_for_key=True, timeout=1)


def display_detections(display: ImageDisplay, image: Imagelike,
                       detector: FaceDetector) -> None:
    """Process an image with a detector and display the results.

    Arguments
    ---------
    display:
        The display used for showing the marked image.
    image:
        The image to which the detector shall be applied.
    detector:
        The detector to be used for detection.
    """
    result = ('mark', )
    data = detector.process_image(image, result=result)
    marked_image = detector.marked_image(data)
    display.show(marked_image, blocking=False)


def display_video(display: ImageDisplay, video: VideoReader,
                  detector: FaceDetector) -> None:
    """Process a video frame-by-frame a detector and display the results.

    Arguments
    ---------
    display:
        The display used for showing the marked images.
    video:
        The video from which to obtain images.
    detector:
        The detector to be used for detection.
    """
    for frame in video:
        if display.closed:
            break
        display_detections(display, frame, detector)


def main():
    """Main program: parse command line options and start face tools.
    """

    parser = ArgumentParser(description='Deep learning based face processing')
    parser.add_argument('images', metavar='IMAGE', type=str, nargs='*',
                        help='an image to use')
    parser.add_argument('--webcam', action='store_true', default=False,
                        help='run on webcam')
    parser.add_argument('--show', action='store_true', default=True,
                        help='show results in a window')

    group_detector = parser.add_argument_group("Detector arguments")
    group_detector.add_argument('--detect', action='store_true', default=False,
                                help='run face detection')
    group_detector.add_argument('--detector', type=str,
                                help='the face detector to use')
    group_detector.add_argument('--list-detectors', action='store_true',
                                default=False, help='list available detectors')

    ToolboxArgparse.add_arguments(parser)

    args = parser.parse_args()
    ToolboxArgparse.process_arguments(parser, args)

    if args.list_detectors:
        print("FaceDetector implementations:")
        for index, implementation in enumerate(implementations(FaceDetector)):
            print(f"{index+1}) {implementation}")
        return

    if args.detector:
        detector = FaceDetector(implementation=args.detector)
    elif args.detector:  # FIXME[old]
        print(f"Detector class: {args.detector}")
        Detector = import_class(args.detector)
        detector = Detector()
        # 'haar', 'ssd', 'hog',  'cnn', 'mtcnn'
        # detector = Tool['haar']
        # detector = Tool['ssd']
        print(f"Detector: {detector} [prepared={detector.prepared}]")
        detector.prepare()
        print(f"Detector: {detector} [prepared={detector.prepared}]")

    if args.detect:

        if args.webcam:
            webcam = Webcam()
            display = ImageDisplay(module='qt')
            display.present(display_video, (webcam, detector))

        for url in args.images:
            if os.path.isdir(url):
                datasource = ImageDirectory('images')
                datasource.prepare()
                for data in datasource:
                    print(detector(data))
                    # detector.process(data, mark=True)
                    # output_detections(detector, data)
            else:
                print(f"Applying detector to {url}")
                # print(detector(url))
                result = ('detections', 'mark')  # , 'extract')
                data = detector.process_image(url, result=result) #mark=True, extract=True
                data.debug()
                output_detections(detector, data) # , extract=True

    else:
        print("No operation specified.")


if __name__ == "__main__":
    main()
