#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""A command line interface for running and testing face tools.

.. moduleauthor:: Ulf Krumnack

"""

# standard imports
import os
import time
from threading import Event
from argparse import ArgumentParser

# thirdparty imports
import imageio

# toolbox imports
from datasource import Datasource, DataDirectory, Datafetcher, Imagesource
from dltb.base.data import Data
from dltb.tool import Tool
from dltb.tool.detector import ImageDetector
from dltb.util.image import imshow
from dltb.thirdparty.qt import ImageDisplay


def output_detections(detector: ImageDetector, data: Data,
                      display: ImageDisplay = None, writer=None,
                      wait_for_key: bool = False,
                      timeout: float = None) -> None:
    marked_image = detector.marked_image(data)

    if display is not None:
        display.show(marked_image, wait_for_key=wait_for_key, timeout=timeout)

    if writer is not None:
        pass  # writer.append_data(marked_image)


def main():
    """Main program: parse command line options and start face tools.
    """

    parser = ArgumentParser(description='Deep learning based face processing')
    parser.add_argument('images', metavar='IMAGE', type=str, nargs='+',
                        help='an image to use')
    parser.add_argument('--detect', action='store_true', default=True,
                        help='run face detection')
    parser.add_argument('--loop', action='store_true', default=True,
                        help='run in loop mode')
    parser.add_argument('--webcam', action='store_true', default=True,
                        help='feed from the webcam')
    args = parser.parse_args()

    if args.detect:
        detector = Tool.register_initialize_key('haar')
        print(f"Detector: {detector} [prepared={detector.prepared}]")
        detector.prepare()
        print(f"Detector: {detector} [prepared={detector.prepared}]")

        if args.webcam:
            Datasource.register_initialize_key('Webcam')
            webcam = Datasource['Webcam']
            webcam.prepare()
            display = ImageDisplay()

            experiment = 1
            if experiment == 1:
                try:
                    with imageio.get_writer('test.mp4', fps=20) as writer:
                        for i in range(100):
                            print(i)
                            data = webcam.get_data()
                            detector.process(data, mark=True)
                            output_detections(detector, data, display=display,
                                              writer=writer)
                            if display.closed:
                                break
                except KeyboardInterrupt:
                    print("stop")

            elif experiment == 2:
                def worker(display):
                    with imageio.get_writer('test.mp4', fps=20) as writer:
                        for i in range(100):
                            print(i)
                            data = webcam.get_data()
                            detector.process(data, mark=True)
                            output_detections(detector, data, display=display,
                                              writer=writer)
                            if not display.active:
                                break
                display.run(worker=worker, args=(display,))

        else:

            for url in args.images:
                if os.path.isdir(url):
                    class MyDatasource(DataDirectory, Imagesource): pass
                    datasource = MyDatasource('images')
                    datasource.prepare()
                    # datafetcher = Datafetcher(datasource)
                    for data in datasource:
                        detector.process(data, mark=True)
                        output_detections(detector, data)
                else:
                    print(f"Applying detector to {url}")
                    data = detector.process_image(url, mark=True, extract=True)
                    output_detections(detector, data, extract=True)

    else:
        print("No operation specified.")


if __name__ == "__main__":
    main()
