#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""A command line interface for running and testing video tools.

.. moduleauthor:: Ulf Krumnack

"""

# standard imports
import os
from argparse import ArgumentParser

# toolbox imports
import dltb.argparse as ToolboxArgparse
from dltb.datasource import Datasource, DataDirectory, Imagesource
from dltb.base.data import Data
from dltb.base.image import ImageDisplay
from dltb.base.video import VideoReader, VideoWriter
from dltb.tool import Tool
from dltb.tool.detector import ImageDetector


def main():
    """Main program: parse command line options and start video tools.
    """

    parser = ArgumentParser(description='Deep learning based video processing')
    parser.add_argument('video', metavar='VIDEO', type=str, nargs='*',
                        help='an image to use')
    parser.add_argument('--detect', action='store_true', default=True,
                        help='run  detection')
    parser.add_argument('--output', type=str,
                        help='write output in the given video file')
    ToolboxArgparse.add_arguments(parser)

    args = parser.parse_args()
    ToolboxArgparse.process_arguments(parser, args)

    if args.webcam:
        Datasource['Webcam']
        webcam = Datasource['Webcam']
        webcam.prepare()
        display = ImageDisplay(module='qt')

        try:
            with VideoWriter(filename='test.mp4',
                             fps=3, size=None) as writer:
                # FIXME[bug]: fps seems to be ignored by ImageIO writer
                for i in range(100):
                    data = webcam.get_data()
                    print(i, data.array.shape)
                    detector.process(data, mark=True)
                    output_detections(detector, data, display=display,
                                      writer=writer)
                    if display.closed:
                        break
        except KeyboardInterrupt:
            print("stop")

    elif args.video:
        # FIXME[todo]: displays other than 'qt' do not work!
        display = ImageDisplay(module='qt')  # module='matplotlib'/'opencv'
        print(type(display))
        filename = ("/net/store/cv/users/krumnack/"
                    "Videos/Kids Go To School _ "
                    "Brother's Birthday Chuns And Friends Make "
                    "a Birthday Cake Big-R_uvHSE5Giw.mkv")
        reader = VideoReader(filename)
        print(type(reader))
        for frame in reader:
            marked_frame = detector.mark_image(frame)
            display.show(marked_frame)
            if display.closed:
                break

    else:
        for url in args.images:
            if os.path.isdir(url):
                class MyDatasource(DataDirectory, Imagesource):
                    """Dummy datasource - make this an official class
                    """
                datasource = MyDatasource('images')
                datasource.prepare()
                for data in datasource:
                    detector.process(data, mark=True)
                    output_detections(detector, data)
            else:
                print(f"Applying detector to {url}")
                data = detector.process_image(url, mark=True, extract=True)
                output_detections(detector, data)  # , extract=True

    else:
        print("No operation specified.")


if __name__ == "__main__":
    main()
