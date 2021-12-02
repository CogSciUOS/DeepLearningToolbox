#!/usr/bin/env python3
"""Apply an operator to data.  This may run:
* general data operator or specific operator (like image, sound, etc.)
* stand alone operators or preprocessing of tools
* apply to a single datum or a batch of data
* treat result in different ways (summarize, display, store, etc.)
* different operations


Examples:
---------

Alignment: align all images from the LFW dataset and store results
in the given directory.

`dl-apply.py --datasource lfw --show --output-directory /work/ulf/lfw112`


Evaluate a face embedding on a verification task:

"""
# FIXME[todo]: this is highly experimental and not to be considered stable.
# The main goal of the demo script is to push forward the design of
# the Tool API.
# Currently the is a focus on face processing

# standard imports
from argparse import ArgumentParser
from pathlib import Path
import os
import sys
import logging

# thirdparty imports
import tqdm

# toolbox imports
import dltb.argparse as ToolboxArgparse
from dltb.base.image import Image, ImageWarper
from dltb.datasource import argparse as DatasourceArgparse
from dltb.tool.face.mtcnn import Detector
from dltb.tool.face.landmarks import apply_single_hack, apply_multi_hack
from dltb.tool.face import Detector as FaceDetector
from dltb.tool.align import LandmarkAligner
from dltb.util.image import get_display
from dltb.thirdparty import implementations

# logging
LOG = logging.getLogger(__name__)


# /space/data/lfw/lfw/Marissa_Jaret_Winokur/Marissa_Jaret_Winokur_0001.jpg
# /space/data/lfw/lfw/John_Allen_Muhammad/John_Allen_Muhammad_0010.jpg
# /space/data/lfw/lfw/John_Allen_Muhammad/John_Allen_Muhammad_0011.jpg
# /space/data/lfw/lfw/Chita_Rivera/Chita_Rivera_0001.jpg

# wrong face:
# /space/data/lfw/lfw/Gordon_Brown/Gordon_Brown_0006.jpg




def main():
    """Main program: parse command line options and start face tools.
    """

    parser = ArgumentParser(description='Deep learning based face processing')
    parser.add_argument('images', metavar='IMAGE', type=str, nargs='*',
                        help='an image to use')
    parser.add_argument('--webcam', action='store_true', default=False,
                        help='run on webcam')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show results in a window')

    group_detector = parser.add_argument_group("Detector arguments")
    group_detector.add_argument('--detect', action='store_true', default=False,
                                help='run face detection')
    group_detector.add_argument('--detector', type=str,
                                help='the face detector to use')
    group_detector.add_argument('--list-detectors', action='store_true',
                                default=False, help='list available detectors')
    group_detector.add_argument('--warper', type=str, default=None,
                                help='the image warper to use')
    group_detector.add_argument('--list-warpers', action='store_true',
                                default=False,
                                help='list available image warpers')
    group_detector.add_argument('--size', type=str, default='112x112',
                                help='size of the output image')
    group_detector.add_argument('--output-directory', type=str,
                                default='output',
                                help='path of the output directory')

    ToolboxArgparse.add_arguments(parser)
    DatasourceArgparse.prepare(parser)

    args = parser.parse_args()
    ToolboxArgparse.process_arguments(parser, args)

    if args.list_detectors:
        print("FaceDetector implementations:")
        for index, implementation in enumerate(implementations(FaceDetector)):
            print(f"{index+1}) {implementation}")
        return os.EX_OK

    if args.list_warpers:
        print("ImageWarper implementations:")
        for index, implementation in enumerate(ImageWarper.implementations()):
            print(f"{index+1}) {implementation}")
        return os.EX_OK

    # obtain the datasource if provided (otherwise None)
    datasource = DatasourceArgparse.datasource(parser, args)

    # obtain an ImageDisplay object if --show is set (otherwise None)
    display = get_display() if args.show else None

    # obtain the face detector
    detector = \
        Detector(implementation='dltb.thirdparty.face_evolve.mtcnn.Detector')
    print(f"Detector: {detector} ({type(detector)})")

    # obtain the ImageWarper
    #warper = ImageWarper(implementation='dltb.thirdparty.skimage.ImageUtil')
    #warper = ImageWarper(implementation='dltb.thirdparty.opencv.ImageUtils')
    warper = ImageWarper(implementation=args.warper)

    # create the LandmarkAligner
    aligner = LandmarkAligner(detector=detector, size=args.size, warper=warper)

    if not datasource:
        for image in args.images:
            apply_single_hack(Image(image), detector, aligner, display=display)
    else:
        apply_multi_hack(datasource, detector, aligner,
                         input_directory=datasource.directory,
                         output_directory=Path(args.output_directory),
                         progress=tqdm.tqdm, display=display)

    return os.EX_OK


if __name__ == '__main__':
    sys.exit(main())
