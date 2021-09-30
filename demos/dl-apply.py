#!/usr/bin/env python3
"""Apply an operator to data.  This may run:
* general data operator or specific operator (like image, sound, etc.)
* stand alone operators or preprocessing of tools
* apply to a single datum or a batch of data
* treat result in different ways (summarize, display, store, etc.)
* different operations
"""
# FIXME[todo]: this is highly experimental and not to be considered stable.
# The main goal of the demo script is to push forward the design of
# the Tool API.
# Currently the is a focus on face processing

# standard imports
from typing import Optional, Iterable, Tuple
from argparse import ArgumentParser
from pathlib import Path
import os
import sys
import logging

# thirdparty imports
import tqdm
import numpy as np

# toolbox imports
import dltb.argparse as ToolboxArgparse
from dltb.base.data import Data
from dltb.base.image import Image, ImageDisplay, ImageWarper
from dltb.base.image import BoundingBox
from dltb.datasource import argparse as DatasourceArgparse
from dltb.tool.face.mtcnn import Detector
from dltb.tool.face import Detector as FaceDetector
from dltb.tool.align import LandmarkAligner, Landmarks
from dltb.util.image import get_display
from dltb.util.keyboard import SelectKeyboardObserver
from dltb.util.canvas import canvas_create, canvas_add_image
from dltb.thirdparty import implementations

# logging
LOG = logging.getLogger(__name__)


# /space/data/lfw/lfw/Marissa_Jaret_Winokur/Marissa_Jaret_Winokur_0001.jpg
# /space/data/lfw/lfw/John_Allen_Muhammad/John_Allen_Muhammad_0010.jpg
# /space/data/lfw/lfw/John_Allen_Muhammad/John_Allen_Muhammad_0011.jpg
# /space/data/lfw/lfw/Chita_Rivera/Chita_Rivera_0001.jpg

# wrong face:
# /space/data/lfw/lfw/Gordon_Brown/Gordon_Brown_0006.jpg


def landmark_face(image: Image,
                  detector) -> Tuple[BoundingBox, Landmarks, bool]:
    """Obtain (best) bounding box and landmarks for a given image using
    a combined bounding box and landmark detector.

    Arguments
    ---------

    Result
    ------
    bounding_box:
        The bounding box for the (best) detection.
    landmarks:
        The facial landmarks for the (best) detection.
    unique:
        A flag indication if the detector returned a singe detection (`True`)
        or if the best of multiple detection was choosen using some
        heuristics (`False`)

    Raises
    ------
    ValueError
        No (best) detection could be identified.  Either there was no
        detection or the heuristics could not decided of a best detection.
    """
    # image = Image(image)
    # detections = detector(image)
    detections = detector.detections(image)

    if not detections.has_regions:
        raise ValueError("No regions returned by detector for "
                         f"image '{image.filename}'.")

    regions = len(detections.regions)
    if regions == 0:
        raise ValueError("There were 0 regions returned by detector for "
                         f"image '{image.filename}'.")

    if regions % 2 != 0:
        raise ValueError("The box-and-landmark-detector returned "
                         f"an odd number ({regions}) of regions "
                         f"for image '{image.filename}'.")

    faces = regions // 2
    if faces == 1:
        best = 0
    else:
        best = detector.select_best(detections, image)
        LOG.info("%d faces detected in '%s' (%d is best).",
                 faces, image.filename, best)

    bbox = detections.regions[best*2].location
    landmarks = detections.regions[best*2+1].location

    detector.mark_data(image, detections, best=best, group_size=2)

    return bbox, landmarks, faces == 1


def align_face(aligner, image, landmarks) -> np.ndarray:
    """Align a face image given an aligner.
    """
    # ./dltb/tool/face/landmarks.py
    #  - definition of a LandmarkDetector (called "DetectorBase")
    #    all subclasses should provide
    #     _reference_landmarkds: Landmarks
    #     _reference_size: Size
    #    the method reference can scale reference landmarks to a
    #    desired size
    #
    # ./dltb/tool/align.py
    #  - definition of abstract class LandmarkAligner
    #
    # ./dltb/thirdparty/opencv/__init__.py
    #  - definition of class OpenCVLandmarkAligner(LandmarkAligner):
    #    implementing the methods compute_transformation and
    #    apply_transformation
    #
    # ./dltb/thirdparty/face_evolve/mtcnn.py
    #   - Implementation of a LandmarkDetector providing
    #     reference landmarks
    #
    # ./dltb/thirdparty/datasource/lfw.py
    #
    #

    # The reference points used by the aligner
    reference = aligner.reference

    # The transformation computed by the aligner
    transformation = aligner.compute_transformation(landmarks)

    # The aligned image computed by the aligner
    aligned_image = aligner.apply_transformation(image, transformation)

    return aligned_image


def apply_single(data: Data, detector, aligner,
                 display: Optional[ImageDisplay] = None) -> None:
    """Apply a transformation to a data objects.

    State: immature - hard-wired transformation

    Arguments
    ---------
    data:
        The :py:class:`Data` object to which the operation is to be
        applied.
    input_directory:
        The base directory of the data.  Relative paths are determined
        relative to the directory.
    output_directory:
        The output directory.  Relative output filenames are interpreted
        relative to this directory.
    """

    # assume data is Imagelike: process_image transforms Imagelike -> Image
    image = detector.process_image(data)

    # bounding box and landmark for the (best) detection
    bbox, landmarks, _unique = landmark_face(image, detector)

    # a marked version or the image
    marked_image = detector.marked_image(image)

    # an aligned (and cropped) version of the marked image
    aligned_image_marked = align_face(aligner, marked_image, landmarks)

    # an aligned version of the original image
    aligned_image = aligner(image, landmarks)

    # print(data.filename, result[0], result[1], result[2])
    # output_tuple = (image.filename,
    #                result[0].x1, result[0].y1, result[0].x2, result[0].y2
    #                )  # FIXME[todo]:  result[1]. ...
    # result[0].mark_image(image.array)
    # result[1].mark_image(image.array)
    if display is not None:
        shape = image.shape
        canvas = canvas_create(size=(shape[0]*2, shape[1]*2),
                               channels=shape[2])

        # 1. the original image
        canvas_add_image(canvas, marked_image, rows=2, columns=2, index=1)

        # 2. the image cropped to the bounding box
        cropped_image = \
            marked_image[int(bbox.y1):int(bbox.y2), int(bbox.x1):int(bbox.x2)]
        canvas_add_image(canvas, cropped_image, rows=2, columns=2, index=2)

        # 3. the aligned image (marked version)
        canvas_add_image(canvas, aligned_image_marked,
                         rows=2, columns=2, index=3)

        # 4. the aligned image
        canvas_add_image(canvas, aligned_image, rows=2, columns=2, index=4)

        #display.show(canvas, blocking=not unique, wait_for_key=not unique)
        display.show(canvas, blocking=False)

    return aligned_image


def apply_multi(datasource: Iterable[Data], detector, aligner,
                input_directory=None, output_directory=None,
                display: Optional[ImageDisplay] = None,
                progress: Optional[Iterable] = None) -> None:
    """Apply a transformation to multiple data objects.

    State: immature - hard-wired transformation
    """
    if progress is not None:
        datasource = progress(datasource)
        output = datasource.set_description
    else:
        output = print

    stop_on_key_press = SelectKeyboardObserver()

    input_directory = input_directory and str(input_directory)
    
    try:
        for data in stop_on_key_press(datasource):
            output(f"{data.filename} [{data.shape}]")
            aligned_image = apply_single(data, detector, aligner,
                                         display=display)
            if display is not None and display.closed:
                LOG.error("Display was closed: stopping operation.")
                break

            if input_directory and output_directory:
                # Write result into a file in the output directory.
                # Relative filenames in input and output directory
                # will be the same.
                if not data.has_attribute('filename'):
                    LOG.warning("No filename for data object. No output "
                                "file will be written.")
                elif not data.filename.startswith(input_directory):
                    LOG.warning("File will not be written: filename '%s' "
                                "is outside the input directory '%s'.",
                                data.filename, input_directory)
                else:
                    output_filename = \
                        output_directory / data.filename[len(input_directory):]
                    print(f"{data.filename} -> {output_filename} "
                          f"[{aligned_image.shape}]")
                    # save(output_filename, aligned_image)

    except KeyboardInterrupt:
        LOG.error("Keyboard interrupt: stopping operation.")


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
            apply_single(Image(image), detector, aligner, display=display)
    else:
        apply_multi(datasource, detector, aligner,
                    input_directory=datasource.directory,
                    output_directory=Path('output'),
                    progress=tqdm.tqdm, display=display)

    return os.EX_OK


if __name__ == '__main__':
    sys.exit(main())
