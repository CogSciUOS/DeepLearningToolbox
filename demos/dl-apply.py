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
from typing import Optional, Iterable
from argparse import ArgumentParser
import os
import sys
import logging

# thirdparty imports
import tqdm

# toolbox imports
import dltb.argparse as ToolboxArgparse
from dltb.base.data import Data
from dltb.base.image import Image, Imagelike, ImageDisplay
from dltb.base.image import BoundingBox
from dltb.base.video import Webcam
from dltb.datasource import ImageDirectory
from dltb.datasource import argparse as DatasourceArgparse
from dltb.tool.face import Detector as FaceDetector
from dltb.util.image import imshow, imwrite, get_display
from dltb.thirdparty import implementations, import_class

# logging
LOG = logging.getLogger(__name__)


# transformation = lambda x : x
def do_nothing(data: Data):
    return data


from dltb.tool.face.mtcnn import Detector
MTCNN = Detector()

# /space/data/lfw/lfw/Marissa_Jaret_Winokur/Marissa_Jaret_Winokur_0001.jpg
# /space/data/lfw/lfw/John_Allen_Muhammad/John_Allen_Muhammad_0010.jpg
# /space/data/lfw/lfw/John_Allen_Muhammad/John_Allen_Muhammad_0011.jpg
# /space/data/lfw/lfw/Chita_Rivera/Chita_Rivera_0001.jpg

# wrong face:
# /space/data/lfw/lfw/Gordon_Brown/Gordon_Brown_0006.jpg


def detect_faces(image: Imagelike):
    detections = MTCNN(image)
    return detections


def select_face(bounding_boxes, shape) -> int:
    """
    """
    if not bounding_boxes:
        raise ValueError("No face detected")

    if len(bounding_boxes) == 1:
        return 0  # there is only one detection -> assume that is correct

    center = (shape[1]/2, shape[0]/2)
    best, distance = -1, float('inf')
    for index, bounding_box in enumerate(bounding_boxes):
        loc_center = bounding_box.center
        dist2 = ((loc_center[0]-center[0]) ** 2 +
                 (loc_center[1]-center[1]) ** 2)
        if dist2 < distance:
            best, distance = index, dist2

    return best


def landmark_face(image: Imagelike):
    """
    """
    # image = Image(image)
    detections = detect_faces(image)

    if not detections.has_regions:
        print(f"No faces in {image.filename}.")
        LOG.warning("No faces detected in '%s'.", image.filename)
        return

    for idx, region in enumerate(detections.regions):
        print(idx, region)
    bounding_boxes = [region.location for region in detections.regions
                      if isinstance(region.location, BoundingBox)]

    faces = len(bounding_boxes)
    if faces != 1:
        LOG.warning("%d faces detected in '%s'.", faces, image.filename)

    best = select_face(bounding_boxes, image.shape)

    # for idx, region in enumerate(detections.regions):
    #     color = (0, 255, 0) if idx // 2 == best else (255, 0, 0)
    #     region.location.mark_image(image.array, color=color)
    for idx, region in enumerate(detections.regions):
        if idx // 2 == best:
            continue
        region.location.mark_image(image.array, color=(255, 0, 0))

    box = detections.regions[2*best].location
    landmarks = detections.regions[2*best+1].location

    box.mark_image(image.array, color=(0, 255, 0))
    landmarks.mark_image(image.array, color=(0, 255, 0))

    return box, landmarks, faces == 1


def apply(data: Data, input_directory=None, output_directory=None,
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

    # save = imwrite
    save = print

    # assume data is Image
    image = data

    result = landmark_face(image)

    print(data.filename, result[0], result[1], result[2])
    #output_tuple = (image.filename,
    #                result[0].x1, result[0].y1, result[0].x2, result[0].y2
    #                )  # FIXME[todo]:  result[1]. ...
    # result[0].mark_image(image.array)
    # result[1].mark_image(image.array)
    if display is not None:
        display.show(image, blocking=not result[2], wait_for_key=not result[2])
        if display.closed:
            raise StopIteration("Display closed")

    if input_directory and output_directory:
        # Write result into a file in the output directory.
        # Relative filenames in input and output directory will be the same.
        if not data.has_attribute('filename'):
            LOG.warning("No filename for data object. No output "
                        "file will be written.")
        elif not data.filename.startswith(input_directory):
            LOG.warning("File will not be written: filename '%s' is outside "
                        "the input directory '%s'.",
                        data.filename, input_directory)
        else:
            output_filename = \
                output_directory / data.filename[len(input_directory):]
            save(output_filename, result)


def apply_multi(datasource: Iterable[Data],
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

    try:
        for data in datasource:
            output(f"{data.filename} [{data.shape}]")
            apply(data, display=display)
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
    DatasourceArgparse.prepare(parser)

    args = parser.parse_args()
    ToolboxArgparse.process_arguments(parser, args)

    datasource = DatasourceArgparse.datasource(parser, args)
    if not datasource:
        display = get_display()
        for image in args.images:
            apply(Image(image), display=display)
        LOG.error("No datasource. "
                  "Specify a datasource (e.g. --datasource lfw).")
        return os.EX_USAGE

    display = get_display()
    apply_multi(datasource, progress=tqdm.tqdm, display=display)
    return os.EX_OK


if __name__ == '__main__':
    sys.exit(main())
