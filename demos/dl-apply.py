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
from typing import Optional, Iterable, Sequence
from argparse import ArgumentParser
import os
import sys
import logging

# thirdparty imports
import tqdm
import numpy as np

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
from dltb.util.keyboard import SelectKeyboardObserver
from dltb.thirdparty import implementations, import_class

# logging
LOG = logging.getLogger(__name__)


# transformation = lambda x : x
def do_nothing(data: Data):
    return data


from dltb.tool.face.mtcnn import Detector
#MTCNN = Detector()
MTCNN = Detector(implementation='dltb.thirdparty.face_evolve.mtcnn.Detector')

# /space/data/lfw/lfw/Marissa_Jaret_Winokur/Marissa_Jaret_Winokur_0001.jpg
# /space/data/lfw/lfw/John_Allen_Muhammad/John_Allen_Muhammad_0010.jpg
# /space/data/lfw/lfw/John_Allen_Muhammad/John_Allen_Muhammad_0011.jpg
# /space/data/lfw/lfw/Chita_Rivera/Chita_Rivera_0001.jpg

# wrong face:
# /space/data/lfw/lfw/Gordon_Brown/Gordon_Brown_0006.jpg


def detect_faces(image: Imagelike):
    detections = MTCNN(image)
    return detections


def select_face(bounding_boxes: Sequence[BoundingBox],
                image: Imagelike) -> int:
    """Select the main face from a set of alternative detections.

    The criteria for selecting the main face may vary depending on the
    data.  For example, if data is known to have face as central
    prosition and/or with a specific size, these information can be
    used to select the best candidate.  Also the confidence score
    provided by the detector may be used.

    Arguments
    ---------
    bounding_boxes:
        The bouding boxes returned by the detector.
    image:
        The image to which the bounding boxes refer.
    """
    if not bounding_boxes:
        raise ValueError("No face detected")

    if len(bounding_boxes) == 1:
        return 0  # there is only one detection -> assume that is correct

    # select bounding box with center closest to the image center
    #shape = Image.as_shape(image)
    shape = image.shape
    center = (shape[1]/2, shape[0]/2)
    best, distance = -1, float('inf')
    for index, bounding_box in enumerate(bounding_boxes):
        loc_center = bounding_box.center
        dist2 = ((loc_center[0]-center[0]) ** 2 +
                 (loc_center[1]-center[1]) ** 2)
        if dist2 < distance:
            best, distance = index, dist2

    return best


def align_face(image, landmarks) -> np.ndarray:
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
    reference = MTCNN.reference(size=(112, 112))
    print(f"reference:\n {reference}")
    for point in reference.points:
        print(f"  {point}")


def landmark_face(image: Imagelike):
    """
    """
    # image = Image(image)
    detections = detect_faces(image)

    if not detections.has_regions:
        LOG.warning("No regions returned by detector for image '%s'.",
                    image.filename)
        return None, None, False

    if not len(detections.regions):
        LOG.warning("No faces detected in image '%s'.", image.filename)
        return None, None, False

    if len(detections.regions) % 2 != 0:
        LOG.warning("Odd number (%d) of regions for image '%s'.",
                    len(detections.regions), image.filename)
        return None, None, False

    # for idx, region in enumerate(detections.regions):
    #     print(idx, region)
    bounding_boxes = [region.location for region in detections.regions
                      if isinstance(region.location, BoundingBox)]

    faces = len(bounding_boxes)
    if faces != 1:
        LOG.warning("%d faces detected in '%s'.", faces, image.filename)

    best = select_face(bounding_boxes, image)

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


def apply_single(data: Data, input_directory=None, output_directory=None,
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

    bbox, landmarks, unique = landmark_face(image)

    align_face(image, landmarks)

    # print(data.filename, result[0], result[1], result[2])
    #output_tuple = (image.filename,
    #                result[0].x1, result[0].y1, result[0].x2, result[0].y2
    #                )  # FIXME[todo]:  result[1]. ...
    # result[0].mark_image(image.array)
    # result[1].mark_image(image.array)
    if display is not None:
        shape = image.shape
        canvas = np.zeros((shape[0]*2, shape[1]*2, shape[2]))
        array = image.array
        canvas[:shape[0],:shape[1]] = array
        canvas[int(bbox.y1):int(bbox.y2),
               shape[0]+int(bbox.x1):shape[0]+int(bbox.x2)] = \
            array[int(bbox.y1):int(bbox.y2), int(bbox.x1):int(bbox.x2)]
        #display.show(canvas, blocking=not unique, wait_for_key=not unique)
        display.show(canvas, blocking=False)
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

    stop_on_key_press = SelectKeyboardObserver()
    
    try:
        for data in stop_on_key_press(datasource):
            output(f"{data.filename} [{data.shape}]")
            apply_single(data, display=display)
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

    ToolboxArgparse.add_arguments(parser)
    DatasourceArgparse.prepare(parser)

    args = parser.parse_args()
    ToolboxArgparse.process_arguments(parser, args)

    datasource = DatasourceArgparse.datasource(parser, args)

    display = get_display() if args.show else None

    print(f"Detector: {MTCNN} ({type(MTCNN)})")
    
    if not datasource:
        for image in args.images:
            apply_single(Image(image), display=display)
        LOG.error("No datasource. "
                  "Specify a datasource (e.g. --datasource lfw).")
        return os.EX_USAGE

    apply_multi(datasource, progress=tqdm.tqdm, display=display)
    return os.EX_OK


if __name__ == '__main__':
    sys.exit(main())
