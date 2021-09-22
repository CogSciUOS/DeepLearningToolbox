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
import logging

# thirdparty imports
import tqdm

# toolbox imports
import dltb.argparse as ToolboxArgparse
from dltb.base.data import Data
from dltb.base.image import ImageDisplay
from dltb.base.image import BoundingBox
from dltb.base.video import Webcam
from dltb.datasource import ImageDirectory
from dltb.datasource import argparse as DatasourceArgparse
from dltb.tool.face import Detector as FaceDetector
from dltb.util.image import imshow, imwrite
from dltb.thirdparty import implementations, import_class

# logging
LOG = logging.getLogger(__name__)


# transformation = lambda x : x
def do_nothing(data: Data):
    return data


from dltb.tool.face.mtcnn import Detector
MTCNN = Detector()


def landmark_face(data: Data):
    detections = MTCNN(data)
    faces = (len(detections.regions)//2 if detections.has_regions else 0)
    if not faces:
        print(f"No faces in {data.filename}.")
        LOG.warning("No faces detected in '%s'.", data.filename)
        return None, None

    if faces == 1:
        best = 0
    else:
        LOG.warning("%d faces detected in '%s'.", faces, data.filename)
        center = (data.shape[0]/2, data.shape[1]/2)
        best = -1
        distance = float('inf')
        for index, region in enumerate(detections.regions):
            if isinstance(region.location, BoundingBox):
                loc_center = region.location.center
                dist2 = ((loc_center[0]-center[0]) ** 2 +
                         (loc_center[1]-center[1]) ** 2)
                if dist2 < distance:
                    best = index
                    distance = dist2
    box = detections.regions[best].location
    landmarks = detections.regions[best+1].location

    return box, landmarks


def apply(data: Data, input_directory=None, output_directory=None) -> None:
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
    # transformation = do_nothing
    transformation = landmark_face

    result = transformation(data)
    output_tuple = (data.filename,
                    result[0].x1, result[0].y1, result[0].x2, result[0].y2
                    )  # FIXME[todo]:  result[1]. ...

    print(data.filename, result[0], result[1], output_tuple)

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
                progress: Optional[Iterable]) -> None:
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
            apply(data)
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
    apply_multi(datasource, tqdm.tqdm)

if __name__ == '__main__':
    main()
