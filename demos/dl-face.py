#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""A command line interface for running and testing face tools.

.. moduleauthor:: Ulf Krumnack


Demos
-----

Face verification:

`dl-face.py `


Resources
---------

FIXME[todo]: 
[1] keras-vggface
    https://github.com/rcmalli/keras-vggface
"""

# standard imports
from argparse import ArgumentParser
import os
from pathlib import Path

# thirdparty imports
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tqdm

# toolbox imports
import dltb.argparse as ToolboxArgparse
from dltb.base.data import Data
from dltb.base.image import Image, Imagelike, ImageDisplay, ImageWarper
from dltb.base.video import Webcam, VideoReader
from dltb.datasource import ImageDirectory
from dltb.datasource import argparse as DatasourceArgparse
from dltb.tool.face import Detector as FaceDetector
from dltb.tool.face.landmarks import apply_single_hack, apply_multi_hack
from dltb.tool.align import LandmarkAligner
from dltb.util.image import imshow, get_display


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


def plot_image(image, feature_extractor, name) -> None:
    """Apply preprocessing to an image and plot results.
    """
    # colors: eye_left, eye_right, nose, mouth_left, mouth_right
    colors = ('green', 'blue', 'orange', 'red', 'red')

    plt.clf()
    plt.suptitle(name)

    #
    # 1. image: input image
    #
    plt.subplot(1, 3, 1)
    plt.title("Input image")
    plt.imshow(image)

    #
    # 2. image: detected bounding box and landmarks
    #
    plt.subplot(1, 3, 2)
    plt.title("MTCNN results")
    plt.imshow(image)
    # detections = feature_extractor.detect(image)
    # for bbox, landmarks, confidence in detections:
    bbox, landmarks, confidence, _unique = feature_extractor.detect(image)
    left, top, width, height = bbox
    rect = patches.Rectangle((left, top), width, height,
                             linewidth=1, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    for mark in range(5):
        point = landmarks[mark]
        plt.plot(point[0], point[1], color=colors[mark], marker='*')

    #
    # 3. image: aligned image with landmarks and reference points
    #
    plt.subplot(1, 3, 3)
    plt.title("Aligned image")
    aligned_image, aligned_landmarks = \
        feature_extractor.align_face(image, landmarks)

    reference = feature_extractor._reference()
    for mark in range(5):
        plt.plot(*reference[mark], color='white', marker='+', markersize=30)
        plt.plot(*aligned_landmarks[mark], color=colors[mark], marker='*')

    plt.imshow(aligned_image)
    plt.show()


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
    parser.add_argument('--evaluate', action='store_true', default=True,
                        help='perform evaluation')
    parser.add_argument('--output-directory', type=str, default='output',
                        help='path of the output directory')

    group_detector = parser.add_argument_group("Detector arguments")
    group_detector.add_argument('--detect', action='store_true', default=False,
                                help='run face detection')
    group_detector.add_argument('--detector', type=str,
                                help='the face detector to use')
    group_detector.add_argument('--list-detectors', action='store_true',
                                default=False, help='list available detectors')

    group_aligner = parser.add_argument_group("Alignment arguments")
    group_aligner.add_argument('--align', action='store_true', default=False,
                               help='run face alignment')
    group_aligner.add_argument('--warper', type=str, default=None,
                               help='the image warper to use')
    group_aligner.add_argument('--list-warpers', action='store_true',
                               default=False,
                               help='list available image warpers')
    group_aligner.add_argument('--size', type=str, default='112x112',
                               help='size of the output image')

    group_recognizer = parser.add_argument_group("Recognition arguments")
    group_recognizer.add_argument('--verify', action='store_true',
                                  default=False,
                                  help='run face verification')

    ToolboxArgparse.add_arguments(parser)
    DatasourceArgparse.prepare(parser)

    args = parser.parse_args()
    ToolboxArgparse.process_arguments(parser, args)

    if args.list_detectors:
        print(FaceDetector.implementation_info())
        return os.EX_OK

    if args.list_warpers:
        print(ImageWarper.implementation_info())
        return os.EX_OK

    # obtain the datasource if provided (otherwise None)
    datasource = DatasourceArgparse.datasource(parser, args)

    if args.detector:
        detector = FaceDetector(implementation=args.detector)

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
                output_detections(detector, data)  # , extract=True

    elif args.align:
        #
        # perform face alignment
        #

        # obtain the face detector
        detector_implementation = 'dltb.thirdparty.face_evolve.mtcnn.Detector'
        detector = FaceDetector(implementation=detector_implementation)
        print(f"Detector: {detector} ({type(detector)})")

        # obtain the ImageWarper
        warper = ImageWarper(implementation=args.warper)

        # create an aligner
        aligner = LandmarkAligner(detector=detector, size=args.size,
                                  warper=warper)

        # obtain an ImageDisplay object if --show is set (otherwise None)
        display = get_display() if args.show else None

        if not datasource:
            for image in args.images:
                apply_single_hack(Image(image), detector, aligner,
                                  display=display)
        else:
            apply_multi_hack(datasource, detector, aligner,
                             input_directory=datasource.directory,
                             output_directory=Path(args.output_directory),
                             progress=tqdm.tqdm, display=display)

    elif args.evaluate:

        # obtain the face detector
        detector_implementation = 'dltb.thirdparty.face_evolve.mtcnn.Detector'
        detector = FaceDetector(implementation=detector_implementation)
        print(f"Detector: {detector} ({type(detector)})")

        # obtain the ImageWarper
        warper = ImageWarper(implementation=args.warper)

        # create an aligner
        aligner = LandmarkAligner(detector=detector, size=args.size,
                                  warper=warper)

        from dltb.thirdparty.arcface import ArcFace

        arcface = ArcFace(aligner=aligner)

        embedding_file_name = Path("embeddings.npz")
        if embedding_file_name.is_file():
            content = np.load(embedding_file_name)
            embeddings, labels = content['embeddings'], content['labels']
        else:
            iterable = datasource.pairs()
            iterable = tqdm.tqdm(iterable)

            embeddings, labels = arcface.embed_labeled_pairs(iterable)

            print(f"Writing embeddings of shape {embeddings.shape} to "
                  f"'{embedding_file_name}'")
            np.savez_compressed(embedding_file_name,
                                embeddings=embeddings, labels=labels)

        print("embeddings:", embeddings.shape, embeddings.dtype)
        print("labels:", labels.shape, labels.dtype)
        #for image1, image2, same in iterable:
        #    print(image1.shape, image2.shape, same)
        #    embedding1 = embed(image1)
        #    embedding2 = embed(image1)
        #    distance = distance(embedding1, embedding2)

    else:
        print("No operation specified.")


if __name__ == "__main__":
    main()
