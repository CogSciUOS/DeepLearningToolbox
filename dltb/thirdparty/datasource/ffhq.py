"""The Flickr-Faces-HQ Dataset (FFHQ).
"""

# standard imports
import json
import logging

# third party imports
import numpy as np

# toolbox imports
from dltb import config
from dltb.base.data import Data
from dltb.datasource import ImageDirectory

# logging
LOG = logging.getLogger(__name__)


class FFHQ(ImageDirectory):
    # pylint: disable=too-many-ancestors
    """An interface to the Flickr-Faces-HQ Dataset (FFHQ) dataset.
    The dataset consists of 70,000 aligned, high-resolution (1024x1024)
    face images crawled from Flickr.

    """

    def __init__(self, key: str = None, ffhq_meta: str = None,
                 **kwargs) -> None:
        """Initialize the ImageNet dataset.

        Parameters
        ----------
        imagenet_data: str
            The path to the FFHQ data directory. This directory
            should contain the 70,000 images, named from `'00000.png'`
            to `'69999.png'`.
            If no value is provided, the 'FFHQ_DATA' environment
            variable will be checked.
        """
        # FIXME[hack]:
        # directory = ffhq_data or os.getenv('FFHQ_DATA', '.')
        directory = config.data_directory / 'FFHQ' / 'images1024x1024'
        self._meta_filename = \
            ffhq_meta or config.data_directory / 'FFHQ' / 'ffhq-dataset-v2.json'
        description = 'FFHQ dataset'
        super().__init__(key=key or "ffhq",
                         directory=directory, suffix='png',
                         description=description, **kwargs)
        self._ffhq_meta = None
        LOG.info("Initialized FFHQ: %s", self.directory)

    #
    # Preparation
    #

    def _prepare(self, **kwargs) -> None:
        # pylint: disable=arguments-differ
        """Prepare the ImageNet data.
        """
        LOG.debug("Preparing FFHQ: %s", self.directory)

        filenames_cache = "ffhq_filelist.p"
        super()._prepare(filenames_cache=filenames_cache, **kwargs)

        # FIXME[bug]: when loading this (256M) file, the index navigation
        # (in the Resources panel) does not work anymore
        if self._meta_filename is not None and False:
            with open(self._meta_filename, 'r') as infile:
                self._ffhq_meta = json.load(infile)
                LOG.debug("Loaded FFHQ metadata from '%s'",
                          self._meta_filename)

    def _get_meta(self, data: Data, **kwargs) -> None:
        # pylint: disable=arguments-differ
        if self._ffhq_meta is not None:
            data.add_attribute('url', batch=True)
        super()._get_meta(data, **kwargs)

    def _get_data(self, data: Data, **kwargs) -> None:
        # pylint: disable=arguments-differ
        """
        Raises
        ------
        ValueError:
            If the index is present, but the :py:class:`DataFiles`
            has no filename register.
        """
        super()._get_data(data, **kwargs)
        if self._ffhq_meta is not None:
            data.url = self._ffhq_meta[data.filename]['metadata']['photo_url']

    #
    # The face alignment with FFHQ method has been taken from
    # https://gist.github.com/lzhbrian/bde87ab23b499dd02ba4f588258f57d5
    # author: lzhbrian (https://lzhbrian.me)
    #
    # FIXME[todo]: better toolbox integration

    @staticmethod
    def get_landmark(filename: str) -> np.ndarray:
        """Get face landmarks with dlib.

        Arguments
        ---------
        filename:

        Result
        ------
        landmarks:
            The detected landmarks, with shape=(68, 2).
        """
        # pylint: disable=invalid-name,import-outside-toplevel,no-member

        import dlib
        predictor_file = './shape_predictor_68_face_landmarks.dat'
        predictor = dlib.shape_predictor(predictor_file)
        detector = dlib.get_frontal_face_detector()

        img = dlib.load_rgb_image(filename)
        dets = detector(img, 1)

        print("Number of faces detected: {}".format(len(dets)))
        for _k, d in enumerate(dets):
            print("Detection {k}: Left: {d.left()} Top: {d.top()} "
                  "Right: {d.right()} Bottom: {d.bottom()}")
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            print(f"Part 0: {shape.part(0)}, Part 1: {shape.part(1)} ...")

        t = list(shape.parts())
        a = []
        for tt in t:
            a.append([tt.x, tt.y])
        lm = np.array(a)
        # lm is a shape=(68,2) np.array
        return lm

    @staticmethod
    def align_face(filename: str):
        """
        :param filename: str
        :return: PIL Image
        """
        # pylint: disable=invalid-name,import-outside-toplevel
        # pylint: disable=too-many-statements,too-many-locals
        import PIL
        import scipy

        lm = FFHQ.get_landmark(filename)

        # lm_chin          = lm[0:17]  # left-right
        # lm_eyebrow_left  = lm[17:22]  # left-right
        # lm_eyebrow_right = lm[22:27]  # left-right
        # lm_nose          = lm[27:31]  # top-down
        # lm_nostrils      = lm[31:36]  # top-down
        lm_eye_left = lm[36:42]  # left-clockwise
        lm_eye_right = lm[42:48]  # left-clockwise
        lm_mouth_outer = lm[48:60]  # left-clockwise
        # lm_mouth_inner   = lm[60:68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # read image
        img = PIL.Image.open(filename)

        output_size = 1024
        transform_size = 4096
        enable_padding = True

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)),
                     int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))),
                int(np.floor(min(quad[:, 1]))),
                int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0),
                max(crop[1] - border, 0),
                min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:, 0]))),
               int(np.floor(min(quad[:, 1]))),
               int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0),
               max(-pad[1] + border, 0),
               max(pad[2] - img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]),
                                           (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0],
                                               np.float32(w-1-x) / pad[2]),
                              1.0 - np.minimum(np.float32(y) / pad[1],
                                               np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img)\
                * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += \
                (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)),
                                      'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD,
                            (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        # Save aligned image.
        return img
