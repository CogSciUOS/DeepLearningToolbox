# -----------------------------------------------------------------------------
# File: face_preprocess.py
"""This code is take from

git checkout 07f6547
src/common/face_preprocess.py

I have added some comments and docstrings to this code (which are
completely missing in the original).

"""

from typing import Tuple
import cv2
from skimage import transform as trans
import numpy as np

# toolbox imports
from dltb.base.image import Image, Imagelike, Sizelike


class FaceAligner:

    @staticmethod
    def parse_lst_line(line):
        """Parse alignment information from a text line (tabulator separated
        values). The line should contain either 3, 7 or 17 values.
        The first 3 values specify (aligned, image_path, label),
        the (numeriic) aligned flag, the path to the image file
        and a numerical label.  This can be followed in fields 3 to 6 by
        four integer coordinates for the bouding boxes, and then in
        fields 7-16 by ten coordinates for the facial landmarks.

        Arguments
        ---------
        line:

        Result
        ------
        image_path: str
        label: int
        bbox: np.ndarray of shape (4,), dtype np.int32
        landmark: np.ndarray of shape (2, 5), dtype np.float32
        aligned: int
        """
        vec = line.strip().split("\t")
        assert len(vec) >= 3
        aligned = int(vec[0])
        image_path = vec[1]
        label = int(vec[2])
        bbox = None
        landmark = None
        # print(vec)
        if len(vec) > 3:
            bbox = np.zeros((4,), dtype=np.int32)
            for i in range(3, 7):
                bbox[i-3] = int(vec[i])
            landmark = None

        # optional: coordinates for 5 landmarks
        if len(vec) > 7:
            coordinates = []
            for i in range(7, 17):
                coordinates.append(float(vec[i]))
            landmark = np.array(coordinates).reshape((2, 5)).T
        return image_path, label, bbox, landmark, aligned

    @staticmethod
    def read_image(img_path, **kwargs):
        """Read an image from a file
        """
        mode = kwargs.get('mode', 'rgb')
        layout = kwargs.get('layout', 'HWC')
        if mode == 'gray':
            img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        else:
            img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_COLOR)
            if mode == 'rgb':
                # print('to rgb')
                img = img[..., ::-1]  # BGR -> RGB
            if layout == 'CHW':
                img = np.transpose(img, (2, 0, 1))
        return img

    def preprocess(self, image: Imagelike, size: Tuple[int, int] = None,
                   bbox=None, landmark=None,
                   margin: int = 0, **kwargs):  # margin=44
        """Preprocess the image. Preprocessing consists of multiple steps:
        1. read the image
        2. obtain the target image size
        3. align the image

        Arguments
        ---------
        image:
            The image to be preprocessed.
        size:
            The target size of the image after preprocessing.
        bbox:
            The bounding for the image.
        landmarks:
            Facial landmarks for face alignment.
        margin:
            Extra margin to put around the face.
        """
        #
        # 1. read the image
        #
        img = Image.as_array(image)

        #
        # 2. obtain the target image size
        #
        
        # str_image_size = image_size
        # image_size = []  # image_size as two-element list [width, height]
        # if str_image_size:
        #    image_size = [int(x) for x in str_image_size.split(',')]
        #    if len(image_size) == 1:
        #        image_size = [image_size[0], image_size[0]]
        if size is None:
            image_size = (112, 112)
        else:
            image_size = size

        assert len(image_size) == 2
        assert image_size[0] == 112
        assert image_size[0] == 112 or image_size[1] == 96

        #
        # 3. align the image
        #

        # obtain a transformation matrix
        transformation = landmark and self._transformation_matrix(landmark)

        # if no transformation was obtained, just resize
        if transformation is None:
            return self._resize_image(img, image_size, margin=margin)

        # otherweise apply the transformation
        return self._transform_image(img, transformation, image_size)

    @staticmethod
    def _transformation_matrix(landmarks, size):
        """

        size:
            The size of the target image.  Only two sizes are supported:
            (112, 1112) or (112, 96).
        """
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if size[1] == 112:
            src[:, 0] += 8.0
        dst = landmarks.astype(np.float32)

        # src = src[0:3,:]
        # dst = dst[0:3,:]

        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        trans_matrix = tform.params[0:2, :]
        # trans_matrix = \
        #     cv2.estimateRigidTransform(dst.reshape(1,5,2),
        #                                src.reshape(1,5,2), False)

        # print(src.shape, dst.shape)
        # print(src)
        # print(dst)
        # print(trans_matrix)

        return trans_matrix

    @staticmethod
    def _resize_image(image: np.ndarray, image_size, bbox=None,
                      margin: int = 44):
        """
        Arguments
        ---------
        image:
            The image to be resized.
        size:
            The new size of the image.
        bbox:
            BoundingBox (x1, y1, x2, y2) to cut out of the input image.
            Only that part of the input image will be resized.
            If the (extended - see margin) bounding box reaches out
            of the image boundaries, it will adapted so that it is
            inside the image (this may distort the aspect ratio).
        margin:
            An extra margin by which the bounding box is extended
            on all sides (half of the margin is added left/rigth and
            top/bottom).
        """
        # no transformation: use bounding box for resizing
        if bbox is None:  # use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(image.shape[1]*0.0625)
            det[1] = int(image.shape[0]*0.0625)
            det[2] = image.shape[1] - det[0]
            det[3] = image.shape[0] - det[1]
        else:
            det = bbox
        bbox = np.zeros(4, dtype=np.int32)
        bbox[0] = np.maximum(det[0]-margin/2, 0)
        bbox[1] = np.maximum(det[1]-margin/2, 0)
        bbox[2] = np.minimum(det[2]+margin/2, image.shape[1])
        bbox[3] = np.minimum(det[3]+margin/2, image.shape[0])
        ret = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret

    @staticmethod
    def _transform_image(image: np.ndarray,
                         transformation,
                         size: Sizelike) -> np.ndarray:
        """

        Arguments
        ---------
        image:
            The image to be transformed.
        transformation:
            An affine transformation matrix.
        size:
            The size of the target image.
        """
        warped = cv2.warpAffine(image, transformation,
                                (size[1], size[0]),
                                borderValue=0.0)
        # tform3 = trans.ProjectiveTransform()
        # tform3.estimate(src, dst)
        # warped = trans.warp(image, tform3, output_shape=_shape)
        return warped
