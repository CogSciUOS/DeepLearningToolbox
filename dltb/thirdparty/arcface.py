"""Arcface face recognition model. Based on the code of the
`arcface-tf2` project by Kuan-Yu Huang (peteryuX) [1].

The `arcface-tf2` repository provides code with model definitions and
functions for evaluation.  Links to four pretrained models are
provided on the website.  The models are trained on a 112x112 pixel
version the MS-Celeb-1M dataset [2], taken from the face.evoLVe
project [3].

This module provides easy access to the functionality of the
`arcface-tf2` repository.  Some parts of the code have been copied and
adapted for this module, while other parts are will call function from
the repository (given that the repository is available).


Usage
-----

This module provides the class `ArcFace` as central interface.  It
provides functionality for loading (pretrained) models, evaluation and
training.

.. code-block:: python

    from dltb.thirdparty.arcface import ArcFace
    from dltb.config import config

    arcface = ArcFace()
    image = config.github_directory / 'arcface-tf2' / 'data' / 'BruceLee.jpg'
    embedding = arcface.embed(str(image))


Training data
-------------

The `arcface-tf2` repository contains the scripts
`data/convert_train_tfrecord.py` and
`data/convert_train_binary_tfrecord.py` that convert the 112x112-pixel
version of the MS-Celeb-!M dataset from the face.evoLVe project
(`ms1m_align_112.zip`) into a TensorFlow `.tfrecord` files (a smaller
`ms1m.tfrecord` version containing only the filename, and a larger
version `ms1m_bin.tfrecord` containing also the image data.  No
further preprocessing is done by by those scripts (however, the files
in `ms1m_align_112.zip` have already undergone some preprocessing in
the face.evoLVe project, including alignment and cropping).

import tensorflow as tf

binary_flag = True

tfrecord_name = f'./data/ms1m{"_bin" if binary_flag else ""}.tfrecord'

raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
for tfrecord in raw_dataset.take(1):
    if binary_flag:
        features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                    'image/filename': tf.io.FixedLenFeature([], tf.string),
                    'image/encoded': tf.io.FixedLenFeature([], tf.string)}
        x = tf.io.parse_single_example(tfrecord, features)
        image_encoded = x['image/encoded']
    else:
        features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                    'image/img_path': tf.io.FixedLenFeature([], tf.string)}
        x = tf.io.parse_single_example(tfrecord, features)
        image_encoded = tf.io.read_file(x['image/img_path'])
    x_train = tf.image.decode_jpeg(image_encoded, channels=3)
    y_train = tf.cast(x['image/source_id'], tf.float32)


The module `modules.dataset` defines a `TFRecordDataset` that provides
these data after applying some further preprocessing (data augmentation).
To make use of this, the following functions can be used:

from modules.dataset import load_tfrecord_dataset

batch_size = 64
binary_flag = True
ccrop_flag = False

tfrecord_name = f'./data/ms1m{"_bin" if binary_flag else ""}.tfrecord'

train_dataset = load_tfrecord_dataset(
        tfrecord_name, batch_size, binary_img=binary_flag,
        is_ccrop=ccrop_flag)

number_of_batches = 4
for (x_train, _), y_train in train_dataset.take(number_of_batches):
    # x_train is tf.Tensor of image data, values in [0-epsilon, 1+epsilon]
    #    shape (batch_size, 112, 112, 3), dtype=float32,
    # y_train is tf Tensor containing the class labels
    #    shape (batch_size,), dtype=float32,
    # _ is just a copy of y_train (no idea why, seems simply redundant)

    # conversion to numpy array can be achieved as follows
    x_train_np = x_train.numpy()
    y_train_np = y_train.numpy()

References
----------

[1] https://github.com/peteryuX/arcface-tf2
[2] https://exposing.ai/msceleb/
[3] https://github.com/ZhaoJ9014/face.evoLVe

"""

# standard imports
from typing import Collection, Tuple, Optional
from argparse import ArgumentParser
from pathlib import Path
import os
import sys
import logging
import itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# FIXME[bug]: when running in script mode, the directory of this
# file is added to sys.path.  For some reason this will make importing
# opencv (cv2) fail!  However, when we remove this directory from
# sys.path, everything works well.

# FIXME[bug]: importing the dltb module here (which performs
# essentially the same cleanup, does not work!
# import dltb

# __file__ may be a relative filename (e.g., 'dltb/thirdparty/arcface.py')
print(f"__file__: '{__file__}'")
# then also Path(__file__) is a relative Path, however,
# Path(__file__).resolve() is guaranteed to be an absolute path.
print(f"Path(__file__): {Path(__file__)}")
print(f"Path(__file__).resolve(): {Path(__file__).resolve()}")

# Obtain root directory and the dltb directory
ROOT_DIRECTORY = Path(__file__).resolve().parents[2]
DLTB_DIRECTORY = ROOT_DIRECTORY / 'dltb'
print(ROOT_DIRECTORY)
print(DLTB_DIRECTORY)

print(sys.path)
sys.path = [directory for directory in sys.path
            if not directory.startswith(str(ROOT_DIRECTORY / 'dltb'))]
if str(DLTB_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ROOT_DIRECTORY))
print(sys.path)

# third party imports
import numpy as np
import cv2
import tensorflow as tf
import tqdm


# toolbox imports
from dltb.config import config
from dltb.base.image import Image, Imagelike
from dltb.base.prepare import Preparable
from dltb.util.importer import Importer
from dltb.tool.face.recognize import ArcFace as ArcFaceBase
from dltb.tool.face.landmarks import landmark_face_hack
from dltb.tool.align import LandmarkAligner

# logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

# typing
EvaluationResult = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


# Config1: arc_res50_small_ulf.yaml
Config1 = """# general
batch_size: 32
input_size: 112
embd_shape: 512
sub_name: 'arc_res50'
backbone_type: 'ResNet50' # 'ResNet50', 'MobileNetV2'
head_type: ArcHead # 'ArcHead', 'NormHead'
is_ccrop: False # central-cropping or not

# train
train_dataset: './data/ms1m_bin.tfrecord'
binary_img: True
num_classes: 85742
num_samples: 5822653
epochs: 5
base_lr: 0.01
w_decay: !!float 5e-4
save_steps: 1000

# test
test_dataset: '~/scratch/data/arcface2-tf/'
"""


class ArcFace(ArcFaceBase, Preparable):
    """TensorFlow ArcFace implementation.  The implementation is partly
    based on code from the `arcface-tf2` repository [1].

    This implementation uses the unofficial ArcFace-tf2 implementation
    from here [1].  To be able to run this code, it is necessary to clone
    that repository into a directory called `arcface-tf2` in the
    `config.github_directory`.  That code has futher third-party
    dependencies: `bcolz`, `tqdm`.

    The evaluation code in this class comes from this repository as
    well and was slightly adapted.  The evaluation is for the
    pair-matching task and takes a list of positive and negative pairs
    for each age group.

    Model weights for pretrained networks can be downloaded from
    GoogleDrive by links [2-5] provided in the README of [1].
    When unzipped, these files contain a directory with TensorFlow
    checkpoint files, from which the model can be loaded.

    In addition, training data (MS-1M [6]) are available for download.
    The training data should be downloaded to ms1m_align_112/imgs
    directory and then preprocessed using the script
    `data/convert_train_binary_tfrecord.py` or
    `data/convert_train_tfrecord.py`.  The data and data loader can be
    checked by runing the `dataset_checker.py` script.

    Also three validation datasets are available for download
    (LFW [7], AgeDB-30 [8] and CFP-FP [9]). These files are already in
    binary format, so they do not neet additional preprocessing.


    Properties
    ----------

    `arcface_code_directory`:
        Path to the `arcface-tf2` repository.
    `arcface_model_directory`:
        Path to the pretrained `arcface-tf2` models.
    `input_size`:
        The input size as a tuple (width, height).
    'embedding_dimensions':
        The dimensionality of the embedding space.

    References
    ----------

    [1] https://github.com/peteryuX/arcface-tf2
    [2] arc_res50.zip [299M]
        https://drive.google.com/file/d/1HasWQb86s4xSYy36YbmhRELg9LBmvhvt
    [3] arc_mbv2.zip [201M]
        https://drive.google.com/file/d/1qG8BChcPHzKuGwjJhrpeIxBqQmhpLvTX
    [4] arc_res50_ccrop.zip [299M]
        https://drive.google.com/file/d/1zUulC-4hSY_kPqZpcoIHO96OmjMivuKB
    [5] arc_mbv2_ccrop.zip [201M]
        https://drive.google.com/file/d/1nSnIc0eV0MkSjg48x29PJwTt3fGXKDU4
    [6] ms1m_align.zip [25G]
        https://drive.google.com/file/d/1WO5Meh_yAau00Gm2Rz2Pc0SRldLQYigT
    [7] lfw_align_112.zip [1.2G]
        https://drive.google.com/file/d/1WO5Meh_yAau00Gm2Rz2Pc0SRldLQYigT
    [8] agedb_align_112.zip [1G]
        https://drive.google.com/file/d/1AoZrZfym5ZhdTyKSxD0qxa7Xrp2Q1ftp
    [9] cfp_align_112.zip [2,3G]
        https://drive.google.com/file/d/1-sDn79lTegXRNhFuRnIRsgdU88cBfW6V

    """

    arcface_code_directory = config.github_directory / 'arcface-tf2'
    arcface_model_directory = config.model_directory / 'arcface-tf2'

    def __init__(self, cfg_path: str = None,  # './configs/arc_res50.yaml',
                 gpu: int = 0,
                 aligner: Optional[LandmarkAligner] = None, **kwargs) -> None:
        """
        Arguments
        ---------
        cfg_path:
            Path to the configuration file.
        gpu:
            Number of the GPU to use.
        """
        super().__init__(**kwargs)
        self._gpu = gpu
        self._cfg = None  # the arcface-tf2 configuration object

        # FIXME[hack]: the following two values should be obtained
        # from the model!
        self._input_size = (112, 112)
        self.embedding_dimensions = 512

        self._arcface_cfg_path = cfg_path
        self._arcface_model_backbone = 'ResNet50'  # or: 'MobileNetV2'
        self._arcface_model_name = 'arc_res50'

        self._arcface_model_checkpoints = \
            self.arcface_model_directory / self._arcface_model_name
        self._arcface_model = None

        # modules to be imported from the arcface-tf2 repository
        self._arcface_utils = None
        self._arcface_evaluations = None

        self._aligner = aligner

    @property
    def input_size(self) -> Tuple[int, int]:
        return self._input_size

    def _preparable(self) -> None:
        return (self.arcface_code_directory.is_dir() and
                self._arcface_model_checkpoints.is_dir() and
                super()._preparable())

    def _prepared(self) -> None:
        return (self._arcface_model is not None) and super()._prepared()

    def _prepare(self) -> None:
        # adapt the TensorFlow logging
        self._prepare_tensorflow()

        super()._prepare()

        # (0) provide further functionality from the arcface-tf repository
        self._arcface_utils = \
            Importer.import_module_from('modules.utils',
                                        directory=self.arcface_code_directory)
        self._arcface_evaluations = \
            Importer.import_module_from('modules.evaluations',
                                        directory=self.arcface_code_directory)

        # adapt the TensorFlow memory handling
        # The function essentially calls
        # tf.config.experimental.set_memory_growth(gpu, True)
        # Note that this must be done before GPUs have been initialized!
        self._arcface_utils.set_memory_growth()

        # load arcface model configuration file
        if self._arcface_cfg_path is not None:
            self._cfg = self._arcface_utils.load_yaml(self._arcface_cfg_path)
            self._input_size = (self._cfg['input_size'],
                                self._cfg['input_size'])

        # (1) Create the model
        modules_models = \
            Importer.import_module_from('modules.models',
                                        directory=self.arcface_code_directory)
        self._arcface_model = modules_models.\
            ArcFaceModel(size=self._input_size[1],  # becomes (size, size)
                         backbone_type=self._arcface_model_backbone,
                         training=False)
        #    ArcFaceModel(size=self._cfg['input_size'],
        #                 backbone_type=self._cfg['backbone_type'],
        #                 training=False)

        # (2) Load model weights from checkpoint
        ckpt_path = tf.train.latest_checkpoint(self._arcface_model_checkpoints)
        # ckpt_path = tf.train.latest_checkpoint('./checkpoints/' +
        #                                        self._cfg['sub_name'])
        if ckpt_path is None:
            LOG.error("Cannot load ckpt from '%s'.",
                      self._arcface_model_checkpoints)
            raise FileNotFoundError("Could not find checkpoint file "
                                    "for ArcFace model: "
                                    f"'{self._arcface_model_checkpoints}'")

        LOG.info("Load ArcFace-TF2 checkpoints from '%s'",
                 self._arcface_model_checkpoints)
        self._arcface_model.load_weights(ckpt_path)

        # (3) Adapt the aligner
        if self._aligner is not None:
            self._aligner.size = self.input_size

    def _unprepare(self) -> None:
        self._arcface_model = None
        super()._unprepare()

    #
    # Preprocessing
    #

    def _preprocess_image(self, image: Imagelike) -> np.ndarray:
        """Preprocess an image. Preprocessing includes (1) resizing to the
        desired input size, (2) conversion to float32 with range 0 to
        1, (3) reshaping to include a batch dimension.

        Arguments
        ---------
        image:
            The image to be preprocessed.

        Result
        ------
        preprocessed:
            The preprocessed image.
        """
        # based on other code, but not from arcface-tf2.
        img = Image.as_array(image)
        if len(image.shape) == 2:  # gray-scale
            img3 = np.empty((img.shape[0], img.shape[1], 3))
            img3[:, :, 0] = img
            img3[:, :, 1] = img
            img3[:, :, 2] = img
            img = img3
        if img.shape[:2] != self._input_size:
            img = cv2.resize(img, self._input_size)
        img = img.astype(np.float32) / 255.
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)
        return img

    def _align_image(self, image: Imagelike) -> np.ndarray:
        #image = Image.as_array(image)
        #print("image:", image.shape, image.dtype)
        image = Image(image)
        print("image:", image.array.shape, image.array.dtype)

        # bounding box and landmark for the (best) detection
        bbox, landmarks, _unique = \
            landmark_face_hack(image, self._aligner.detector)
        print(bbox)

        # an aligned version of the original image
        aligned_image = self._aligner(image, landmarks)

        # do further preprocessing
        aligned_image = aligned_image.astype(np.float32) / 255.
        aligned_image = np.expand_dims(aligned_image, 0)
        return aligned_image

    #
    # Embedding
    #

    def _process(self) -> None:
        # FIXME[todo]:
        """Dummy - check how to implement a _process method ...
        """

    def embed(self, image: Imagelike, is_ccrop=False, is_flip=True,
              outfile: str = None) -> np.ndarray:
        """Embed a given image into the embedding space.

        Apply the model to an image to compute an embedding and
        store the result in a file.

        """
        if self._aligner is not None:
            img = self._align_image(image)
        else:
            img = self._preprocess_image(image)
        # self._preprocess_image(image) -> seems to already add batch
        embeddings = self.embed_batch(img,  # np.expand_dims(img, 0)
                                      is_ccrop=is_ccrop,
                                      is_flip=is_flip)  # .squeeze(0)
        # embeddings is an tensorflow.EagerTensor
        #embeddings = embeddings.numpy().squeeze(0)
        if outfile is not None:
            LOG.info("Writing embeded image ot {%s}". outfile)
            np.save(outfile, embeddings)
        return embeddings

    def embed_batch(self, batch: np.ndarray,
                    is_ccrop=False, is_flip=True) -> np.ndarray:
        """Embed a batch of preprocessed images.

        Arguments
        ---------
        batch:
            A batch of image data, shape (112, 112, 3), RGB, np.float32,
            values range from 0.0 to 1.0.

        Arguments
        ---------
        batch:
            A batch of preprocessed images.  The batch is expected to
            be of shape (batch_size, width, height, 3), of dtype float32,
            in the range from 0 to 1 in RGB color space.

        Result
        ------
        embeddings:
            A batch of embeddings of shape (batch_size, embedding_dimensions).
        """
        if is_ccrop:
            batch = self._arcface_evaluations.ccrop_batch(batch)
        if is_flip:
            flipped = self._arcface_evaluations.hflip_batch(batch)
            emb_batch = \
                self._arcface_model(batch) + self._arcface_model(flipped)
        else:
            emb_batch = self._arcface_model(batch)

        return self._arcface_evaluations.l2_norm(emb_batch)

    def embed_labeled_pairs(self, pairs: Collection, progress=None
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """Embed images given as labeled image pairs.

        Arguments
        ---------
        pairs:
            An iterable providing labeled images pairs, that is triples
            of the form (image-1, image-2, issame).
        """
        embeddings = np.empty((len(pairs), 2, 512))
        issame = np.empty(len(pairs), dtype=int)

        # perform image-wise embedding.
        # FIXME[todo]: if possible, pair-wise embedding would be more efficient
        enumerated_pairs = enumerate(pairs)
        if progress is not None:
            enumerated_pairs = progress(enumerated_pairs)
        for idx, (img1, img2, same) in enumerated_pairs:
            out_1 = self.embed(img1)
            out_2 = self.embed(img2)

            embeddings[idx, 0] = out_1[0].numpy()
            embeddings[idx, 1] = out_2[0].numpy()
            issame[idx] = int(same)
        return embeddings, issame

    def embed_bcolz_carray(self, name: str, **kwargs,
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """

        name:
            The name of the bolz dataset

        kwargs:
            is_ccrop: bool = False, is_flip: bool = True,
            batch_size: int = 32, progress=None
        """
        carray, issame = self._load_bcolz_data(name)
        embeddings = self._embed_bcolz_data(carray, **kwargs)
        return (embeddings, issame)

    def embeddings_file_name(self, model_name: str, data_name: str,
                             is_ccrop: bool = False,
                             is_flip: bool = True) -> str:
        name = f'embeddings-{model_name}-{data_name}'
        name += '-ccrop' if is_ccrop else '-noccrop'
        name += '-flip' if is_flip else '-noflip'
        name += '.npz'
        return os.path.join(self.ARCFACE_TF2_EMBEDDINGS_DIR, name)

    #
    # Verifcation and recognition
    #

    def _distance(self, embedding1: np.ndarray,
                  embedding2: np.ndarray) -> float:
        """Compute distance between two embedding vectors.
        """
        # FIXME[hack]: should actually be cosine distance, shouldn't it?
        return (embedding1 - embedding2) ** 2

    def _distances(self, embeddings1: np.ndarray,
                   embeddings2: np.ndarray,
                   metric: str = 'squared_diff') -> np.ndarray:
        """Compute distances between two collections of embeddings.

        Arguments
        ---------
        embeddings1:
            A collection of embeddings.
        embeddings2:
            Another collection of embeddings, with the same shape
            as `embeddings2`.

        Result
        ------
        distances:
            A vector containing the elementwise distances of the
            elements of the two collections.
        """
        if embeddings1.shape != embeddings2.shape:
            raise ValueError("Embeddings do not match: "
                             f"{embeddings1.shape} vs. {embeddings2.shape}")
        if embeddings1.ndims != 2:
            raise ValueError("Embeddings have incorrect shape: "
                             f"{embeddings1.shape}")
        if embeddings1.shape[1] != self.embedding_dimensions:
            raise ValueError("Embeddings have incorrect dimensionality: "
                             f"{embeddings1.shape[1]}")

        if metric == 'squared_diff':
            # implementation from arcface-tf2, module
            # `modules.evaluations`, function `calculate_roc`:
            diff = np.subtract(embeddings1, embeddings2)
            distances = np.sum(np.square(diff), axis=1)
            # FIXME[hack]: should actually be cosine distance, shouldn't it?
        else:
            distances = np.ndarray(len(embeddings1), np.float)
            pairs = zip(embeddings1, embeddings2)
            for index, (embedding1, embedding2) in enumerate(pairs):
                distances[index] = self._distance(embedding1, embedding2)

        return distances

    #
    # Evaluation
    #

    # FIXME[todo]: different ways to provide batches/datasets of labeled pairs
    #  - Iterable[LabeledPair]
    #  - Tuple[Iterable[Image], Iterable[bool]] (len(images) == 2*len(labels))
    #  - Tuple[Iterable[Image], Iterable[Image], Iterable[bool]]
    #
    # Images can be:
    #  - Image file name
    #  - Index in some array (Image array, Embedding array)
    #     single index vs. double index convention
    #                      (index refers to a pair of two successive elements)
    #  - Image data
    #  - Image embeddings
    def _evaluate_embeddings(self, embeddings1: np.ndarray,
                             embeddings2: np.ndarray,
                             same: np.ndarray, folds=10):
        """

        Arguments
        ---------
        folds:
            The number of folds on which to do evaluation.
        """
        # distances = self._distances(embeddings1, embeddings2)
        # return self._evaluate_istances(distances, same)

    def evaluate_on_lfw(self, batch_size: int = 64, folds=10,
                        is_ccrop: bool = False, is_flip: bool = True,
                        progress=None) -> EvaluationResult:
        """Evaluate the model on the Labeled Faces in the Wild (LFW)
        dataset.
        """
        return self._evaluate_on_bcolz_data('lfw_align_112/lfw',
                                            batch_size=batch_size,
                                            is_ccrop=is_ccrop, is_flip=is_flip,
                                            progress=progress)

    def evaluate_on_agedb(self, batch_size: int = 64, folds=10,
                          is_ccrop: bool = False, is_flip: bool = True,
                          progress=None) -> EvaluationResult:
        """Evaluate the model on the AgeDB30 dataset.
        """
        return self._evaluate_on_bcolz_data('agedb_align_112/agedb_30',
                                            batch_size=batch_size,
                                            is_ccrop=is_ccrop, is_flip=is_flip,
                                            progress=progress)

    def evaluate_on_cfp(self, batch_size: int = 64, folds=10,
                        is_ccrop: bool = False, is_flip: bool = True,
                        progress=None) -> EvaluationResult:
        """Evaluate the model on the CFP-FP dataset.
        """
        return self._evaluate_on_bcolz_data('cfp_align_112/cfp_fp',
                                            batch_size=batch_size,
                                            is_ccrop=is_ccrop, is_flip=is_flip,
                                            progress=progress)

    def print_evaluation(self, message: str, results) -> None:
        """
        """
        tpr, fpr, accuracy, best_thresholds = results
        print(f"{message}: mean accuracy={accuracy.mean()*100:.2f}% "
              f"(tpr={tpr.mean():.3f}, fpr={fpr.mean():3f}) "
              f"with average threshold={best_thresholds.mean():.2}")

    def plot_evaluation(self, message: str, results) -> None:
        tpr, fpr, accuracy, best_thresholds = results
        thresholds = np.arange(0, 4, 0.01)
        import matplotlib.pyplot as plt  # FIXME[import]
        plt.figure()

        plt.subplot(2, 2, 1)
        plt.title("Accuracy")
        plt.bar(range(len(accuracy)), accuracy)

        plt.subplot(2, 2, 2)
        plt.title("Best threshold")
        plt.bar(range(len(best_thresholds)), best_thresholds)

        plt.subplot(2, 1, 2)
        plt.plot(thresholds, tpr)
        plt.plot(thresholds, fpr)
        plt.vlines(best_thresholds, 0, 1)

        plt.show()

        #
        # plot the differences
        #

    @staticmethod
    def show_diff(img1, img2):
        from matplotlib import pyplot as plt  # FIXME[import]

        diff = img1-img2
        diff_real = np.interp(diff, (-1, 1), (0, 1))
        diff_scaled = np.interp(diff, (diff.min(), diff.max()), (0, 1))

        plt.subplot(2, 2, 1)
        plt.title(f"Image 1: {img1.shape}, {img1.dtype}\n"
                  f"({img1.min()}-{img1.max()})")
        plt.imshow(img1)
        plt.subplot(2, 2, 2)
        plt.title(f"Image 2: {img2.shape}, {img2.dtype}\n"
                  f"({img2.min()}-{img2.max()})")
        plt.imshow(img2)
        plt.subplot(2, 2, 3)
        plt.imshow(diff_real)
        plt.subplot(2, 2, 4)
        plt.imshow(diff_scaled)
        plt.show()

    #
    # Training
    #

    def transform_image_batch(images) -> tf.Tensor:
        """Perform data transformation for training.  Images are enlarged to
        128x128 pixels and then randomly cropped, randomly left-right
        flipped, and randomly saturated and randomly brightness
        adapted.

        Arguments
        ---------
        images:
            A batch of images provided as a tensor of shape

        Result
        ------
        transformed:
            The transformed images (as float32 RBG images in the range 0-1).
        """
        images = tf.image.resize(images, (128, 128))
        images = tf.image.random_crop(images, (112, 112, 3))
        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_saturation(images, 0.6, 1.4)
        images = tf.image.random_brightness(images, 0.4)
        images = images / 255
        return images

    #
    # datasets
    #

    def load_lfw_as_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load the Labeled Faces in the Wild (LFW) dataset from
        the arcface-tf2 repository (originally from the face.evoLVe
        project) as numpy arrays.

        Result
        ------
        images:
            numpy array of shape (12000, 112, 112, 3) of dtype float32
            with values between 0.0 and 1.0, in RGB format.
        same:
            numpy array of shape (6000,) and dtype bool
        """
        bcolz_images, same = self._load_bcolz_data('lfw')
        images = bcolz_images[:, ::-1]   # bcolz -> numpy, BGR -> RGB
        images = np.transpose(images, [0, 2, 3, 1]) * 0.5 + 0.5
        return images, same

    def load_embeddings(filename) -> Tuple[np.ndarray, np.ndarray]:
        """

        Arguments
        ---------
        filename:
            Name of the NPZ archive containin the embedding
            informations (`'embeddings'`) along with the
            issame information (`'issame'`).
            There should should be two embeddings for each issame
            entry.

        Results
        -------
        embeddings:
            The embeddings as numpy array of shape (2*pairs,).
        issame:
            The embeddings as numpy array of shape (pairs,).
        """
        npzfile = np.load(filename)
        return npzfile['embeddings'], npzfile['issame']

    # ------------------------------------------------------------------------
    #
    # functionality utilizing code from the arcface-tf2 repository
    #
    # ------------------------------------------------------------------------

    def _prepare_tensorflow(self, aggressive: bool = False) -> None:
        """Prepare TensorFlow.
        """

        # Silence the TensorFlow logger
        logger = tf.get_logger()
        if aggressive:
            tf_cpp_min_log_level = '3'
            log_disabled = True
            log_level = logging.FATAL
        else:
            tf_cpp_min_log_level = '1'
            log_disabled = False
            log_level = logging.WARNING
        LOG.info("Changing TF_CPP_MIN_LOG_LEVEL from '%s' to '%s'",
                 os.environ['TF_CPP_MIN_LOG_LEVEL'], tf_cpp_min_log_level)
        LOG.info("Changing tensorflow logger from %s (%s) to %s (%s)",
                 logger.level, "disabled" if logger.disabled else "enabled",
                 log_level, "disabled" if log_disabled else "enabled")

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_cpp_min_log_level
        logger.disabled = log_disabled
        logger.setLevel(log_level)

        # CUDA_VISIBLE_DEVICES
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self._gpu)
        logger.setLevel(log_level)

    def _load_bcolz_data(self, name: str) -> Tuple:
        import bcolz  # FIXME[import]

        # code for loading one dataset (taken from arcface-tf2
        # repository, module `modules.evaluations`, function
        # `get_val_pair`):
        path = self._cfg['test_dataset']

        # carray is is a bcolz.carray of shape (data_size, 3, width, height)
        # with values between -1 and 1, and BGR (!) color coding.
        carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')

        # issame is a numpy.ndarray of shape (data_size//2,) of dtype bool
        issame = np.load('{}/{}_list.npy'.format(path, name))

        return carray, issame

    @staticmethod
    def _get_bcolz_image(carray, index: int) -> np.ndarray:
        """Pick an image from face.evoLVe archive (BGR, channel first, values
        from -1 to 1) and convert it into a numpy array in the
        standard RGB float format (RGB, channel last, values from 0 to
        1).

        """
        image = carray[index]
        image = np.transpose(image, (1, 2, 0))  # (C,H,W) -> (H,W,C)
        image = image * 0.5 + 0.5  # [-1,1] -> [0,1]
        image = image[:, :, ::-1]  # BGR -> RGB
        return image

    def _evaluate_on_bcolz_data(self, name: str, batch_size: int = 64,
                                is_ccrop: bool = False, is_flip: bool = True,
                                progress=None) -> EvaluationResult:
        """Load data from bcolz carray (that is prepacked face.evoLVe format,
        like in the archive `'lfw_align_112.zip'`), compute
        embeddings and evaluate it.

        """
        carray, issame = self._load_bcolz_data(name)

        # obtain embeddings for the image data in carray:
        embeddings = self._embed_bcolz_data(carray, batch_size=batch_size,
                                            is_ccrop=is_ccrop, is_flip=is_flip,
                                            progress=progress)

        return self.evaluate_embeddings(embeddings[0::2], embeddings[1::2],
                                        issame)

    def _embed_bcolz_data(self, carray, batch_size: Optional[int] = None,
                          is_ccrop: bool = False, is_flip: bool = True,
                          progress=None) -> np.ndarray:
        """Embed images provided as bcolz carray.

        Arguments
        ---------
        """
        # code for embedding the carray (taken from arcface-tf2
        # repository, module `modules.evaluations`, function
        # `perform_val`):
        embeddings = np.zeros([len(carray), self.embedding_dimensions])

        batch_size = batch_size or self._cfg['batch_size']
        batch_indices = range(0, len(carray), batch_size)
        if progress is not None:
            batch_indices = progress(batch_indices)
        for idx in batch_indices:
            batch = carray[idx:idx + batch_size]
            batch = np.transpose(batch, [0, 2, 3, 1]) * 0.5 + 0.5
            batch = batch[:, :, :, ::-1]  # convert BGR to RGB

            if is_ccrop:
                batch = self._arcface_evaluations.ccrop_batch(batch)
            if is_flip:
                flipped = self._arcface_evaluations.hflip_batch(batch)
                embeddings[idx:idx + batch_size] = \
                    self._arcface_model(batch) + self._arcface_model(flipped)
            else:
                embeddings[idx:idx + batch_size] = self._arcface_model(batch)
        return self._arcface_utils.l2_norm(embeddings)

    def evaluate_embeddings(self, embeddings1: np.ndarray,
                            embeddings2: np.ndarray,
                            same: np.ndarray, folds=10):
        """Perform evaluation based on a pair of embedding vectors
        and ground truth information.

        This function essentially calls
        `modules.evaluations.calculate_roc`.

        Arguments
        ---------
        embeddings1:
            A collection of embeddings.
        embeddings2:
            Another collection of embeddings, with the same shape
            as `embeddings2`.

        Result
        ------
        tpr:
            The true positive rate. This is a vector of length 400,
            containing the true positive rate for every tested threshold
            from 0.0 to 4.0, with a step size of 0.1.
        fpr:
            The false positive rate. Same shape as `tpr`.
        accuracy:
            The accuracy values for all folds. This is a vector
            of length `folds`.
        best_thresholds:
            The threshold resulting in the best scores for each fold.
        """
        # distances = self._distances(embeddings1, embeddings2)
        # return self._evaluate_istances(distances, same)
        thresholds = np.arange(0, 4, 0.01)
        tpr, fpr, accuracy, best_thresholds = \
            self._arcface_evaluations.calculate_roc(
                thresholds, embeddings1, embeddings2, same,
                nrof_folds=folds)

        return tpr, fpr, accuracy, best_thresholds

    def evaluate_labeled_pairs(self, pairs: Collection,
                               progress=None, folds: int = 10) -> float:
        """Evaluate the model on given labeled image pairs.

        Arguments
        ---------
        pairs:
            An iterable providing labeled images pairs, that is triples
            of the form (image-1, image-2, issame).

        Result
        ------
        tpr, fpr, accuracies, best_thresholds:
        """
        embeddings, issame = self.embed_labeled_pairs(pairs, progress=progress)
        return self.evaluate_embeddings(embeddings[::2], embeddings[1::2],
                                        issame, folds=folds)

    def evaluate_old(self, lfw: bool = False, agedb: bool = False,
                     cfp: bool = False) -> None:
        """Evaluate the module on different datasets, using the
        function `modueles.evaluations.perform_val` from the
        arcface-tf2 repository.
        """
        # FIXME[bug]: self._cfg may be None!
        print("[*] Loading LFW, AgeDB30 and CFP-FP...")
        lfw, agedb_30, cfp_fp, lfw_issame, agedb_30_issame, cfp_fp_issame = \
            self._arcface_evaluations.get_val_data(self._cfg['test_dataset'])

        if lfw:
            print("[*] Perform Evaluation on LFW...")
            acc_lfw, best_th = self._arcface_evaluations.perform_val(
                self._cfg['embd_shape'], self._cfg['batch_size'],
                self._arcface_model, lfw, lfw_issame,
                is_ccrop=self._cfg['is_ccrop'])
            print("    acc {:.4f}, th: {:.2f}".format(acc_lfw, best_th))

        if agedb:
            print("[*] Perform Evaluation on AgeDB30...")
            acc_agedb30, best_th = self._arcface_evaluations.perform_val(
                self._cfg['embd_shape'], self._cfg['batch_size'],
                self._arcface_model, agedb_30, agedb_30_issame,
                is_ccrop=self._cfg['is_ccrop'])
            print("    acc {:.4f}, th: {:.2f}".format(acc_agedb30, best_th))

        if cfp:
            print("[*] Perform Evaluation on CFP-FP...")
            acc_cfp_fp, best_th = self._arcface_evaluations.perform_val(
                self._cfg['embd_shape'], self._cfg['batch_size'],
                self._arcface_model, cfp_fp, cfp_fp_issame,
                is_ccrop=self._cfg['is_ccrop'])
            print("    acc {:.4f}, th: {:.2f}".format(acc_cfp_fp, best_th))

    def arcface_tf2_evaluate(self, embeddings: np.ndarray, issame: np.ndarray,
                             folds: int = 10):
        """Evaluate embeddings using the function
        `modules.evaluations.evaluate` from the arcface repository.

        Arguments
        ---------
        embeddings:
            The embeddings vector of shape (2*pairs,).
        issame:
            The issame information as vector of shape (pairs,).
        folds:
            The numbers of folds to be used for evaluation.
        """
        tpr, fpr, accuracy, best_thresholds = \
            self._arcface_evaluations.evaluate(embeddings, issame, folds)
        return accuracy.mean(), best_thresholds.mean()

    def evaluate_old2(self, lfw: bool = False, agedb: bool = False,
                      cfp: bool = False) -> None:
        """Evaluate the module on different datasets, using the
        function `modueles.evaluations.perform_val` from the
        arcface-tf2 repository.
        """
        print("[*] Loading LFW, AgeDB30 and CFP-FP...")
        lfw_images, agedb_images, cfp_images, \
            lfw_issame, agedb_issame, cfp_issame = \
            self._arcface_evaluations.get_val_data(self._cfg['test_dataset'])

        is_ccrop = self._cfg['is_ccrop']

        if lfw:
            print("[*] Perform Evaluation on LFW...")
            embeddings = \
                self._embed_bcolz_data(lfw_images, is_ccrop=is_ccrop)
            acc_lfw, best_th = self.arcface_tf2_evaluate(embeddings, lfw_issame)
            print(f"    acc {acc_lfw:.4f}, th: {best_th:.2f}")

        if agedb:
            print("[*] Perform Evaluation on AgeDB30...")
            embeddings = \
                self._embed_bcolz_data(agedb_images, is_ccrop=is_ccrop)
            acc_agedb, best_th = \
                self.arcface_tf2_evaluate(embeddings, agedb_issame)
            print(f"    acc {acc_agedb:.4f}, th: {best_th:.2f}")

        if cfp:
            print("[*] Perform Evaluation on CFP-FP...")
            embeddings = \
                self._embed_bcolz_data(cfp_images, is_ccrop=is_ccrop)
            acc_cfp, best_th = \
                self.arcface_tf2_evaluate(embeddings, cfp_issame)
            print(f"    acc {acc_cfp:.4f}, th: {best_th:.2f}")

    def evaluate_old3(self, lfw: bool = False, agedb: bool = False,
                      cfp: bool = False,
                      batch_size: Optional[int] = None) -> None:
        embedding_size = self.embedding_dimensions
        batch_size = batch_size or self._cfg['batch_size']

        print("[*] Loading LFW, AgeDB30 and CFP-FP...")
        lfw_images, agedb_30_images, cfp_fp_images, \
            lfw_issame, agedb_30_issame, cfp_fp_issame = \
            self._arcface_evaluations.get_val_data(self._cfg['test_dataset'])

        datasets = []
        if lfw:
            datasets.append(('LFW', lfw_images, lfw_issame))
        if agedb:
            datasets.append(('AgeDB30', agedb_30_images, agedb_30_issame))
        if cfp:
            datasets.append(('CFP-FP', cfp_fp_images, cfp_fp_issame))
        for (name, images, issame) in datasets:
            images = images[:, ::-1]  # Convert BGR -> RGB
            print(images.dtype, images.shape, images.min(), images.max())
            print(issame.dtype, issame.shape, issame.min(), issame.max())
            for ccrop, flip in itertools.product((False, True), (False, True)):
                print(f"[*] Perform Evaluation on {name} "
                      f"[ccrop={ccrop}, flip={flip}]... ")
                acc_lfw, best_th = \
                    self._arcface_evaluations.perform_val(embedding_size,
                                                          batch_size,
                                                          self._arcface_model,
                                                          images, issame,
                                                          is_ccrop=ccrop,
                                                          is_flip=flip)
                print(f"  -> Results on {name} [ccrop={ccrop}, flip={flip}]:  "
                      f"acc {acc_lfw:.4f}, th: {best_th:.2f}\n\n")

    def old_experiment(self, image_dir,
                       batch_size: Optional[int] = None) -> None:
        """Load a dataset (image pairs and boolean same information) as numpy
        arrays and feed them to the arcface-tf2 evaluation function

        """

        #
        # load the data
        #


        print(f"Loading images and same from '{LFW_DATA_DIR}' "
              "(as 'lfw_images.npy' and 'lfw_same.npy')")

        # images: float32 (12000, 3, 112, 112) -1.0 1.0 (BGR!)
        # issame: bool (6000,) False True
        images = np.load(os.path.join(LFW_DATA_DIR, "lfw_images.npy"))
        issame = np.load(os.path.join(LFW_DATA_DIR, "lfw_same.npy"))
        print(images.shape, images.dtype)
        images = images.astype(np.float32) * 2/ 255. - 1
        images = np.transpose(images, (0, 3, 1, 2))
        print(images.shape, images.dtype, images.min(), images.max())
        print(issame.shape, issame.dtype)
        #issame = np.logical_not(issame)

        if False:
            from matplotlib import pyplot as plt
            # -> shape=(4, 112, 112, 3)
            batch = np.transpose(images, [0, 2, 3, 1])
            batch = batch * 0.5 + 0.5
            plt.imshow(batch[0, :, :, ::-1])  # Bilder sind BGR nicht RGB!
            plt.show()

        #
        # Experiment:
        #

        embedding_size = self.embedding_dimensions
        batch_size = batch_size or self._cfg['batch_size']
        images = images[:, ::-1]  # Convert BGR -> RGB
        ccrop, flip = False, True
        acc_lfw, best_th = \
            self._arcface_evaluations.perform_val(embedding_size,
                                                  batch_size,
                                                  self._arcface_model,
                                                  images, issame,
                                                  is_ccrop=ccrop, is_flip=flip)
        print(f"  -> Results [ccrop={ccrop}, flip={flip}]:  "
              f"acc {acc_lfw:.4f}, th: {best_th:.2f}\n\n")

    #
    # tensorflow related code
    #

    def _tensorflow_helper(self) -> None:
        print("Running tool:")
        print("=============")
        print(f"Python {'.'.join(str(v) for v in sys.version_info)}")
        print(f"TensorFlow {tf.__version__}")
        # print(f"model_name = '{model_name}'")
        # print(f"data_name = '{data_name}'")

    #
    # face.evoLVe related code
    #

    # Load data in face.evoLVe archive format:
    # Shape: (batch, channels, height, width),  usually size = (112, 112)
    # Color: BGR
    # min/max: -1/1

    face_evolve_code_directory = config.github_directory / 'face.evoLVe'

    # ARCFACE_TF2_DATADIR = 'test_data'
    ARCFACE_TF2_DATADIR = '/space/data/face.evoLVe'
    ARCFACE_TF2_EMBEDDINGS_DIR = '/space/home/ulf/arcface-tf2/embeddings'

    DATASETS = {
        'lfw': {
            'archive': 'lfw_align_112.zip',
            'folder': 'lfw_align_112/lfw',
            'pairs': 6000
        },
        'agedb': {
            'archive': 'lfw_align_112.zip',
            'folder': 'lfw_align_112/lfw',
            'pairs': 6000
        },
        'cfp': {
            'archive': 'lfw_align_112.zip',
            'folder': 'lfw_align_112/lfw',
            'pairs': 7000
        },
    }

    def misc(self) -> None:
        assert self.face_evolve_code_directory.is_dir(), \
            f"No face.evoLVe directory '{self.face_evolve_code_directory}'"


# == old-1 ===================================================================

# A test script to evaluate the performance of pretrained models

# this requires:
#  - bcolz=1.2.1   (for reading the test data)
#  - tqdm          (for showing progress bars)
#
# Furthermore:
#  - a config file defining network parameters
#  - (pretrained) model weights (checkpoints)
#  - the validation data: LFW, AgeDB, and CFP
#    the path to the validation data is given in the config file
#

# =============================================================================



# =============================================================================

def _prepare_parser() -> ArgumentParser:
    parser = ArgumentParser(description="ArcFace-TF2 script.")
    group1 = parser.add_argument_group("Commands")
    group1.add_argument('--embed', action='store_true', default=False,
                        help="compute embedding(s) for input file(s)")
    group1.add_argument('--evaluate', action='store_true', default=False,
                        help="evaluate arcface-tf2")
    group1.add_argument('--compare', action='store_true', default=False,
                        help="compare two versions of an embedding")
    group1.add_argument('--old1', action='store_true', default=False,
                        help="run old program 1: evaluate arcface-tf2")
    group1.add_argument('--eval-old', action='store_true', default=False,
                        help="evaluate the model using the old "
                        "evaluation function")
    group1.add_argument('--eval-old2', action='store_true', default=False,
                        help="evaluate the model using the old "
                        "evaluation function")
    group1.add_argument('--eval-old3', action='store_true', default=False,
                        help="evaluate the model using the old "
                        "evaluation function")
    group1.add_argument('--old-experiment', action='store_true', default=False,
                        help="run the old experiment to evaluate the model")
    group1.add_argument('--batch_size', type=int, default=32,
                        help="batch size for processing data")
    group1.add_argument('--ccrop', action='store_true', default=False,
                        help="apply center cropping during evaluation")
    group1.add_argument('--flip', action='store_true', default=True,
                        help="apply (horizontal) flipping during evaluation")

    group2 = parser.add_argument_group("Files and directories")
    group2.add_argument('--arcface-tf2',
                        default=ArcFace.arcface_code_directory,
                        help="path to the arcface-tf2 repository")
    group2.add_argument('--lfw-images', default=None,
                        help="path to the LFW data directory "
                        "(containing the image files)")
    group2.add_argument('--lfw-pairs', default=None,
                        help="the LFW pairs to be used")

    group3 = parser.add_argument_group("Datasets")
    group3.add_argument('--lfw', action='store_true', default=False,
                        help="Use the LFW dataset")
    group3.add_argument('--agedb', action='store_true', default=False,
                        help="Use the AgdDB30 dataset")
    group3.add_argument('--cfp', action='store_true', default=False,
                        help="Use the CFP-FP dataset")

    group4 = parser.add_argument_group("Output")
    group4.add_argument('--progress', action='store_true', default=False,
                        help="show progress bar during operation")
    group4.add_argument('--plot', action='store_true', default=False,
                        help="plot results")


def main() -> None:
    """The main program.  Process command line options, setup the
    environment and invoke the desired functions.

    """
    parser = _prepare_parser()
    args = parser.parse_args()

    cfg_path = ArcFace.arcface_code_directory / 'configs' / 'ulf_test.yaml'
    arcface = ArcFace(cfg_path=cfg_path)

    progress = tqdm.tqdm if args.progress else None

    if args.evaluate:
        if args.lfw:
            results = arcface.evaluate_on_lfw(progress=progress,
                                              batch_size=args.batch_size,
                                              is_ccrop=args.ccrop,
                                              is_flip=args.flip)
            arcface.print_evaluation("LFW evaluation results", results)
            if args.plot:
                arcface.plot_evaluation("LFW evaluation results", results)
        if args.agedb:
            results = arcface.evaluate_on_agedb(progress=progress,
                                                batch_size=args.batch_size,
                                                is_ccrop=args.ccrop,
                                                is_flip=args.flip)
            arcface.print_evaluation("AgeDB-30 evaluation results", results)
            if args.plot:
                arcface.plot_evaluation("AgeDB-30 evaluation results", results)
        if args.cfp:
            results = arcface.evaluate_on_cfpw(progress=progress,
                                               batch_size=args.batch_size,
                                               is_ccrop=args.ccrop,
                                               is_flip=args.flip)
            arcface.print_evaluation("CFP-FP evaluation results", results)
            if args.plot:
                arcface.plot_evaluation("CFP-FP evaluation results", results)

    if args.embed:
        model_name = 'arc_res50'
        data_name = 'lfw'
        is_ccrop = args.ccrop
        is_flip = args.flip

        embedding_file_name = \
            arcface.embeddings_file_name(model_name, data_name,
                                         is_ccrop=is_ccrop, is_flip=is_flip)

        folder = ArcFace.DATASETS[data_name]['folder']
        print(f"[*] Loading LFW from '{folder}' "
              f"(extracted from {ArcFace.DATASETS[data_name]['archive']}:")
        _data, issame = get_val_pair(ArcFace.ARCFACE_TF2_DATADIR, folder)

        if os.path.isfile(embedding_file_name):
            # if the embeddings file already exists, just load the
            # embeddings and do some sanity checks
            npzfile = np.load(embedding_file_name)
            embeddings, issame = npzfile['embeddings'], npzfile['issame']
            print(f"Loaded embeddings of shape {embeddings.shape} from "
                  f"'{embedding_file_name}'")
            assert embeddings.ndim == 2
            assert issame.ndim == 1
            assert len(embeddings) == 2 * len(issame)
        else:
            # if the embeddings file does not yet exist, compute them
            # embeddings and store them to the file
            from tensorflow.python.framework.errors_impl \
                import ResourceExhaustedError  # FIXME[import]

            try:
                embeddings = arcface.embed_bcolz_carray(data_name,
                                                        is_ccrop=is_ccrop,
                                                        is_flip=is_flip)
            except ResourceExhaustedError:
                print("Resources exhausted. "
                      "Try to run with smaller batch size!")
                sys.exit(1)

            print(f"Writing embeddings of shape {embeddings.shape} to "
                  f"'{embedding_file_name}'")
            np.savez_compressed(embedding_file_name,
                                embeddings=embeddings, issame=issame)

        # Evaluation
        tpr, fpr, accuracy, best_thresholds = \
            arcface.arcface_tf2_evaluate(embeddings, issame)
        print("tpr:", tpr.shape)
        print("fpr:", fpr.shape)
        print(f"accuracy[{accuracy.mean()}]:", accuracy)
        print(f"best_thresholds[{best_thresholds.mean()}]:", best_thresholds)

        # import matplotlib.pyplot as plt
        # plt.hist(accuracy)
        # plt.show()

    if args.compare:
        # Compare two versions of an embedding, either using different
        # preprocessing, or different embedding options (ccrop, flip).
        model_name = 'arc_res50'
        data_name = 'lfw'

        filename1 = \
            arcface.embeddings_file_name(model_name, data_name,
                                         is_ccrop=True, is_flip=True)
        filename2 = \
            arcface.embeddings_file_name(model_name, data_name,
                                         is_ccrop=False, is_flip=True)

        embeddings1, issame1 = arcface(filename1)
        print(f"Loaded embeddings of shape {embeddings1.shape} from "
              f"'{filename1}'")
        assert embeddings1.ndim == 2
        assert issame1.ndim == 1
        assert len(embeddings1) == 2 * len(issame1)

        embeddings2, issame2 = arcface(filename2)
        print(f"Loaded embeddings of shape {embeddings2.shape} from "
              f"'{filename2}'")
        assert embeddings2.ndim == 2
        assert issame2.ndim == 2
        assert len(embeddings2) == 2 * len(issame2)

    if args.compare2:
        #
        # load the data (distributed with face_evolve "lfw_align_112.zip")
        #
        print("[*] Loading LFW from 'lfw_align_112.zip':")
        lfw1, lfw1_issame = arcface.load_lfw_as_numpy()
        print(lfw1.shape, lfw1.dtype, lfw1.min(), lfw1.max())
        print(lfw1_issame.shape, lfw1_issame.dtype,
              lfw1_issame.min(), lfw1_issame.max())

        assert lfw1.shape == (12000, 112, 112, 3)
        assert lfw1.dtype == np.float32

        assert lfw1_issame.shape == (6000, )
        assert lfw1_issame.dtype == np.bool

        #
        # load the data (manually aligned with face_evolve)
        #
        LFW_DATA_DIR = "~/scratch/krumnack/data"

        print(f"[*] Loading images and same from '{LFW_DATA_DIR}' "
              "(as 'lfw_images.npy' and 'lfw_same.npy')")

        # images: float32 (12000, 3, 112, 112) -1.0 1.0 (BGR!)
        lfw2 = np.load(os.path.join(LFW_DATA_DIR, "lfw_images.npy"))

        # issame: bool (6000,) False True
        lfw2_issame = np.load(os.path.join(LFW_DATA_DIR, "lfw_same.npy"))

        print(lfw2.shape, lfw2.dtype, lfw2.min(), lfw2.max())
        # lfw2 = lfw2.astype(np.float32) / 255.
        # lfw2 = lfw2[:, :, :, ::-1]  # convert RGB to BGR
        lfw2 = lfw2.astype(np.float32) * 2 / 255. - 1
        lfw2 = np.transpose(lfw2, (0, 3, 1, 2))

        # lfw2 = np.transpose(lfw2, [0, 2, 3, 1]) * 0.5 + 0.5
        # lfw2 = lfw2[:, :, :, ::-1]  # convert BGR to RGB
        print(lfw2.shape, lfw2.dtype, lfw2.min(), lfw2.max())
        print(lfw2_issame.shape, lfw2_issame.dtype,
              lfw2_issame.min(), lfw2_issame.max())
        # issame = np.logical_not(issame)

        assert lfw1.shape == lfw2.shape
        assert lfw1.dtype == lfw2.dtype
        assert np.array_equal(lfw1_issame, lfw2_issame)

        #
        # plot the differences
        #

        index = 0
        ArcFace.show_diff(lfw1[index], lfw2[index])

        #
        # Experiment:
        #
        _tpr_lfw1, _fpr_lfw1, acc_lfw1, best_th1 = \
            arcface.evaluate_labeled_pairs(zip(lfw1[::2], lfw1[1::2],
                                               lfw1_issame),
                                           ccrop=args.ccrop, flip=args.flip)
        print(f"  -> Results LFW1 [ccrop={args.ccrop}, flip={args.flip}]:  "
              f"acc {acc_lfw1:.4f}, th: {best_th1:.2f}\n\n")

        _tpr_lfw2, _fpr_lfw2, acc_lfw2, best_th2 = \
            arcface.evaluate_labeled_pairs(zip(lfw2[::2], lfw2[1::2],
                                               lfw2_issame),
                                           ccrop=args.ccrop, flip=args.flip)
        print(f"  -> Results LFW2 [ccrop={args.ccrop}, flip={args.flip}]:  "
              f"acc {acc_lfw2:.4f}, th: {best_th2:.2f}\n\n")

    if args.eval_old:
        arcface.evaluate_old(lfw=args.lfw, agedb=args.agedb, cfp=args.cfp)

    if args.eval_old2:
        arcface.evaluate_old2(lfw=args.lfw, agedb=args.agedb, cfp=args.cfp)

    if args.eval_old3:
        arcface.evaluate_old3(lfw=args.lfw, agedb=args.agedb, cfp=args.cfp)

    if args.old_experiment:
        # a directory holding the LFW dataset in form of two numpy files
        # (one holding the images, the other holding the "same" information)
        LFW_DATA_DIR = ("/net/projects/scratch/winter/valid_until_31_July_2022"
                        "/krumnack/data")
        arcface.old_experiment(LFW_DATA_DIR)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Program interrupted. Good bye!")
