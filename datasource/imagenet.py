"""ImageNet (ILSVRC2012) related data structures.

"""

# standard imports
import os
import re
import random
import logging

# third party imports
import numpy as np

# toolbox imports
from .data import Data, ClassScheme
from .datasource import Imagesource, Sectioned
from .directory import DataDirectory

# logging
LOG = logging.getLogger(__name__)


class ImagenetScheme(ClassScheme):
    """The ImageNet classification scheme.
    ImageNet (ILSVRC2012) introduces 1,000 classes. Classes are
    identified with WordNet synsets (representing word meaning),
    allowing to map the a class to a textual description.

    Unfortunately, there are different convention on how to label
    the ILSVRC2012 dataset.
    The natural convention is to use Synsets, either in form of the
    Synset ID (format='synset', e.g., "n01440764"), a long textual form
    listing all wordforms of the synset (format='text_long',
    e.g. "tench, Tinca tinca"), a short textual form containing just
    the first wordform (format='text', e.g. "tench",
    possibly containing spaces for multiword expressions like "bell pepper").
    However, for neural networks some numerical form is needed.
    Here different forms are in used. The ILSVRC2012 dataset comes with
    a 1-based indexing (format="ilsvrc") given by the file "classes.txt"
    from the imagenet_data directory.  The file is expected to contain mapping
    of labels to synset names and text representation for labels:

      1:n02119789:kit fox, Vulpes macrotis
      2:n02100735:English setter
      ...
      1000:n03255030:dumbbell

    An alternative way of indexing the ILSVRC2012 class is used by
    Caffe: this approach simply sorts the synset IDs and indexes them
    starting from 0
    (format='caffe', maps 0 to n01440764 "tench, Tinca tincan").
    Torch seems to use the same scheme as caffe (redundantly provided
    by format='torch').

    The toolbox provides the file 'assets/imagenet_labels.txt' with
    mappings between all these different labels.


    See Also
    --------
    * http://caffe.berkeleyvision.org/gathered/examples/imagenet.html
    * https://github.com/pytorch/vision/issues/484

    """

    def __init__(self, *args, **kwargs) -> None:
        """
        """
        super().__init__(length=1000, *args, **kwargs)

    @property
    def prepared(self) -> bool:
        """Check if the :py:class:`ImagenetScheme` has been initialized.
        """
        return 'synset' in self._labels

    def prepare(self) -> None:
        """Prepare the labels for the ImageNet dataset.
        The labels will be read in from the file 'imagenet_labels.txt'
        contained in the `assets` folder of the toolbox.
        """
        if self.prepared:
            return  # nothing to do ...

        imagenet_labels_filename = \
            os.path.join('assets', 'imagenet_labels.txt')
        ids = []
        caffe_ids = []
        torch_ids = []
        synsets = []
        wordforms = []
        wordforms_long = []

        try:
            with open(imagenet_labels_filename) as file:
                for line in file.readlines():
                    fields = line.strip().split(':')
                    ids.append(int(fields[0]))
                    torch_ids.append(int(fields[1]))
                    caffe_ids.append(int(fields[2]))
                    synsets.append(fields[3])
                    wordforms.append(fields[4])
                    wordforms_long.append(fields[5])

            self.add_labels(np.asarray(ids), 'imagenet', True)
            self.add_labels(np.asarray(caffe_ids), 'caffe', True)
            self.add_labels(np.asarray(torch_ids), 'torch', True)
            self.add_labels(synsets, 'synset', True)
            self.add_labels(wordforms, 'text')
            self.add_labels(wordforms_long, 'text_long')

        except FileNotFoundError:
            LOG.warning("ImageNet labels files not found. "
                        "Make sure that '%s' is available.",
                        imagenet_labels_filename)

        LOG.info("ImageNet is now prepared: %d", len(self))

    def _read_classes_txt(self, classes_txt: str):
        """Prepare the labels for the ImageNet Datasource. This will
        read in the file "classes.txt" from the imagenet_data directory.
        The file is expected to contain mapping of labels to synset names
        and text representation for labels,

          1:n02119789:kit fox, Vulpes macrotis
          2:n02100735:English setter
          ...
          1000:n03255030:dumbbell

        Usually it is not necessary to call this method, as the
        information from "classes.txt" is incorporated into the
        toolbox file "assets/imagenet_labels.txt" which is read in
        by default.

        classe_txt: str
            The (fully qualified) name of the file classes.txt
            (i.e,. "${IMAGENET_DATA}/classes.txt").
        """
        try:
            with open(classes_txt) as file:
                classes = file.readlines()
            texts = [c.strip().split(':')[2] for c in classes]
            self.add_labels('text', texts)
        except FileNotFoundError:
            LOG.error("ImageNet class names not found. "
                      "Make sure that '%s' is available.", classes_txt)
            raise


class ImageNet(DataDirectory, Imagesource, Sectioned,
               sections={'train', 'val', 'test'}):
    # pylint: disable=too-many-ancestors
    """An interface to the ILSVRC2012 dataset.

    **Labels**

    The label information is obtained in different ways, depending on
    the section (train/val/test) of the dataset.

    In the 'train' section, files are grouped in subdirectories,
    one subdirectory per class. Directories are named by WordNet 3.0
    synset identifier. To obtain the actual (numeric) label
    (in the range 0-999), the class stores a mapping
    :py:attr:`_wn_table` mapping synset names to labels.

    In the 'val' section, all 50.000 files are store in one directory,
    with filenames providing a numeric identifier (in the range 1-50,000).
    A mapping from these numeric identifiers to 1-based labels (1-1000)
    is provided by the file 'val_labels.txt'. This information is hold in
    the attribute :py:attr:`_val_labels` in a slightly modified form:
    image identifiers and labels are both shifted to be 0-based, i.e.,
    this list contains at index i the information for image id=i+1,
    and the value obtained is the 0-based numerical label (0-999)

    Attributes
    ----------

    _imagenet_data: str
        The path to the ImageNet root directory. This is the directory
        that contains 'train/', 'val/', and 'test/' subdirectories.
        It may also contain the ILSVRC2012_devkit_t12 which provides
        some metadata, allowing to map the (numeric) validation labels
        from 'val_labels.txt' to synsets.

    _val_labels: list[int]
        A list assigning labels to the images from the validation set.
        This is only initialized for the 'val' section.  It is read
        in from the file ''ILSVRC2012_validation_ground_truth.txt'
        in the 'imagenet_data/ILSVRC2012_devkit_t12/data/' directory.
        Notice, that the number from the filename is 1-based, while
        this mapping is 0-based, and also the class numbers in the
        file are 1-based while out internal representation is 0-based,
        so _val_labels actually maps
        (number_from_filename - 1) -> (class_number - 1).
    """

    def __init__(self, key: str = None, section: str = 'val',
                 imagenet_data: str = None, **kwargs) -> None:
        """Initialize the ImageNet dataset.

        Parameters
        ----------
        imagenet_data: str
            The path to the imagenet root directory. This directory
            should contain the 'train', 'val', and 'test' subdirectories.
            If no value is provided, the 'IMAGENET_DATA' environment
            variable will be checked.
        section: str
            The section of ImageNet to provide by this Datasource object.
            One of 'train', 'val', 'test'.
        """
        description = f"ImageNet {section}"
        self._imagenet_data = imagenet_data or os.getenv('IMAGENET_DATA', '.')
        directory = os.path.join(self._imagenet_data, section)
        scheme = ClassScheme.register_initialize_key('ImageNet')
        super().__init__(key=key or f"imagenet-{section}", section=section,
                         directory=directory,
                         description=description,
                         scheme=scheme,
                         label_from_directory='synset',
                         **kwargs)
        self._val_labels = None

    def __str__(self):
        return f"ImageNet ({self._section})"

    #
    # Preparation
    #

    def _preparable(self):
        """Check if this Datasource is can be prepared.

        Returns
        -------
        preparable: bool
            True if the Datasource can (probably) be prepared, False otherwise.
        """
        return os.path.isdir(self.directory) and super()._preparable()

    def _prepare(self, **kwargs) -> None:
        # pylint: disable=arguments-differ
        """Prepare the ImageNet data.
        """
        LOG.info("Preparing ImageNet (%s): %s", self._section, self.directory)

        filenames_cache = f"imagenet_{self._section}_filelist.p"
        super()._prepare(filenames_cache=filenames_cache, **kwargs)

        # prepare the ImagenetScheme
        self._scheme.prepare()

        if self._section == 'train':
            self._prepare_labels_train()
        elif self._section == 'val':
            self._prepare_labels_val()
        elif self._section == 'test':
            self._prepare_labels_test()

    def _prepare_labels_train(self) -> None:
        """Prepare the image to label mapping for the ILSVRC2012
        training set. This mapping can be achieved as the training
        files are grouped in directories named by the respective
        synset.
        """
        # nothing to do - labels can be directly obtained from filenames

    def _prepare_labels_val(self, val_labels: str = None) -> None:
        """Load the image to label mapping for the ILSVRC2012
        validation set. This mapping is provided in the file
        `data/ILSVRC2012_validation_ground_truth.txt`
        as part of the 'ILSVRC2012_devkit_t12'.
        If you have not installed that devkit, you may provide
        this file as `val_labels.txt` in the imagenet_data
        directory.
        """
        if val_labels is None:
            val_labels = \
                os.path.join(self._imagenet_data, 'ILSVRC2012_devkit_t12',
                             'data', 'ILSVRC2012_validation_ground_truth.txt')
            if not os.path.isfile(val_labels):
                val_labels = \
                    os.path.join(self._imagenet_data, 'val_labels.txt')
        try:
            with open(val_labels) as file:
                self._val_labels = [int(l.strip())-1 for l in file.readlines()]
                LOG.info("val_labels: %d", len(self._val_labels))
        except FileNotFoundError:
            LOG.warning("ImageNet validation labels not found, "
                        "will provide unlabeled data."
                        "Make sure that '%s' is available.", val_labels)

    def _prepare_labels_test(self) -> None:
        """Prepare the labels for the ILSVRC2012 test data.
        Note: ILSVRC2012 test set labels are not publicly available,
        so we cannot provide them here. If you use the ILSVRC2012 test data,
        this will be an unlabeled dataset.

        Note: in practice, many people use the ILSVRC2012 validation set
        instead of the test set to report their results.
        """
        # nothing to do - we have no data

    #
    # Data
    #

    def _get_label_for_filename(self, data: Data, filename: str) -> None:
        """Get the (internal) label for a given filename from the ImageNet
        dataset.  The way to determine the label depends on what section
        of the dataset ('train', 'val', or 'test') we are working on and
        requires additional data.  If this data is not available,
        the method will simply return None.

        Parameters
        ----------
        filename: str
            The filename of the image, relative to the base directory
            imagenet_data/{train,val,test}/. That is for example
            'n01440764/n01440764_3421.JPEG' in case of 'train' or
            'ILSVRC2012_val_00031438.JPEG' in case of 'val' images.

        Returns
        ------
        label: int
            A number from 0-999 indicating the class to which the image
            belongs or None if no class could be determined.
        """
        label = None

        if self._section == 'train':
            super()._get_data_from_file(data, filename)
        elif self._section == 'val' and self._val_labels is not None:
            match_object = re.search('ILSVRC2012_val_([0-9]*).JPEG', filename)
            if match_object is not None:
                label = self._val_labels[int(match_object.group(1))-1]
                data.label = self._scheme.identifier(label)

    def _get_random(self, data: Data, category: int = None, **kwargs) -> None:
        # pylint: disable=arguments-differ
        """Randomly fetch an image from this ImageNet datasource.  After
        fetching, the image can be obtained by calling
        :py:meth:get_data() and the the label by calling
        :py:meth:get_label().
        """
        if category is not None and self._section == 'train':
            subdir = self._scheme.get_label(category, 'synset')
            img_dir = os.path.join(self.directory, subdir)
            filename = random.choice(os.listdir(img_dir))
            self._get_data_from_file(data, os.path.join(subdir, filename))
        else:
            super()._get_random(data, **kwargs)
