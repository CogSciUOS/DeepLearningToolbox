import os, random
from scipy.misc import imread, imresize
from datasources.imagenet_classes import class_names

from datasources import DataSource, InputData

# FIXME[todo]: this is currently just a hack, not a full-fleshed
# DataSource!
#  - implement the standard DataSource API
#  - allow for all "sections" (train/valid/test)
#  - automatically provide teh ImageNet class labels


class ImageNet(DataSource):

    _prefix = None

    def __init__(self, prefix=None, section='train'):
        self._prefix = os.path.join(os.getenv('IMAGENET_DATA', '.'), section)

    def random(self):
        img_dir = os.path.join(self._prefix,
                               random.choice(os.listdir(self._prefix)))
        img_file = os.path.join(img_dir, random.choice(os.listdir(img_dir)))
        im = imread(img_file)
        return InputData(im,img_file)
        
