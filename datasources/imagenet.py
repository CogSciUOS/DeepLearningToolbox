import os
import random
from scipy.misc import imread, imresize
from datasources.imagenet_classes import class_names

from datasources import DataSource, DataDirectory, InputData, Predefined

# FIXME[todo]: this is currently just a hack, not a full-fleshed
# DataSource!
#  - implement the standard DataSource API
#  - allow for all "sections" (train/valid/test)
#  - automatically provide teh ImageNet class labels


class ImageNet(DataDirectory, Predefined):

    _prefix = None
    _section = None

    _section_ids = ['train', 'test', 'val']
    
    def __init__(self, prefix=None, section='train'):
        self._section = section
        super(ImageNet, self).__init__()
        Predefined.__init__(self, 'imagenet')

    def prepare(self):
        dir = os.path.join(os.getenv('IMAGENET_DATA', '.'), self._section)
        self.setDirectory(dir)
        print(dir)


    def get_section_ids(self):
        '''Get a list of sections provided by this data source.  A data source
        may for example be divided into training and test data.
        '''
        return self._section_ids
    
    def get_section(self):
        '''Get the current section in the data source.
        '''
        return self.section

    def set_section(self, section_id):
        '''Set the current section in the data source.
        '''
        self.section = section_id
        self.prepare()

class ImageNet2(DataSource, Predefined):

    _prefix = None
    _available = False
    _category = None
    _image = None

    def __init__(self, prefix=None, section='train'):
        self._prefix = os.path.join(os.getenv('IMAGENET_DATA', '.'), section)
        self._available = os.path.isfile(self._prefix)
        Predefined.__init__(self, 'imagenet')

    def __len__(self):
        '''Get the number of entries in this data source.'''
        return 1000

    def random(self):
        self._category = random.choice(os.listdir(self._prefix))
        img_dir = os.path.join(self._prefix, self.category)
        self._image = random.choice(os.listdir(img_dir))
        img_file = os.path.join(img_dir, self._image)
        im = imread(img_file)
        return InputData(im,img_file)

    def check_availability():
        '''Check if this Datasource is available.
        
        Returns
        -------
        True if the DataSource can be instantiated, False otherwise.
        '''
        return self._prefix

    def download():
        raise NotImplementedError("Downloading ImageNet is not implemented yet")

