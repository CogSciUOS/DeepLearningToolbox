"""Predefined Datasources.
"""

# standard imports
import sys
import logging

# toolbox imports
from dltb.datasource import Datasource

Datasource.register_instance('mnist-train', 'datasource.keras',
                             'KerasDatasource', name='mnist',
                             section='train')
Datasource.register_instance('mnist-test', 'datasource.keras',
                             'KerasDatasource', name='mnist',
                             section='test')
Datasource.register_instance('cifar10-train', 'datasource.keras',
                             'KerasDatasource', name='cifar10',
                             section='train')

Datasource.register_instance('imagenet-val', 'datasource.imagenet',
                             'ImageNet', section='val')  # section='train'
Datasource.register_instance('dogsandcats', 'datasource.dogsandcats',
                             'DogsAndCats')
Datasource.register_instance('widerface', 'datasource.widerface', 'WiderFace')
Datasource.register_instance('fgnet', 'datasource.fgnet', 'FGNet')
Datasource.register_instance('Helen', 'datasource.helen', 'Helen')
Datasource.register_instance('lfw', 'datasource.lfw', 'LabeledFacesInTheWild')
Datasource.register_instance('ms-celeb-1m', 'datasource.face', 'MSCeleb1M')
Datasource.register_instance('Noise', 'dltb.datasource.noise', 'Noise',
                             shape=(100, 100, 3))

# FIXME[problem]: it seems to be problematic to use two webcams
# at the same time. It may help to reduce the resolution or
# to connect the webcams to different USB ports:
# import cv2
# cap0 = cv2.VideoCapture(0)
# cap0.set(3,160)
# cap0.set(4,120)
# cap1 = cv2.VideoCapture(1)
# cap1.set(3,160)
# cap1.set(4,120)
# ret0, frame0 = cap0.read()
# assert ret0 # succeeds
# ret1, frame1 = cap1.read()
# assert ret1 # fails?!
#
# import imageio
# reader0 = image.get_reader('<video0>', size=(160, 120))
# reader1 = image.get_reader('<video1>', size=(160, 120))
# frame0 = reader0.get_next_data()
# frame1 = reader1.get_next_data()
Datasource.register_instance('Webcam', 'dltb.datasource.webcam', 'DataWebcam')
Datasource.register_instance('Webcam2', 'dltb.datasource.webcam', 'DataWebcam',
                             device=1)

Datasource.register_instance('Movie', 'dltb.datasource.video', 'Video',
                             filename='/pub/ulf/media/music/Laibach/'
                             'Laibach - God is God.mp4')
Datasource.register_instance('dummy', 'dltb.datasource.dummy', 'Dummy')
Datasource.register_instance('5celeb', 'datasource.fivecelebface',
                             'FiveCelebFace')


Datasource.register_class('WiderFace', 'dltb.datasource.widerface')


# logging
LOG = logging.getLogger(__name__)
del logging

LOG.info("Predefined data sources: %r",
         list(Datasource.instance_register.keys()))

# FIXME[old]: may be used at some other place ...
# for keras_id in KerasDatasource.KERAS_IDS:
#    try:
#        KerasDatasource(keras_id)
#    except ValueError as err:
#        LOG.error("Error instantiating keras data source '%s': %s",
#                  keras_id, err)
