"""Predefined Datasources.
"""

from .datasource import Datasource

Datasource.add_module_requirement('datasource.webcam', 'cv2')


Datasource.register_key('mnist-train', 'datasource.keras',
                        'KerasDatasource', name='mnist', section='train')
Datasource.register_key('mnist-test', 'datasource.keras',
                        'KerasDatasource', name='mnist', section='test')
Datasource.register_key('cifar10-train', 'datasource.keras',
                        'KerasDatasource', name='cifar10', section='train')

Datasource.register_key('imagenet-val', 'datasource.imagenet',
                        'ImageNet', section='val')  # section='train'
Datasource.register_key('dogsandcats', 'datasource.dogsandcats',
                        'DogsAndCats')
Datasource.register_key('widerface', 'datasource.widerface', 'WiderFace')
Datasource.register_key('fgnet', 'datasource.fgnet', 'FGNet')
Datasource.register_key('Helen', 'datasource.helen', 'Helen')
Datasource.register_key('Noise', 'datasource.noise', 'Noise',
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
Datasource.register_key('Webcam', 'datasource.webcam', 'DataWebcam')
Datasource.register_key('Webcam2', 'datasource.webcam', 'DataWebcam',
                        device=1)

Datasource.register_key('Movie', 'datasource.video', 'Video',
                        filename='/pub/ulf/media/music/Laibach/'
                        'Laibach - God is God.mp4')
Datasource.register_key('dummy', 'datasource.dummy', 'Dummy')
Datasource.register_key('5celeb', 'datasource.fivecelebface', 'FiveCelebFace')


Datasource.register_class('WiderFace', 'datasource.widerface')
