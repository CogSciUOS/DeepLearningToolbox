"""Predefined Datasources.
"""

from .datasource import Datasource

Datasource.add_module_requirement('datasource.webcam', 'cv2')


Datasource.register_key('mnist-train', 'datasource.keras',
                        'KerasDatasource', name='mnist', section='train')
Datasource.register_key('mnist-test', 'datasource.keras',
                        'KerasDatasource', name='mnist', section='test')
Datasource.register_key('imagenet-val', 'datasource.imagenet',
                        'ImageNet', section='val')  # section='train'
Datasource.register_key('dogsandcats', 'datasource.dogsandcats',
                        'DogsAndCats')
Datasource.register_key('widerface', 'datasource.widerface', 'WiderFace')
Datasource.register_key('Helen', 'datasource.helen', 'Helen')
Datasource.register_key('Noise', 'datasource.noise', 'Noise',
                        shape=(100, 100, 3))
Datasource.register_key('Webcam', 'datasource.webcam', 'DataWebcam')

Datasource.register_key('Movie', 'datasource.video', 'Video',
                        filename='/pub/ulf/media/music/Laibach/'
                        'Laibach - God is God.mp4')
Datasource.register_key('dummy', 'datasource.dummy', 'Dummy')
Datasource.register_key('5celeb', 'datasource.fivecelebface', 'FiveCelebFace')


Datasource.register_class('WiderFace', 'datasource.widerface')
