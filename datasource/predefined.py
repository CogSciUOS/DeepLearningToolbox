from .datasource import Datasource

Datasource.register('mnist-train', 'datasource.keras',
                    'KerasDatasource', name='mnist', section='train')
Datasource.register('mnist-test', 'datasource.keras',
                    'KerasDatasource', name='mnist', section='test')
Datasource.register('imagenet-val', 'datasource.imagenet',
                    'ImageNet', section='val')  # section='train'
Datasource.register('dogsandcats', 'datasource.dogsandcats',
                    'DogsAndCats')
Datasource.register('widerface', 'datasource.widerface', 'WiderFace')
Datasource.register('Helen', 'datasource.helen', 'Helen')
Datasource.register('Noise', 'datasource.noise', 'DataNoise',
                    shape=(100,100,3))
Datasource.register('Webcam', 'datasource.webcam', 'DataWebcam')
