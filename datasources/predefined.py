from .source import Datasource

Datasource.register('mnist-train', 'datasources.keras',
                    'KerasDatasource', name='mnist', section='train')
Datasource.register('mnist-test', 'datasources.keras',
                    'KerasDatasource', name='mnist', section='test')
Datasource.register('imagenet-val', 'datasources.imagenet',
                    'ImageNet', section='val')  # section='train'
Datasource.register('dogsandcats', 'datasources.dogsandcats',
                    'DogsAndCats')
Datasource.register('widerface', 'datasources.widerface', 'WiderFace')
Datasource.register('Helen', 'datasources.helen', 'Helen')
Datasource.register('Noise', 'datasources.noise', 'DataNoise',
                    shape=(100,100,3))
Datasource.register('Webcam', 'datasources.webcam', 'DataWebcam')
